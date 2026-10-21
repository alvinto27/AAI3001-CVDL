"""API-only server: python -m cvdl.pipeline.serve --host 127.0.0.1 --port 8000."""
import argparse
from contextlib import asynccontextmanager
import logging
import os
import tempfile
import threading
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from . import InputError, InputTooLarge, UnsupportedInput, load_pipeline
from .inputs import MAX_ENCODED_BYTES

LOG=logging.getLogger(__name__)


def _error(state,status,message):
    return JSONResponse(dict(error=dict(status=status,message=message),bundle=state.bundle),status_code=status)


class _BodyAdmission:
    """Bound/admit the raw body BEFORE Starlette's multipart parser sees it.

    The spool is closed after response completion/disconnection. The model and
    its identity are retained; uploads and generated results are not.
    """
    def __init__(self,app,*,state,limit):
        self.app=app;self.state=state;self.limit=limit

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method']!='POST' or scope['path'].rstrip('/')!='/analyze':
            return await self.app(scope,receive,send)
        if not self.state.ready:
            return await _error(self.state,503,'model is not ready')(scope,receive,send)
        if not self.state.analysis_lock.acquire(blocking=False):
            return await _error(self.state,429,'another analysis is in progress')(scope,receive,send)
        try:
            headers=dict(scope.get('headers',[]))
            declared=headers.get(b'content-length')
            if declared is not None:
                try:
                    length=int(declared)
                    if length<0:raise ValueError
                except ValueError:
                    return await _error(self.state,422,'invalid request length')(scope,receive,send)
                if length>self.limit:
                    return await _error(self.state,413,'request exceeds 64 MiB')(scope,receive,send)
            with tempfile.SpooledTemporaryFile(max_size=1024*1024,mode='w+b') as spool:
                total=0
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect':return
                    if message['type']!='http.request':continue
                    chunk=message.get('body',b'');total+=len(chunk)
                    if total>self.limit:
                        return await _error(self.state,413,'request exceeds 64 MiB')(scope,receive,send)
                    await run_in_threadpool(spool.write,chunk)
                    if not message.get('more_body',False):break
                await run_in_threadpool(spool.seek,0)
                exhausted=False
                async def replay():
                    nonlocal exhausted
                    if exhausted:return await receive()
                    chunk=await run_in_threadpool(spool.read,64*1024)
                    more=spool.tell()<total
                    if not more:exhausted=True
                    return dict(type='http.request',body=chunk,more_body=more)
                await self.app(scope,replay,send)
        finally:
            self.state.analysis_lock.release()


class _IdentityCORS(CORSMiddleware):
    def __init__(self,*args,state,**kwargs):
        super().__init__(*args,**kwargs);self.state=state

    def preflight_response(self,request_headers):
        response=super().preflight_response(request_headers)
        return JSONResponse(dict(bundle=self.state.bundle,cors='allowed' if response.status_code==200 else 'denied'),
            status_code=response.status_code,headers={k:v for k,v in response.headers.items() if k.lower() not in ('content-type','content-length')})


def _origins(values):
    result=[]
    for value in values:
        value=value.rstrip('/')
        parsed=urlsplit(value)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password or '*' in value:
            raise ValueError('allow-origin must be an explicit http(s) origin, not a wildcard')
        result.append(value)
    return result


def _boolean(form,key):
    value=form.get(key,'false')
    if not isinstance(value,str) or value.lower() not in ('true','false','1','0'):
        raise ValueError('options must be true or false')
    return value.lower() in ('true','1')


def create_app(bundle_dir=None,*,device='cpu',batch_size=8,allowed_origins=(),max_request_bytes=MAX_ENCODED_BYTES):
    """One immutable bundle per process. Optional dependencies stay in this module."""
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.pipeline=await run_in_threadpool(load_pipeline,bundle_dir,device=device,batch_size=batch_size)
            app.state.bundle=app.state.pipeline.identity
            app.state.ready=True
        except Exception as exc:
            # The health route remains available. Never leak load paths to clients.
            LOG.error('Model startup failed (%s)',type(exc).__name__)
        try:yield
        finally:
            app.state.ready=False
            app.state.pipeline=None

    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.pipeline=None;app.state.bundle=None;app.state.ready=False
    app.state.analysis_lock=threading.Lock()
    app.add_middleware(_BodyAdmission,state=app.state,limit=max_request_bytes)
    app.add_middleware(_IdentityCORS,state=app.state,allow_origins=_origins(allowed_origins),
        allow_methods=['GET','POST'],allow_headers=['Content-Type'],allow_credentials=False)

    @app.exception_handler(HTTPException)
    async def http_error(request,exc):
        status=422 if exc.status_code==400 else exc.status_code
        return _error(app.state,status,'invalid request' if status==422 else 'request not supported')

    @app.exception_handler(Exception)
    async def unexpected_error(request,exc):
        LOG.error('Request failed (%s)',type(exc).__name__)
        return _error(app.state,503,'request is unavailable')

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return _error(app.state,422,'invalid request fields')

    @app.get('/health')
    async def health():
        pipeline=app.state.pipeline
        return JSONResponse(dict(ready=app.state.ready,bundle=app.state.bundle,device=pipeline.device if pipeline else None),
            status_code=200 if app.state.ready else 503)

    @app.post('/analyze')
    async def analyze(request:Request):
        if not request.headers.get('content-type','').lower().startswith('multipart/form-data;'):
            return _error(app.state,415,'multipart photo upload required')
        try:
            async with request.form(max_files=1,max_fields=2,max_part_size=max_request_bytes) as form:
                if set(form)-{'photo','include_images','allow_rgba'} or any(len(form.getlist(k))!=1 for k in form):
                    return _error(app.state,422,'invalid or repeated form fields')
                photo=form.get('photo')
                if not isinstance(photo,UploadFile):return _error(app.state,422,'one photo file is required')
                include_images=_boolean(form,'include_images');allow_rgba=_boolean(form,'allow_rgba')
                data=await photo.read()
                result=await run_in_threadpool(app.state.pipeline.analyze,data,
                    include_images=include_images,allow_rgba=allow_rgba)
                return JSONResponse(result)
        except InputTooLarge:
            return _error(app.state,413,'photo exceeds the size limit')
        except UnsupportedInput:
            return _error(app.state,415,'unsupported photo format or pixel representation')
        except (InputError,ValueError):
            return _error(app.state,422,'invalid photo or form options')
        except HTTPException:
            return _error(app.state,422,'invalid multipart upload')
        except Exception as exc:
            LOG.error('Analysis failed (%s)',type(exc).__name__)
            return _error(app.state,503,'analysis is unavailable')

    return app


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--bundle')
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--allow-origin',action='append',default=[])
    args=parser.parse_args(argv)
    import torch
    import uvicorn
    threads=os.environ.get('CVDL_THREADS')
    if threads is not None:
        try:
            count=int(threads)
            if count<1:raise ValueError
        except ValueError:
            parser.error('CVDL_THREADS must be a positive integer')
        torch.set_num_threads(count)
    app=create_app(args.bundle,device=args.device,
        allowed_origins=args.allow_origin or [v for v in os.environ.get('CVDL_ALLOWED_ORIGINS','').split(',') if v])
    uvicorn.run(app,host=args.host,port=args.port,workers=1,access_log=False)


if __name__=='__main__':main()
