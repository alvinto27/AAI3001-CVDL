"""Optional API tests; the training environment needs no HTTP dependencies."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from PIL import Image

try:
    import fastapi
    import httpx
    import python_multipart
    from fastapi.testclient import TestClient
    from cvdl.pipeline.serve import create_app
    HTTP_AVAILABLE=True
except (ImportError,RuntimeError):
    HTTP_AVAILABLE=False


def photo(format='PNG',size=(256,256),mode='RGB'):
    buffer=io.BytesIO();Image.new(mode,size,color=0).save(buffer,format=format);return buffer.getvalue()


@unittest.skipUnless(HTTP_AVAILABLE,'optional HTTP dependencies (fastapi/httpx/python-multipart) not installed')
class PipelineHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from cvdl.pipeline import load_pipeline
        cls.pipeline=load_pipeline(device='cpu')

    def client(self,**kwargs):
        app=create_app(**kwargs)
        # The real bundle/model is shared by test cases; each app still loads
        # it through the startup loader exactly once per lifespan.
        loader=patch('cvdl.pipeline.serve.load_pipeline',return_value=self.pipeline)
        self.addCleanup(loader.stop);self.loader=loader.start()
        return TestClient(app)

    def assert_identity(self,response):
        self.assertIn('bundle',response.json())
        self.assertEqual(response.json()['bundle'],self.pipeline.identity)
        self.assertNotIn('/work',response.text)

    def test_health_and_one_startup_load(self):
        with self.client() as client:
            for _ in range(2):
                response=client.get('/health');self.assertEqual(response.status_code,200)
                self.assertTrue(response.json()['ready']);self.assertEqual(response.json()['device'],'cpu')
                self.assert_identity(response)
            client.post('/analyze',files={'photo':('p.png',photo(),'image/png')})
            self.assertEqual(self.loader.call_count,1)
        self.assertFalse(client.app.state.ready)
        self.assertIsNone(client.app.state.pipeline)

    def test_png_identity_images_off_and_on(self):
        with self.client() as client:
            plain=client.post('/analyze',files={'photo':('p.png',photo(),'image/png')})
            self.assertEqual(plain.status_code,200);self.assert_identity(plain)
            self.assertNotIn('images',plain.json())
            rich=client.post('/analyze',files={'photo':('p.png',photo(),'image/png')},data={'include_images':'true'})
            self.assertEqual(rich.status_code,200);self.assert_identity(rich)
            self.assertEqual(rich.json()['photo_rule'],plain.json()['photo_rule'])
            self.assertIn('overlay_png_base64',rich.json()['images'])
            self.assertIn('heatmap_png_base64',rich.json()['images'])
            self.assertEqual(rich.json()['images']['content_type'],'image/png')

    def test_jpeg_warning_and_explicit_rgba_option(self):
        with self.client() as client:
            jpeg=client.post('/analyze',files={'photo':('p.jpg',photo('JPEG'),'image/jpeg')})
            self.assertEqual(jpeg.status_code,200);self.assert_identity(jpeg)
            self.assertTrue(any('DCT' in message for message in jpeg.json()['input']['warnings']))
            rgba=photo(mode='RGBA')
            denied=client.post('/analyze',files={'photo':('p.png',rgba,'image/png')})
            self.assertEqual(denied.status_code,415)
            allowed=client.post('/analyze',files={'photo':('p.png',rgba,'image/png')},data={'allow_rgba':'true'})
            self.assertEqual(allowed.status_code,200)

    def test_unsupported_corrupt_tiny_and_bad_fields(self):
        with self.client() as client:
            for data,status in ((photo('GIF'),415),(b'corrupt',422),(photo(size=(128,256)),422)):
                with self.subTest(status=status):
                    response=client.post('/analyze',files={'photo':('claimed.png',data,'image/png')})
                    self.assertEqual(response.status_code,status);self.assert_identity(response)
                    self.assertNotIn('Traceback',response.text)
            for data in ({'include_images':'perhaps'},{'unknown':'true'}):
                response=client.post('/analyze',files={'photo':('p.png',photo(),'image/png')},data=data)
                self.assertEqual(response.status_code,422);self.assert_identity(response)
            self.assertEqual(client.post('/analyze',content=b'not multipart').status_code,415)
            self.assert_identity(client.get('/missing'))

    def test_stream_limit_precedes_parse_and_ignores_understated_length(self):
        with self.client(max_request_bytes=1024) as client:
            # TestClient generator: no trusted Content-Length. Parser must not run.
            with patch('starlette.requests.Request.form',side_effect=AssertionError('parser invoked')):
                response=client.post('/analyze',content=(b'x'*600 for _ in range(3)),
                    headers={'Content-Type':'multipart/form-data; boundary=test'})
                self.assertNotIn('content-length',response.request.headers)
                self.assertEqual(response.status_code,413);self.assert_identity(response)
                # Also drive genuinely separate ASGI body frames, including a
                # deliberately false low Content-Length, below the multipart layer.
                async def streamed():
                    bodies=iter([dict(type='http.request',body=b'x'*600,more_body=True)]*3)
                    sent=[];reads=0
                    async def receive():
                        nonlocal reads
                        reads+=1;return next(bodies)
                    async def send(message):sent.append(message)
                    scope=dict(type='http',asgi={'version':'3.0'},http_version='1.1',method='POST',scheme='http',
                        path='/analyze',raw_path=b'/analyze',query_string=b'',server=('testserver',80),client=('testclient',1),
                        headers=[(b'content-type',b'multipart/form-data; boundary=test'),(b'content-length',b'1')])
                    await client.app(scope,receive,send)
                    self.assertEqual(sent[0]['status'],413)
                    self.assertEqual(reads,2)
                    self.assertEqual(json.loads(sent[1]['body'])['bundle'],self.pipeline.identity)
                asyncio.run(streamed())
            self.assertFalse(client.app.state.analysis_lock.locked())
            self.assertEqual(client.get('/health').status_code,200)

    def test_busy_429_and_health_remains_responsive_off_event_loop(self):
        entered=threading.Event();release=threading.Event()
        real=self.pipeline.analyze
        def slow(*args,**kwargs):
            entered.set()
            if not release.wait(10):raise TimeoutError('test did not release worker')
            return real(*args,**kwargs)
        with self.client() as client,patch.object(self.pipeline,'analyze',side_effect=slow):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first=pool.submit(client.post,'/analyze',files={'photo':('p.png',photo(),'image/png')})
                try:
                    self.assertTrue(entered.wait(5))
                    second=client.post('/analyze',files={'photo':('p.png',photo(),'image/png')})
                    self.assertEqual(second.status_code,429);self.assert_identity(second)
                    self.assertEqual(client.get('/health').status_code,200)
                finally:release.set()
                self.assertEqual(first.result(timeout=10).status_code,200)
            self.assertFalse(client.app.state.analysis_lock.locked())

    def test_failed_bundle_stays_unready_without_path_leak(self):
        with patch('cvdl.pipeline.serve.load_pipeline',side_effect=ValueError('/private/server/weights.pt')):
            with TestClient(create_app('/private/server/bundle')) as client:
                for response in (client.get('/health'),client.post('/analyze',content=b'x')):
                    self.assertEqual(response.status_code,503)
                    self.assertIn('bundle',response.json());self.assertIsNone(response.json()['bundle'])
                    self.assertNotIn('/private',response.text)

    def test_explicit_cors_origin_only(self):
        with self.client(allowed_origins=['http://localhost:3000']) as client:
            for origin,status in (('http://localhost:3000',200),('https://not-allowed.example',400)):
                response=client.options('/analyze',headers={'Origin':origin,'Access-Control-Request-Method':'POST'})
                self.assertEqual(response.status_code,status);self.assert_identity(response)
                if status==200:self.assertEqual(response.headers['access-control-allow-origin'],origin)
                else:self.assertNotIn('access-control-allow-origin',response.headers)
        with self.assertRaises(ValueError):create_app(allowed_origins=['*'])

    def test_upload_spools_are_closed_after_response(self):
        original=tempfile.SpooledTemporaryFile;spools=[]
        def tracked(*args,**kwargs):
            value=original(*args,**kwargs);spools.append(value);return value
        with self.client() as client:
            with patch('tempfile.SpooledTemporaryFile',side_effect=tracked):
                response=client.post('/analyze',files={'photo':('p.png',photo(),'image/png')})
            self.assertEqual(response.status_code,200)
            self.assertTrue(spools)
            self.assertTrue(all(value.closed for value in spools))


if __name__=='__main__':unittest.main()
