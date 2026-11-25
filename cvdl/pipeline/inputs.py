"""Strict photo boundary: preserve stored RGB8 pixels, never rotate or rescale."""
import io
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_ENCODED_BYTES=64*1024*1024
MAX_PIXELS=32_000_000


class InputError(ValueError):
    """Invalid/corrupt image or unsupported input value."""


class UnsupportedInput(InputError):
    """The carrier cannot be examined without an unapproved conversion."""


class InputTooLarge(InputError):
    """Encoded or decoded size exceeds the deployment limit."""


def _dimensions(size):
    w,h=size
    if w*h>MAX_PIXELS:raise InputTooLarge('decoded photo exceeds 32 megapixels')
    if min(w,h)<256:raise InputError('both photo dimensions must be at least 256; no resizing or padding')


def _png_header(header):
    if len(header)<33 or header[:16]!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
        raise InputError('invalid PNG header')
    if header[24]!=8 or header[25] not in (2,6):
        raise UnsupportedInput('PNG must store 8-bit RGB or explicitly permitted RGBA samples')


def _preflight(image,header,allow_rgba):
    _dimensions(image.size)
    if image.format and image.format not in ('PNG','JPEG','TIFF'):
        raise UnsupportedInput('only RGB8 PNG, JPEG and strict uncompressed TIFF are supported')
    if image.format=='PNG' and header is not None:_png_header(header)
    if getattr(image,'n_frames',1)!=1:raise UnsupportedInput('multiframe images are unsupported')
    if image.format=='TIFF':
        tags=image.tag_v2
        if (image.mode!='RGB' or tuple(tags.get(258,()))!=(8,8,8) or tags.get(259)!=1
                or tags.get(262)!=2 or tags.get(274,1)!=1 or tags.get(277)!=3
                or tags.get(284,1)!=1 or tags.get(338) is not None
                or tuple(tags.get(339,(1,1,1)))!=(1,1,1)):
            raise UnsupportedInput('TIFF requires uncompressed RGB8 with orientation 1')
    if image.mode!='RGB' and not (allow_rgba and image.mode=='RGBA'):
        raise UnsupportedInput('RGB8 required; RGBA needs explicit opt-in; no palette/grey/16-bit conversion')
    if image.mode=='RGB' and 'transparency' in image.info:
        raise UnsupportedInput('RGB PNG transparency is unsupported; provide explicit RGB samples')


def _from_image(image,header,allow_rgba):
    _preflight(image,header,allow_rgba)
    image.load()  # PNG EXIF can become available only after loading; never transpose.
    orientation=image.getexif().get(274,1)
    if not isinstance(orientation,int) or orientation not in range(1,9):
        raise UnsupportedInput('invalid EXIF orientation')
    messages=[]
    if orientation!=1:messages.append('EXIF orientation was not applied; coordinates follow stored pixels.')
    if image.format=='JPEG':
        messages.append('JPEG is decoded RGB only, not JPEG/DCT steganalysis; recompression can destroy RGB-LSB payloads.')
    pixels=np.array(image,copy=True)
    if image.mode=='RGBA':
        pixels=pixels[:,:,:3].copy()
        messages.append('Alpha was discarded without compositing; alpha-channel payloads are not examined.')
    if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:
        raise UnsupportedInput('decoded photo must be HWC RGB uint8')
    return np.ascontiguousarray(pixels),dict(format=image.format or 'decoded',orientation=orientation,warnings=messages)


def decode_photo(photo,*,allow_rgba=False):
    """Return private RGB8 pixels plus warnings. No color/orientation changes."""
    if isinstance(photo,np.ndarray):
        if photo.dtype!=np.uint8 or photo.ndim!=3 or photo.shape[2] not in (3,4):
            raise UnsupportedInput('array must be HWC uint8 RGB (or explicit RGBA)')
        _dimensions((photo.shape[1],photo.shape[0]))
        if photo.shape[2]==4 and not allow_rgba:raise UnsupportedInput('RGBA requires explicit opt-in')
        messages=['Already-decoded input: caller must guarantee original RGB8 samples and RGB, not BGR, order.']
        if photo.shape[2]==4:messages.append('Alpha was discarded without compositing; alpha-channel payloads are not examined.')
        return np.array(photo[:,:,:3],copy=True,order='C'),dict(format='array',orientation=1,warnings=messages)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            if isinstance(photo,Image.Image):
                # Inspect an original PNG stream if it remains available. Once
                # decoded externally, original depth/orientation cannot be proven.
                header=None;stream=getattr(photo,'fp',None)
                if photo.format=='PNG' and stream is not None:
                    position=stream.tell()
                    try:stream.seek(0);header=stream.read(33)
                    finally:stream.seek(position)
                pixels,metadata=_from_image(photo,header,allow_rgba)
                metadata['warnings'].append('Already-decoded input: original carrier format/depth is the caller\'s responsibility.')
                return pixels,metadata
            if isinstance(photo,(str,Path)):
                with Path(photo).open('rb') as stream:encoded=stream.read(MAX_ENCODED_BYTES+1)
            elif isinstance(photo,(bytes,bytearray,memoryview)):
                if len(photo)>MAX_ENCODED_BYTES:raise InputTooLarge('encoded photo exceeds 64 MiB')
                encoded=bytes(photo)
            else:raise UnsupportedInput('expected path, encoded bytes, PIL image or numpy array')
            if len(encoded)>MAX_ENCODED_BYTES:raise InputTooLarge('encoded photo exceeds 64 MiB')
            header=encoded[:33]
            if header.startswith(b'\x89PNG'):_png_header(header)
            with Image.open(io.BytesIO(encoded)) as image:
                return _from_image(image,header if image.format=='PNG' else None,allow_rgba)
    except InputError:
        raise
    except (Image.DecompressionBombWarning,Image.DecompressionBombError) as exc:
        raise InputTooLarge('image decompression limit exceeded') from exc
    except (OSError,ValueError,TypeError,SyntaxError,UnidentifiedImageError) as exc:
        raise InputError('cannot decode a valid supported photo') from exc
