"""Helper functions for a standard streaming compression API"""

import io
import sys
from zipfile import ZipFile

import fsspec.utils
from fsspec.spec import AbstractBufferedFile


def noop_file(file, mode, **kwargs):
    return file


# TODO: files should also be available as contexts
# should be functions of the form func(infile, mode=, **kwargs) -> file-like
compr = {None: noop_file}

# Magic bytes identifying a compression format by content, filled in by
# register_compression(); maps compression name -> magic byte prefix
magic_bytes: dict[str, bytes] = {}


def register_compression(name, callback, extensions, force=False, magic=None):
    """Register an "inferable" file compression type.

    Registers transparent file compression type for use with fsspec.open.
    Compression can be specified by name in open, or "infer"-ed for any files
    ending with the given extensions.

    Args:
        name: (str) The compression type name. Eg. "gzip".
        callback: A callable of form (infile, mode, **kwargs) -> file-like.
            Accepts an input file-like object, the target mode and kwargs.
            Returns a wrapped file-like object.
        extensions: (str, Iterable[str]) A file extension, or list of file
        extensions for which to infer this compression scheme. Eg. "gz".
        force: (bool) Force re-registration of compression type or extensions.
        magic: (bytes, optional) Byte prefix identifying this compression
            format when reading a file's content, used to detect compression
            of files whose name carries no (or a misleading) extension.

    Raises:
        ValueError: If name or extensions already registered, and not force.

    """
    if isinstance(extensions, str):
        extensions = [extensions]

    # Validate registration
    if name in compr and not force:
        raise ValueError(f"Duplicate compression registration: {name}")

    for ext in extensions:
        if ext in fsspec.utils.compressions and not force:
            raise ValueError(f"Duplicate compression file extension: {ext} ({name})")

    compr[name] = callback

    for ext in extensions:
        fsspec.utils.compressions[ext] = name

    if magic is not None:
        magic_bytes[name] = bytes(magic)


def _max_magic_length():
    """Length of the longest registered magic byte prefix."""
    return max((len(magic) for magic in magic_bytes.values()), default=0)


def _match_magic(prefix):
    """Compression name whose magic bytes start ``prefix``, if any."""
    # longest magic first, so more specific signatures win
    for name, magic in sorted(magic_bytes.items(), key=lambda kv: -len(kv[1])):
        if prefix.startswith(magic):
            return name
    return None


def _read_up_to(file, n):
    """Read up to n bytes, tolerating streams that return short reads."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = file.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class PrefixedReader(io.RawIOBase):
    """Read-only stream replaying ``prefix`` bytes before the wrapped file.

    Used to put back bytes that were consumed from a non-seekable stream
    while sniffing for compression magic bytes, so that detection never
    eats into the payload.
    """

    def __init__(self, prefix, file):
        self._prefix = memoryview(prefix)
        self._file = file

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def readinto(self, b):
        n = len(b)
        written = 0
        if self._prefix:
            k = min(n, len(self._prefix))
            b[:k] = self._prefix[:k]
            self._prefix = self._prefix[k:]
            written = k
            if written == n:
                return written
        chunk = self._file.read(n - written)
        if chunk:
            chunk = chunk[: n - written]
            b[written : written + len(chunk)] = chunk
            written += len(chunk)
        return written

    def close(self):
        # the wrapped file is owned by our caller, do not close it here
        super().close()


def peek_compression(file, try_seek=True):
    """Detect compression from a stream's content without consuming data.

    Reads at most as many bytes as the longest registered magic prefix. For
    seekable files (when ``try_seek``) the position is restored afterwards;
    otherwise the file is wrapped in a ``PrefixedReader`` which transparently
    replays the peeked bytes. Empty and short streams simply yield no match.

    Parameters
    ----------
    file: file-like
        Open binary stream to inspect.
    try_seek: bool
        Whether to attempt restoring the position with tell/seek. Decompressor
        streams may claim to be seekable but silently misbehave, so pass
        False when inspecting an already-wrapped stream.

    Returns
    -------
    (name or None, file): the detected compression name and the file object
    to continue reading from (identical to the input for seekable files).
    """
    n = _max_magic_length()
    if n == 0:
        return None, file
    if try_seek:
        try:
            seekable = file.seekable()
        except Exception:
            seekable = False
        if seekable:
            try:
                pos = file.tell()
                prefix = _read_up_to(file, n)
                file.seek(pos)
                return _match_magic(prefix), file
            except (OSError, ValueError):
                # position could not be restored; fall through and treat the
                # stream as non-seekable (some bytes may be lost, but this is
                # not expected for well-behaved seekable files)
                pass
    prefix = _read_up_to(file, n)
    return _match_magic(prefix), PrefixedReader(prefix, file)


def sniff_compression(file, mode="rb", max_layers=1, try_seek=True):
    """Wrap a stream in decompressors according to its content.

    Detects registered compression formats by their magic bytes and applies
    the matching decompressor. Only a single layer is unwrapped by default;
    pass ``max_layers`` > 1 to also unwrap nested compression (e.g. a gzip
    file within a gzip file). Detection reads only a few bytes per layer and
    never consumes payload data; non-seekable streams are supported.

    Parameters
    ----------
    file: file-like
        Open binary stream to read from.
    mode: str
        Read mode to pass to the decompressor constructors.
    max_layers: int
        Maximum number of nested compression layers to unwrap.
    try_seek: bool
        Whether the initial stream may be inspected using tell/seek.

    Returns
    -------
    (file, names, wrappers): the file object to read decompressed data from,
    the list of detected compression names (outermost first) and the list of
    created decompressor wrappers in the same order.
    """
    names = []
    wrappers = []
    for layer in range(max(1, int(max_layers))):
        name, file = peek_compression(file, try_seek=try_seek and layer == 0)
        if name is None:
            break
        file = compr[name](file, mode=mode)
        names.append(name)
        wrappers.append(file)
    return file, names, wrappers


def unzip(infile, mode="rb", filename=None, **kwargs):
    if "r" not in mode:
        filename = filename or "file"
        z = ZipFile(infile, mode="w", **kwargs)
        fo = z.open(filename, mode="w")
        fo.close = lambda closer=fo.close: closer() or z.close()
        return fo
    z = ZipFile(infile)
    if filename is None:
        filename = z.namelist()[0]
    return z.open(filename, mode="r", **kwargs)


register_compression("zip", unzip, "zip", magic=b"PK\x03\x04")

try:
    from bz2 import BZ2File
except ImportError:
    pass
else:
    register_compression("bz2", BZ2File, "bz2", magic=b"BZh")

try:  # pragma: no cover
    from isal import igzip

    def isal(infile, mode="rb", **kwargs):
        return igzip.IGzipFile(fileobj=infile, mode=mode, **kwargs)

    register_compression("gzip", isal, "gz", magic=b"\x1f\x8b")
except ImportError:
    from gzip import GzipFile

    register_compression(
        "gzip",
        lambda f, **kwargs: GzipFile(fileobj=f, **kwargs),
        "gz",
        magic=b"\x1f\x8b",
    )

try:
    from lzma import LZMAFile

    register_compression("lzma", LZMAFile, "lzma")
    register_compression("xz", LZMAFile, "xz", magic=b"\xfd7zXZ\x00")
except ImportError:
    pass

try:
    import lzmaffi

    register_compression(
        "lzma", lzmaffi.LZMAFile, "lzma", force=True, magic=b"\xfd7zXZ\x00"
    )
    register_compression(
        "xz", lzmaffi.LZMAFile, "xz", force=True, magic=b"\xfd7zXZ\x00"
    )
except ImportError:
    pass


class SnappyFile(AbstractBufferedFile):
    def __init__(self, infile, mode, **kwargs):
        import snappy

        super().__init__(
            fs=None, path="snappy", mode=mode.strip("b") + "b", size=999999999, **kwargs
        )
        self.infile = infile
        if "r" in mode:
            self.codec = snappy.StreamDecompressor()
        else:
            self.codec = snappy.StreamCompressor()

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        out = self.codec.add_chunk(self.buffer.read())
        self.infile.write(out)
        return True

    def seek(self, loc, whence=0):
        raise NotImplementedError("SnappyFile is not seekable")

    def seekable(self):
        return False

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        data = self.infile.read(end - start)
        return self.codec.decompress(data)


try:
    import snappy

    snappy.compress(b"")
    # Snappy may use the .sz file extension, but this is not part of the
    # standard implementation.
    register_compression("snappy", SnappyFile, [])

except (ImportError, NameError, AttributeError):
    pass

try:
    import lz4.frame

    register_compression("lz4", lz4.frame.open, "lz4", magic=b"\x04\x22\x4d\x18")
except ImportError:
    pass

try:
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        from backports import zstd

    register_compression("zstd", zstd.ZstdFile, "zst", magic=b"\x28\xb5\x2f\xfd")
except ImportError:
    try:
        import zstandard as zstd

        def zstandard_file(infile, mode="rb"):
            if "r" in mode:
                cctx = zstd.ZstdDecompressor()
                return cctx.stream_reader(infile)
            else:
                cctx = zstd.ZstdCompressor(level=10)
                return cctx.stream_writer(infile)

        register_compression("zstd", zstandard_file, "zst", magic=b"\x28\xb5\x2f\xfd")
    except ImportError:
        pass
    pass


def available_compressions():
    """Return a list of the implemented compressions."""
    return list(compr)
