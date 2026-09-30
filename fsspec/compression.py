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

# Magic byte signatures for content-based inference, registered via
# register_compression(magic=...). Maps compression name to signatures.
magics: dict[str, tuple[bytes, ...]] = {}

# Policies for resolving extension/content conflicts during inference
INFER_POLICIES = ("strict", "content", "extension")


def register_compression(name, callback, extensions, force=False, magic=None):
    """Register an "inferable" file compression type.

    Registers transparent file compression type for use with fsspec.open.
    Compression can be specified by name in open, or "infer"-ed for any files
    ending with the given extensions. If ``magic`` bytes are given, the
    compression can also be inferred from the file contents.

    Args:
        name: (str) The compression type name. Eg. "gzip".
        callback: A callable of form (infile, mode, **kwargs) -> file-like.
            Accepts an input file-like object, the target mode and kwargs.
            Returns a wrapped file-like object.
        extensions: (str, Iterable[str]) A file extension, or list of file
            extensions for which to infer this compression scheme. Eg. "gz".
        force: (bool) Force re-registration of compression type or extensions.
        magic: (bytes, Iterable[bytes]) Optional leading byte signature(s)
            identifying this compression within file contents. Eg. b"\\x1f\\x8b"
            for gzip.

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
        if isinstance(magic, bytes):
            magic = [magic]
        magics[name] = tuple(magic)


def infer_compression_from_content(prefix):
    """Infer compression, if available, from leading content bytes.

    Matches the given bytes against the magic signatures registered via
    fsspec.compression.register_compression. Returns the compression name
    with the longest matching signature, or None if nothing matches. A
    signature only matches if it is fully contained in ``prefix``.
    """
    best = None
    for name, signatures in magics.items():
        for signature in signatures:
            if prefix.startswith(signature):
                if best is None or len(signature) > len(best[1]):
                    best = (name, signature)
    return best[0] if best is not None else None


def _max_magic_length():
    return max((len(sig) for sigs in magics.values() for sig in sigs), default=0)


def _read_prefix(f, size):
    """Read up to ``size`` bytes from ``f``, tolerating fragmented reads."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = f.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_and_restore(f, size):
    """Read up to ``size`` leading bytes without consuming them.

    Seeks back to the original position when possible; otherwise wraps
    the stream so the bytes are replayed. Returns ``(prefix, stream)``.
    """
    position = None
    try:
        if f.seekable():
            position = f.tell()
    except (AttributeError, OSError):
        position = None
    prefix = _read_prefix(f, size)
    if position is not None:
        f.seek(position)
        return prefix, f
    return prefix, _PrefixFile(prefix, f)


class _PrefixFile:
    """File-like replaying ``prefix`` bytes before delegating to ``f``.

    Pushes back the bytes consumed while sniffing, so that the payload is
    not eaten and non-seekable streams remain readable from the start.
    """

    def __init__(self, prefix, f):
        self._buffer = io.BytesIO(prefix)
        self.f = f

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            return self._buffer.read() + self.f.read()
        out = self._buffer.read(size)
        if len(out) < size:
            out += self.f.read(size - len(out))
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        self.f.close()

    @property
    def closed(self):
        return getattr(self.f, "closed", False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _equivalent_compression(name1, name2):
    """Whether two compression names refer to the same codec."""
    return name1 == name2 or compr.get(name1) is compr.get(name2)


def apply_inferred_compression(f, name=None, policy="strict", max_layers=1):
    """Wrap a binary read stream, decompressing based on content sniffing.

    Only the limited number of magic bytes required to recognise the
    registered compression types is read; those bytes are pushed back onto
    the stream, so no payload is consumed and non-seekable streams remain
    readable. The whole file is never read to "try" a decompressor.

    Args:
        f: file-like
            Binary stream to wrap.
        name: str or None
            Compression name inferred from the file extension, if any.
        policy: "strict", "content" or "extension"
            How to resolve a conflict between ``name`` (from the extension)
            and the compression detected in the content:

            - "strict": raise ValueError on conflict
            - "content": trust the content
            - "extension": trust the extension; content is not sniffed
        max_layers: int
            Maximum number of nested compression layers to unwrap; layers
            beyond the first are always detected from content. Defaults to
            1, i.e. only a single layer is decompressed.

    Returns:
        A file-like object reading the (possibly) decompressed stream.
    """
    if policy not in INFER_POLICIES:
        raise ValueError(
            f"Unknown compression inference policy: {policy!r}; "
            f"expected one of {INFER_POLICIES}"
        )
    layer = 0
    while layer < max_layers:
        if layer == 0 and name is not None and policy == "extension":
            # Trust the extension; no need to inspect the content.
            pass
        else:
            prefix, f = _read_and_restore(f, _max_magic_length())
            detected = infer_compression_from_content(prefix)
            if (
                layer == 0
                and name is not None
                and detected is not None
                and not _equivalent_compression(name, detected)
            ):
                if policy == "strict":
                    raise ValueError(
                        f"Compression inferred from extension ({name!r}) does "
                        f"not match the content ({detected!r}); set the "
                        f"compression inference policy to 'content' or "
                        f"'extension' to disambiguate."
                    )
                name = detected
            if name is None:
                name = detected
        if name is None:
            break
        f = compr[name](f, mode="rb")
        layer += 1
        name = None
    return f


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


register_compression(
    "zip", unzip, "zip", magic=[b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"]
)

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

    register_compression("lzma", lzmaffi.LZMAFile, "lzma", force=True)
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
    register_compression("snappy", SnappyFile, [], magic=b"\xff\x06\x00\x00sNaPpY")

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
