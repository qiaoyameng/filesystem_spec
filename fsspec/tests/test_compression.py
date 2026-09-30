import io
import pathlib
import sys

import pytest

import fsspec.core
import fsspec.implementations.memory
from fsspec.compression import (
    compr,
    magic_bytes,
    peek_compression,
    register_compression,
    sniff_compression,
)
from fsspec.utils import compressions, infer_compression


def test_infer_custom_compression():
    """Inferred compression gets values from fsspec.compression.compr."""
    assert infer_compression("fn.zip") == "zip"
    assert infer_compression("fn.gz") == "gzip"
    assert infer_compression("fn.unknown") is None
    assert infer_compression("fn.test_custom") is None
    assert infer_compression("fn.tst") is None

    register_compression("test_custom", lambda f, **kwargs: f, "tst")

    try:
        assert infer_compression("fn.zip") == "zip"
        assert infer_compression("fn.gz") == "gzip"
        assert infer_compression("fn.unknown") is None
        assert infer_compression("fn.test_custom") is None
        assert infer_compression("fn.tst") == "test_custom"

        # Duplicate registration in name or extension raises a value error.
        with pytest.raises(ValueError):
            register_compression("test_custom", lambda f, **kwargs: f, "tst")

        with pytest.raises(ValueError):
            register_compression("test_conflicting", lambda f, **kwargs: f, "tst")
        assert "test_conflicting" not in compr

        # ...but can be forced.
        register_compression(
            "test_conflicting", lambda f, **kwargs: f, "tst", force=True
        )
        assert infer_compression("fn.zip") == "zip"
        assert infer_compression("fn.gz") == "gzip"
        assert infer_compression("fn.unknown") is None
        assert infer_compression("fn.test_custom") is None
        assert infer_compression("fn.tst") == "test_conflicting"

    finally:
        del compr["test_custom"]
        del compr["test_conflicting"]
        del compressions["tst"]


def test_infer_uppercase_compression():
    assert infer_compression("fn.ZIP") == "zip"
    assert infer_compression("fn.GZ") == "gzip"
    assert infer_compression("fn.UNKNOWN") is None
    assert infer_compression("fn.TEST_UPPERCASE") is None
    assert infer_compression("fn.TEST") is None


def test_lzma_compression_name():
    pytest.importorskip("lzma")
    assert infer_compression("fn.xz") == "xz"
    assert infer_compression("fn.lzma") == "lzma"


def test_lz4_compression(tmpdir):
    """Infer lz4 compression for .lz4 files if lz4 is available."""
    tmp_path = pathlib.Path(str(tmpdir))

    lz4 = pytest.importorskip("lz4")

    tmp_path.mkdir(exist_ok=True)

    tdat = "foobar" * 100

    with fsspec.core.open(
        str(tmp_path / "out.lz4"), mode="wt", compression="infer"
    ) as outfile:
        outfile.write(tdat)

    compressed = (tmp_path / "out.lz4").open("rb").read()
    assert lz4.frame.decompress(compressed).decode() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.lz4"), mode="rt", compression="infer"
    ) as infile:
        assert infile.read() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.lz4"), mode="rt", compression="lz4"
    ) as infile:
        assert infile.read() == tdat


def test_zstd_compression(tmpdir):
    """Infer zstd compression for .zst files if zstandard is available."""
    tmp_path = pathlib.Path(str(tmpdir))

    try:
        if sys.version_info >= (3, 14):
            from compression import zstd
        else:
            zstd = pytest.importorskip("backports.zstd")
    except ImportError:
        zstd = pytest.importorskip("zstandard")

    tmp_path.mkdir(exist_ok=True)

    tdat = "foobar" * 100

    with fsspec.core.open(
        str(tmp_path / "out.zst"), mode="wt", compression="infer"
    ) as outfile:
        outfile.write(tdat)

    compressed = (tmp_path / "out.zst").open("rb").read()
    assert zstd.ZstdDecompressor().decompress(compressed, len(tdat)).decode() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.zst"), mode="rt", compression="infer"
    ) as infile:
        assert infile.read() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.zst"), mode="rt", compression="zstd"
    ) as infile:
        assert infile.read() == tdat

    # fails in https://github.com/fsspec/filesystem_spec/issues/725
    infile = fsspec.core.open(
        str(tmp_path / "out.zst"), mode="rb", compression="infer"
    ).open()

    infile.close()


def test_snappy_compression(tmpdir):
    """No registered compression for snappy, but can be specified."""
    tmp_path = pathlib.Path(str(tmpdir))

    snappy = pytest.importorskip("snappy")

    tmp_path.mkdir(exist_ok=True)

    tdat = "foobar" * 100

    # Snappy isn't inferred.
    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="wt", compression="infer"
    ) as outfile:
        outfile.write(tdat)
    assert (tmp_path / "out.snappy").open("rb").read().decode() == tdat

    # but can be specified.
    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="wt", compression="snappy"
    ) as outfile:
        outfile.write(tdat)

    compressed = (tmp_path / "out.snappy").open("rb").read()
    assert snappy.StreamDecompressor().decompress(compressed).decode() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="rb", compression="infer"
    ) as infile:
        assert infile.read() == compressed

    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="rt", compression="snappy"
    ) as infile:
        assert infile.read() == tdat


PAYLOAD = b"hello world\n" * 100


def _gzip(data):
    import gzip

    return gzip.compress(data)


def _bz2(data):
    import bz2

    return bz2.compress(data)


def _xz(data):
    import lzma

    return lzma.compress(data)


def _zip(data):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("file", data)
    return buffer.getvalue()


def _zstd(data):
    import io

    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        try:
            from backports import zstd
        except ImportError:
            import zstandard as zstd

    buffer = io.BytesIO()
    with zstd.ZstdFile(buffer, "wb") as f:
        f.write(data)
    return buffer.getvalue()


def _available_content_compressions():
    """Map of compression name -> compress function for sniffable formats."""
    out = {}
    for name, func in [("gzip", _gzip), ("bz2", _bz2), ("xz", _xz), ("zip", _zip)]:
        if name in compr:
            out[name] = func
    if "zstd" in compr:
        try:
            _zstd(b"")
        except ImportError:
            pass
        else:
            out["zstd"] = _zstd
    return out


class FragmentedReader(io.RawIOBase):
    """Non-seekable stream yielding its data in small fragments."""

    def __init__(self, data, chunk_size=1):
        self._data = data
        self._pos = 0
        self._chunk_size = chunk_size

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, b):
        if self._pos >= len(self._data):
            return 0
        n = min(len(b), self._chunk_size, len(self._data) - self._pos)
        b[:n] = self._data[self._pos : self._pos + n]
        self._pos += n
        return n


class FragmentedMemoryFS(fsspec.implementations.memory.MemoryFileSystem):
    """Memory filesystem returning non-seekable, fragmented read streams."""

    def __init__(self, *args, chunk_size=1, **kwargs):
        self.chunk_size = chunk_size
        self.open_count = 0
        self.created = []
        super().__init__(*args, **kwargs)

    def _open(self, path, mode="rb", **kwargs):
        f = super()._open(path, mode=mode, **kwargs)
        if "r" in mode:
            self.open_count += 1
            f = FragmentedReader(f.read(), self.chunk_size)
            self.created.append(f)
        return f


@pytest.fixture()
def fragmented_fs():
    fs = FragmentedMemoryFS()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")
    fs.open_count = 0
    fs.created = []
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")


@pytest.mark.parametrize("fmt", sorted(_available_content_compressions()))
def test_infer_from_content_no_extension(m, fmt):
    """Files without extension are decompressed based on their content."""
    compress = _available_content_compressions()[fmt]
    m.pipe("/data", compress(PAYLOAD))
    with fsspec.open("memory:///data", "rb", compression="infer") as f:
        assert f.read() == PAYLOAD


def test_infer_from_content_unknown_extension(m):
    m.pipe("/data.bin", _gzip(PAYLOAD))
    with fsspec.open("memory:///data.bin", "rb", compression="infer") as f:
        assert f.read() == PAYLOAD


def test_infer_from_content_plain_passthrough(m):
    """Uncompressed content without extension is returned untouched."""
    m.pipe("/data", PAYLOAD)
    with fsspec.open("memory:///data", "rb", compression="infer") as f:
        assert f.read() == PAYLOAD


def test_infer_from_content_empty_file(m):
    m.pipe("/empty", b"")
    with fsspec.open("memory:///empty", "rb", compression="infer") as f:
        assert f.read() == b""


@pytest.mark.parametrize("data", [b"\x1f", b"P", b"PK"])
def test_infer_from_content_short_file(m, data):
    """Streams shorter than the magic prefixes are returned as-is."""
    m.pipe("/short", data)
    with fsspec.open("memory:///short", "rb", compression="infer") as f:
        assert f.read() == data


def test_infer_from_content_does_not_consume_payload(m):
    """The bytes read for detection are still delivered to the reader."""
    data = _gzip(PAYLOAD)
    m.pipe("/data", data)
    of = fsspec.open("memory:///data", "rb", compression="infer")
    with of as f:
        first = f.read(5)
        rest = f.read()
    assert first + rest == PAYLOAD


def test_infer_from_content_nonseekable_fragmented(fragmented_fs):
    """Detection works on non-seekable streams with fragmented reads."""
    fragmented_fs.pipe("/data", _gzip(PAYLOAD))
    of = fsspec.core.OpenFile(fragmented_fs, "/data", "rb", compression="infer")
    with of as f:
        assert not of.fobjects[0].seekable()
        assert f.read() == PAYLOAD
    assert fragmented_fs.open_count == 1


def test_infer_from_content_opened_once(m):
    """Content detection does not re-open the underlying file."""
    from unittest import mock

    m.pipe("/data", _gzip(PAYLOAD))
    with mock.patch.object(m, "open", wraps=m.open) as mocked:
        with fsspec.open("memory:///data", "rb", compression="infer") as f:
            assert f.read() == PAYLOAD
    assert mocked.call_count == 1


def test_infer_from_content_read_half_then_close(fragmented_fs):
    """Closing mid-read is clean and closes the underlying stream."""
    fragmented_fs.pipe("/data", _gzip(PAYLOAD))
    of = fsspec.core.OpenFile(fragmented_fs, "/data", "rb", compression="infer")
    with of as f:
        assert f.read(10) == PAYLOAD[:10]
    assert of.fobjects == []
    assert fragmented_fs.created[0].closed


def test_infer_from_content_corrupted_header(m):
    """A matching magic with a corrupt header fails on read, not on open."""
    # valid gzip magic, invalid compression method byte
    m.pipe("/data", b"\x1f\x8b\x07" + b"\x00" * 20)
    of = fsspec.open("memory:///data", "rb", compression="infer")
    with of as f:
        with pytest.raises(OSError):
            f.read()


def test_infer_nested_default_single_layer(m):
    """Nested compression is only unwrapped once by default."""
    m.pipe("/data", _gzip(_gzip(PAYLOAD)))
    with fsspec.open("memory:///data", "rb", compression="infer") as f:
        assert f.read() == _gzip(PAYLOAD)


def test_infer_nested_max_layers(m):
    m.pipe("/data", _gzip(_gzip(PAYLOAD)))
    with fsspec.open(
        "memory:///data", "rb", compression="infer", compression_max_layers=2
    ) as f:
        assert f.read() == PAYLOAD
    # extra layers are harmless when the content is not further compressed
    with fsspec.open(
        "memory:///data", "rb", compression="infer", compression_max_layers=5
    ) as f:
        assert f.read() == PAYLOAD


def test_infer_nested_max_layers_nonseekable(fragmented_fs):
    fragmented_fs.pipe("/data", _gzip(_gzip(PAYLOAD)))
    of = fsspec.core.OpenFile(
        fragmented_fs, "/data", "rb", compression="infer", compression_max_layers=2
    )
    with of as f:
        assert f.read() == PAYLOAD
    assert fragmented_fs.open_count == 1


def test_compression_policy_extension_is_default(m):
    """By default the extension wins over conflicting content."""
    m.pipe("/data.gz", _bz2(PAYLOAD))
    with fsspec.open("memory:///data.gz", "rb", compression="infer") as f:
        with pytest.raises(OSError):
            f.read()


def test_compression_policy_content(m):
    """The content policy prefers the detected content over the extension."""
    m.pipe("/data.gz", _bz2(PAYLOAD))
    with fsspec.open(
        "memory:///data.gz", "rb", compression="infer", compression_policy="content"
    ) as f:
        assert f.read() == PAYLOAD


def test_compression_policy_content_falls_back_to_extension(m):
    """Content without known magic is still handled per the extension."""
    m.pipe("/data.gz", _gzip(PAYLOAD))
    with fsspec.open(
        "memory:///data.gz", "rb", compression="infer", compression_policy="content"
    ) as f:
        assert f.read() == PAYLOAD


def test_compression_policy_strict(m):
    """The strict policy raises when extension and content disagree."""
    m.pipe("/data.gz", _bz2(PAYLOAD))
    of = fsspec.open(
        "memory:///data.gz", "rb", compression="infer", compression_policy="strict"
    )
    with pytest.raises(ValueError, match="conflicts"):
        with of:
            pass
    # agreeing extension and content is fine
    m.pipe("/other.gz", _gzip(PAYLOAD))
    with fsspec.open(
        "memory:///other.gz", "rb", compression="infer", compression_policy="strict"
    ) as f:
        assert f.read() == PAYLOAD


def test_compression_policy_strict_closes_underlying_file(fragmented_fs):
    fragmented_fs.pipe("/data.gz", _bz2(PAYLOAD))
    of = fsspec.core.OpenFile(
        fragmented_fs, "/data.gz", "rb", compression="infer",
        compression_policy="strict",
    )
    with pytest.raises(ValueError, match="conflicts"):
        of.open()
    assert fragmented_fs.created[0].closed


def test_compression_policy_invalid(m):
    with pytest.raises(ValueError, match="compression_policy"):
        fsspec.open("memory:///data", "rb", compression="infer",
                    compression_policy="bogus")


def test_compression_policy_from_config(m):
    """The policy can be set via the fsspec config system."""
    from fsspec.config import conf

    m.pipe("/data.gz", _bz2(PAYLOAD))
    conf["compression_policy"] = "content"
    try:
        with fsspec.open("memory:///data.gz", "rb", compression="infer") as f:
            assert f.read() == PAYLOAD
    finally:
        del conf["compression_policy"]


def test_explicit_compression_skips_sniffing(m):
    """An explicit compression name is used as-is, without content sniffing."""
    m.pipe("/data", _gzip(PAYLOAD))
    with fsspec.open("memory:///data", "rb", compression="gzip") as f:
        assert f.read() == PAYLOAD


def test_infer_write_mode_no_extension(m):
    """Writing with compression="infer" and no extension stays uncompressed."""
    with fsspec.open("memory:///out", "wb", compression="infer") as f:
        f.write(PAYLOAD)
    assert m.cat("/out") == PAYLOAD


def test_register_compression_with_magic(m):
    """Custom compressions can register magic bytes for content detection."""
    calls = []

    def callback(f, **kwargs):
        calls.append(f)
        return f

    register_compression("test_magic", callback, [], magic=b"ABRA")
    try:
        assert magic_bytes["test_magic"] == b"ABRA"
        m.pipe("/data", b"ABRACADABRA")
        with fsspec.open("memory:///data", "rb", compression="infer") as f:
            assert f.read() == b"ABRACADABRA"
        assert len(calls) == 1
    finally:
        del compr["test_magic"]
        del magic_bytes["test_magic"]


def test_peek_compression_seekable_restores_position():
    import io

    stream = io.BytesIO(_gzip(PAYLOAD))
    name, out = peek_compression(stream)
    assert name == "gzip"
    assert out is stream
    assert stream.tell() == 0
    assert stream.read(2) == b"\x1f\x8b"


def test_peek_compression_nonseekable_replays_prefix():
    stream = FragmentedReader(_gzip(PAYLOAD), chunk_size=1)
    name, out = peek_compression(stream)
    assert name == "gzip"
    assert out is not stream
    assert not out.seekable()
    # no byte was lost: the whole original content can still be read
    assert out.read() == _gzip(PAYLOAD)


def test_peek_compression_empty_and_short_streams():
    import io

    for data in (b"", b"\x1f"):
        name, out = peek_compression(io.BytesIO(data))
        assert name is None
        assert out.read() == data
        name, out = peek_compression(FragmentedReader(data))
        assert name is None
        assert out.read() == data


def test_sniff_compression_returns_names_and_wrappers():
    import io

    file, names, wrappers = sniff_compression(io.BytesIO(_gzip(_gzip(PAYLOAD))))
    assert names == ["gzip"]
    assert len(wrappers) == 1
    assert file.read() == _gzip(PAYLOAD)

    file, names, wrappers = sniff_compression(
        io.BytesIO(_gzip(_gzip(PAYLOAD))), max_layers=2
    )
    assert names == ["gzip", "gzip"]
    assert len(wrappers) == 2
    assert file.read() == PAYLOAD

    file, names, wrappers = sniff_compression(io.BytesIO(PAYLOAD))
    assert names == []
    assert wrappers == []
    assert file.read() == PAYLOAD
