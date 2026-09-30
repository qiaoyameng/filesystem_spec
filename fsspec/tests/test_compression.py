import bz2
import gzip
import io
import os
import pathlib
import sys

import pytest

import fsspec.config
import fsspec.core
from fsspec.compression import (
    apply_inferred_compression,
    compr,
    infer_compression_from_content,
    magics,
    register_compression,
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

    # ...but the content magic is recognised when reading.
    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="rb", compression="infer"
    ) as infile:
        assert infile.read().decode() == tdat

    with fsspec.core.open(
        str(tmp_path / "out.snappy"), mode="rt", compression="snappy"
    ) as infile:
        assert infile.read() == tdat


class SlowStream:
    """Non-seekable stream returning at most ``chunk`` bytes per read."""

    def __init__(self, data, chunk=1):
        self._bio = io.BytesIO(data)
        self.chunk = chunk

    def read(self, n=-1):
        if n is None or n < 0:
            return self._bio.read()
        return self._bio.read(min(n, self.chunk))

    def readable(self):
        return True

    def seekable(self):
        return False

    def close(self):
        self._bio.close()

    @property
    def closed(self):
        return self._bio.closed


def test_infer_from_content_no_extension(tmpdir):
    """Files without a known extension are sniffed by content."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"hello world\n" * 100

    (tmp_path / "datafile").write_bytes(gzip.compress(data))
    with fsspec.core.open(str(tmp_path / "datafile"), compression="infer") as f:
        assert f.read() == data

    with fsspec.core.open(
        str(tmp_path / "datafile"), mode="rt", compression="infer"
    ) as f:
        assert f.read() == data.decode()

    (tmp_path / "datafile2").write_bytes(bz2.compress(data))
    with fsspec.core.open(str(tmp_path / "datafile2"), compression="infer") as f:
        assert f.read() == data

    # uncompressed content passes through untouched
    (tmp_path / "plain").write_bytes(data)
    with fsspec.core.open(str(tmp_path / "plain"), compression="infer") as f:
        assert f.read() == data


def test_infer_from_content_empty_and_short(tmpdir):
    """Empty and short streams have defined passthrough behaviour."""
    tmp_path = pathlib.Path(str(tmpdir))

    for i, content in enumerate([b"", b"a", b"\x1f", b"BZ"]):
        (tmp_path / f"f{i}").write_bytes(content)
        with fsspec.core.open(str(tmp_path / f"f{i}"), compression="infer") as f:
            assert f.read() == content


def test_infer_from_content_fragmented_prefix():
    """Sniffing tolerates reads returning fewer bytes than requested."""
    data = b"fragmented\n" * 100
    payload = gzip.compress(data)

    f = apply_inferred_compression(SlowStream(payload, chunk=1))
    assert f.read() == data


def test_infer_from_content_non_seekable_passthrough():
    """Non-seekable streams without known magic remain fully readable."""
    data = b"plain data, no magic" * 10

    f = apply_inferred_compression(SlowStream(data))
    assert not f.seekable()
    assert f.read() == data


def test_infer_read_half_close(tmpdir):
    """Reading part of a stream and closing works and closes the raw file."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"partial\n" * 100
    (tmp_path / "data").write_bytes(gzip.compress(data))

    of = fsspec.core.open(str(tmp_path / "data"), compression="infer")
    f = of.open()
    raw = of.fobjects[0]
    assert f.read(10) == data[:10]
    of.close()
    assert raw.closed
    assert of.fobjects == []


def test_infer_corrupt_header(tmpdir):
    """A corrupt compression header surfaces as an error from the codec."""
    tmp_path = pathlib.Path(str(tmpdir))
    (tmp_path / "bad").write_bytes(b"\x1f\x8b" + os.urandom(100))

    with pytest.raises(Exception):  # noqa: B017 - codec-specific error type
        with fsspec.core.open(str(tmp_path / "bad"), compression="infer") as f:
            f.read()


def test_infer_extension_content_conflict(tmpdir):
    """Conflicts between extension and content follow the chosen policy."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"conflict-data" * 50
    # a ".gz" file containing bz2 payload
    (tmp_path / "data.gz").write_bytes(bz2.compress(data))
    path = str(tmp_path / "data.gz")

    # strict is the default policy
    with pytest.raises(ValueError, match="does not match"):
        with fsspec.core.open(path, compression="infer") as f:
            f.read()

    with pytest.raises(ValueError, match="does not match"):
        with fsspec.core.open(
            path, compression="infer", compression_policy="strict"
        ) as f:
            f.read()

    with fsspec.core.open(
        path, compression="infer", compression_policy="content"
    ) as f:
        assert f.read() == data

    # trusting the extension defers the failure to the codec
    with pytest.raises(OSError):
        with fsspec.core.open(
            path, compression="infer", compression_policy="extension"
        ) as f:
            f.read()

    with pytest.raises(ValueError, match="Unknown compression inference policy"):
        with fsspec.core.open(
            path, compression="infer", compression_policy="bogus"
        ) as f:
            f.read()


def test_infer_extension_content_conflict_config(tmpdir):
    """The conflict policy can be set via the config system."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"conflict-data" * 50
    (tmp_path / "data.gz").write_bytes(bz2.compress(data))

    fsspec.config.conf["compression_infer_policy"] = "content"
    try:
        with fsspec.core.open(
            str(tmp_path / "data.gz"), compression="infer"
        ) as f:
            assert f.read() == data
    finally:
        del fsspec.config.conf["compression_infer_policy"]


def test_infer_matching_extension_and_content(tmpdir):
    """A matching extension and content works under the strict policy."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"consistent" * 50
    (tmp_path / "ok.gz").write_bytes(gzip.compress(data))

    with fsspec.core.open(str(tmp_path / "ok.gz"), compression="infer") as f:
        assert f.read() == data


def test_infer_nested_layers(tmpdir):
    """Only one layer is decompressed by default; more on request."""
    tmp_path = pathlib.Path(str(tmpdir))
    data = b"nested-data" * 100
    (tmp_path / "nested").write_bytes(gzip.compress(gzip.compress(data)))
    path = str(tmp_path / "nested")

    with fsspec.core.open(path, compression="infer") as f:
        out = f.read()
    assert out.startswith(b"\x1f\x8b")
    assert gzip.decompress(out) == data

    with fsspec.core.open(path, compression="infer", compression_max_layers=2) as f:
        assert f.read() == data


def test_infer_opens_underlying_file_once(tmpdir):
    """Sniffing reads from the single opened file, never reopens it."""
    from fsspec.core import OpenFile
    from fsspec.implementations.local import LocalFileSystem

    class CountingFS(LocalFileSystem):
        def __init__(self):
            super().__init__()
            self.opens = 0

        def open(self, *args, **kwargs):
            self.opens += 1
            return super().open(*args, **kwargs)

    tmp_path = pathlib.Path(str(tmpdir))
    data = b"count-me" * 100
    (tmp_path / "noext").write_bytes(gzip.compress(data))

    fs = CountingFS()
    of = OpenFile(fs, str(tmp_path / "noext"), mode="rb", compression="infer")
    with of as f:
        assert f.read() == data
    assert fs.opens == 1


def test_register_compression_magic():
    """Custom compressions can register content magic bytes."""
    register_compression("test_magic", lambda f, **kwargs: f, "tmagic", magic=b"TMG!")
    try:
        assert infer_compression_from_content(b"TMG!payload") == "test_magic"
        assert infer_compression_from_content(b"other") is None

        out = apply_inferred_compression(io.BytesIO(b"TMG!payload"))
        assert out.read() == b"TMG!payload"
    finally:
        del compr["test_magic"]
        del compressions["tmagic"]
        del magics["test_magic"]
