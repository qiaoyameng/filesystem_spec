import os
import pickle
import platform
import sys
import uuid

import pytest

import fsspec
from fsspec.exceptions import (
    ChangeSetPreconditionFailed,
    ChangeSetState,
    ChangeSetUnsupported,
    ChangeSetValidationError,
)
from fsspec.spec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem


class UnsupportedChangesetFileSystem(AbstractFileSystem):
    protocol = "unsupported-changesets"
    root_marker = ""

    def _strip_protocol(self, path):
        return str(path).removeprefix("unsupported-changesets://")


def test_mapping_prefix(tmpdir):
    tmpdir = str(tmpdir)
    os.makedirs(os.path.join(tmpdir, "afolder"))
    open(os.path.join(tmpdir, "afile"), "w").write("test")
    open(os.path.join(tmpdir, "afolder", "anotherfile"), "w").write("test2")

    m = fsspec.get_mapper(f"file://{tmpdir}")
    assert "afile" in m
    assert m["afolder/anotherfile"] == b"test2"

    fs = fsspec.filesystem("file")
    m2 = fs.get_mapper(tmpdir)
    m3 = fs.get_mapper(f"file://{tmpdir}")

    assert m == m2 == m3


@pytest.mark.parametrize("protocol", ["file", "memory"])
@pytest.mark.parametrize("key", ["a", "a/nested"])
def test_check_preserves_contents(tmp_path, protocol, key):
    url = f"{protocol}://{tmp_path.as_posix()}/mapping"
    mapper = fsspec.get_mapper(url, create=True)
    contents = {key: b"existing data", "other": b"more data"}
    mapper.update(contents)

    checked = fsspec.get_mapper(url, check=True)

    assert dict(checked) == contents


def test_getitems_errors(tmpdir):
    tmpdir = str(tmpdir)
    os.makedirs(os.path.join(tmpdir, "afolder"))
    open(os.path.join(tmpdir, "afile"), "w").write("test")
    open(os.path.join(tmpdir, "afolder", "anotherfile"), "w").write("test2")
    m = fsspec.get_mapper(f"file://{tmpdir}")
    assert m.getitems(["afile", "bfile"], on_error="omit") == {"afile": b"test"}
    with pytest.raises(KeyError):
        m.getitems(["afile", "bfile"])
    out = m.getitems(["afile", "bfile"], on_error="return")
    assert isinstance(out["bfile"], KeyError)
    m = fsspec.get_mapper(f"file://{tmpdir}", missing_exceptions=())
    assert m.getitems(["afile", "bfile"], on_error="omit") == {"afile": b"test"}
    with pytest.raises(FileNotFoundError):
        m.getitems(["afile", "bfile"])


def test_ops():
    MemoryFileSystem.store.clear()
    m = fsspec.get_mapper("memory://")
    assert not m
    assert list(m) == []

    with pytest.raises(KeyError):
        m["hi"]

    assert m.pop("key", 0) == 0

    m["key0"] = b"data"
    assert list(m) == ["key0"]
    assert m["key0"] == b"data"

    m.clear()

    assert list(m) == []


def test_pickle():
    m = fsspec.get_mapper("memory://")
    assert isinstance(m.fs, MemoryFileSystem)
    m["key"] = b"data"
    m2 = pickle.loads(pickle.dumps(m))
    assert list(m) == list(m2)
    assert m.missing_exceptions == m2.missing_exceptions


def test_keys_view():
    # https://github.com/fsspec/filesystem_spec/issues/186
    m = fsspec.get_mapper("memory://")
    m["key"] = b"data"

    keys = m.keys()
    assert len(keys) == 1
    # check that we don't consume the keys
    assert len(keys) == 1
    m.clear()


def test_multi():
    m = fsspec.get_mapper("memory:///")
    data = {"a": b"data1", "b": b"data2"}
    m.setitems(data)

    assert m.getitems(list(data)) == data
    m.delitems(list(data))
    assert not list(m)


def test_setitem_types():
    import array

    m = fsspec.get_mapper("memory://")
    m["a"] = array.array("i", [1])
    if sys.byteorder == "little":
        assert m["a"] == b"\x01\x00\x00\x00"
    else:
        assert m["a"] == b"\x00\x00\x00\x01"
    m["b"] = bytearray(b"123")
    assert m["b"] == b"123"
    m.setitems({"c": array.array("i", [1]), "d": bytearray(b"123")})
    if sys.byteorder == "little":
        assert m["c"] == b"\x01\x00\x00\x00"
    else:
        assert m["c"] == b"\x00\x00\x00\x01"
    assert m["d"] == b"123"


def test_setitem_numpy():
    m = fsspec.get_mapper("memory://")
    np = pytest.importorskip("numpy")
    m["c"] = np.array(1, dtype="<i4")  # scalar
    assert m["c"] == b"\x01\x00\x00\x00"
    m["c"] = np.array([1, 2], dtype="<i4")  # array
    assert m["c"] == b"\x01\x00\x00\x00\x02\x00\x00\x00"
    m["c"] = np.array(
        np.datetime64("2000-01-01T23:59:59.999999999"), dtype="<M8[ns]"
    )  # datetime64 scalar
    assert m["c"] == b"\xff\xff\x91\xe3c\x9b#\r"
    m["c"] = np.array(
        [
            np.datetime64("1900-01-01T23:59:59.999999999"),
            np.datetime64("2000-01-01T23:59:59.999999999"),
        ],
        dtype="<M8[ns]",
    )  # datetime64 array
    assert m["c"] == b"\xff\xff}p\xf8fX\xe1\xff\xff\x91\xe3c\x9b#\r"
    m["c"] = np.array(
        np.timedelta64(3155673612345678901, "ns"), dtype="<m8[ns]"
    )  # timedelta64 scalar
    assert m["c"] == b"5\x1c\xf0Rn4\xcb+"
    m["c"] = np.array(
        [
            np.timedelta64(450810516049382700, "ns"),
            np.timedelta64(3155673612345678901, "ns"),
        ],
        dtype="<m8[ns]",
    )  # timedelta64 scalar
    assert m["c"] == b',M"\x9e\xc6\x99A\x065\x1c\xf0Rn4\xcb+'


def test_empty_url():
    m = fsspec.get_mapper()
    assert isinstance(m.fs, LocalFileSystem)


def test_fsmap_access_with_root_prefix(tmp_path):
    # "/a" and "a" are the same for LocalFileSystem
    tmp_path.joinpath("a").write_bytes(b"data")
    m = fsspec.get_mapper(f"file://{tmp_path}")
    assert m["/a"] == m["a"] == b"data"

    # "/a" and "a" differ for MemoryFileSystem
    m = fsspec.get_mapper(f"memory://{uuid.uuid4()}")
    m["/a"] = b"data"

    assert m["/a"] == b"data"
    with pytest.raises(KeyError):
        _ = m["a"]


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(b"k", id="bytes"),
        pytest.param(1234, id="int"),
        pytest.param((1,), id="tuple"),
        pytest.param([""], id="list"),
    ],
)
def test_fsmap_non_str_keys(key):
    m = fsspec.get_mapper()

    # Once the deprecation period passes
    # FSMap.__getitem__ should raise TypeError for non-str keys
    #   with pytest.raises(TypeError):
    #       _ = m[key]

    with pytest.warns(DeprecationWarning):
        with pytest.raises(KeyError):
            _ = m[key]


def test_fsmap_error_on_protocol_keys():
    root = uuid.uuid4()
    m = fsspec.get_mapper(f"memory://{root}", create=True)
    m["a"] = b"data"

    assert m["a"] == b"data"
    with pytest.raises(KeyError):
        _ = m[f"memory://{root}/a"]


def test_fsmap_access_with_suffix(tmp_path):
    tmp_path.joinpath("b").mkdir()
    tmp_path.joinpath("b", "a").write_bytes(b"data")
    if platform.system() == "Windows":
        # on Windows opening a directory will raise PermissionError
        # see: https://bugs.python.org/issue43095
        missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
        )
    else:
        missing_exceptions = None
    m = fsspec.get_mapper(f"file://{tmp_path}", missing_exceptions=missing_exceptions)
    with pytest.raises(KeyError):
        _ = m["b/"]
    assert m["b/a/"] == b"data"


def test_fsmap_dirfs():
    m = fsspec.get_mapper("memory://")

    fs = m.dirfs
    assert isinstance(fs, fsspec.implementations.dirfs.DirFileSystem)
    assert fs.path == m.root


@pytest.mark.parametrize("protocol", ["memory", "file"])
@pytest.mark.parametrize("default", [None, False, 0, b"", []])
def test_pop_missing_key_returns_explicit_default(tmp_path, protocol, default):
    mapper = fsspec.get_mapper(
        f"{protocol}://{tmp_path.as_posix()}/mapping", create=True
    )

    assert mapper.pop("missing", default) is default


@pytest.mark.parametrize("protocol", ["memory", "file"])
def test_pop_without_default_raises_for_missing_key(tmp_path, protocol):
    mapper = fsspec.get_mapper(
        f"{protocol}://{tmp_path.as_posix()}/mapping", create=True
    )

    with pytest.raises(KeyError, match="missing"):
        mapper.pop("missing")


@pytest.mark.parametrize("protocol", ["memory", "file"])
def test_pop_existing_key_returns_and_removes_value(tmp_path, protocol):
    mapper = fsspec.get_mapper(
        f"{protocol}://{tmp_path.as_posix()}/mapping", create=True
    )
    mapper["key"] = b"value"

    assert mapper.pop("key", None) == b"value"
    assert "key" not in mapper


def test_memory_changeset_commits_writes_and_deletes_atomically():
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = fsspec.FSMap("", fs)
    mapper["old"] = b"old"
    old_version = fs.ukey("/old")
    changeset = mapper.changeset(
        writes={"new": b"new", "replaced": b"one"},
        deletes=["old"],
        preconditions={"old": old_version, "missing": None},
    )

    result = changeset.commit(mapper)

    assert changeset.state == ChangeSetState.COMMITTED
    assert result["new"] == fs.ukey("/new")
    assert result["replaced"] == fs.ukey("/replaced")
    assert result["old"] is None
    assert dict(mapper) == {"new": b"new", "replaced": b"one"}
    assert fs.store is not mapper.fs.store or "/old" not in fs.store


def test_memory_changeset_aborts_when_precondition_is_stale():
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = fsspec.FSMap("", fs)
    mapper["key"] = b"one"
    version = fs.ukey("/key")
    changeset = mapper.changeset(
        writes={"key": b"two", "other": b"three"},
        preconditions={"key": version},
    )
    mapper["key"] = b"newer"

    with pytest.raises(ChangeSetPreconditionFailed):
        changeset.commit(mapper)

    assert changeset.state == ChangeSetState.ABORTED
    assert mapper["key"] == b"newer"
    assert "other" not in mapper

    with pytest.raises(ChangeSetValidationError) as info:
        changeset.commit(mapper)
    assert info.value.state == ChangeSetState.ABORTED
    assert mapper["key"] == b"newer"
    assert "other" not in mapper


def test_changeset_rejects_collision_before_backend_call():
    fs = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    mapper = fsspec.FSMap("", fs)
    changeset = mapper.changeset(writes={"1": b"x"}, deletes=[1])

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ChangeSetValidationError):
            changeset.commit(mapper)

    assert changeset.state == ChangeSetState.ABORTED


def test_changeset_rejects_unsupported_backend_without_writing():
    fs = UnsupportedChangesetFileSystem(skip_instance_cache=True)
    called = []
    fs.commit_changeset = lambda **kwargs: called.append(kwargs)
    mapper = fsspec.FSMap("", fs)
    changeset = mapper.changeset(writes={"key": b"value"})

    with pytest.raises(ChangeSetUnsupported):
        changeset.commit(mapper)

    assert changeset.state == ChangeSetState.ABORTED
    assert called == []


def test_non_atomic_changeset_does_not_require_backend_capability():
    fs = UnsupportedChangesetFileSystem(skip_instance_cache=True)
    writes = {}
    fs.pipe = lambda values: writes.update(values)
    fs.rm = lambda paths: None
    fs.isfile = lambda path: False
    fs.ukey = lambda path: None
    fs._parent = lambda path: ""
    fs.mkdirs = lambda path, exist_ok=True: None
    mapper = fsspec.FSMap("", fs)
    changeset = mapper.changeset(
        writes={"key": b"value"}, deletes=["gone"], atomic=False
    )

    result = changeset.commit(mapper)

    assert changeset.state == ChangeSetState.COMMITTED
    assert result == {"key": None, "gone": None}
    assert writes == {"key": b"value"}
