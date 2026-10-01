import pytest

import fsspec
from fsspec import AbstractFileSystem, FSMap
from fsspec.exceptions import (
    ChangesetCollisionError,
    ChangesetConflictError,
    ChangesetError,
    ChangesetUnsupportedError,
)
from fsspec.implementations.memory import MemoryFileSystem


class NoTransactionMemoryFileSystem(MemoryFileSystem):
    """A backend that deliberately does not support atomic changesets

    Inherits everything from the memory filesystem (including version
    tokens) but falls back to the base-class changeset behaviour, which
    can only apply entries sequentially and must refuse atomic mode.
    """

    protocol = "notxn-memory"
    supports_atomic_changeset = False
    apply_changeset = AbstractFileSystem.apply_changeset


@pytest.fixture
def fs():
    return MemoryFileSystem(global_store=False, skip_instance_cache=True)


@pytest.fixture
def mapper(fs):
    return FSMap("/root", fs)


@pytest.fixture
def notxn_fs():
    return NoTransactionMemoryFileSystem(global_store=False, skip_instance_cache=True)


@pytest.fixture
def notxn_mapper(notxn_fs):
    return FSMap("/root", notxn_fs)


def test_changeset_writes_and_deletes(mapper):
    mapper["keep"] = b"old"
    mapper["gone"] = b"bye"

    statuses = mapper.apply_changeset(
        {"a": b"1", "b/c": b"2", "gone": None, "absent": None, "keep": b"new"}
    )

    assert statuses == {
        "a": "applied",
        "b/c": "applied",
        "gone": "applied",
        "absent": "applied",  # deleting a missing key is a no-op success
        "keep": "applied",
    }
    assert mapper["a"] == b"1"
    assert mapper["b/c"] == b"2"
    assert mapper["keep"] == b"new"
    assert "gone" not in mapper
    assert "absent" not in mapper


def test_get_versions_track_mutation(mapper):
    versions = mapper.get_versions(["k", "missing"])
    # both keys are absent; "absent" is a well-defined token of its own
    assert versions["missing"] == versions["k"]

    mapper["k"] = b"1"
    v1 = mapper.get_versions(["k"])["k"]
    assert v1 != versions["k"]  # create changed the token
    assert v1 != versions["missing"]  # existing != missing

    mapper["k"] = b"2"
    v2 = mapper.get_versions(["k"])["k"]
    assert v2 != v1  # write changed the token

    del mapper["k"]
    v3 = mapper.get_versions(["k"])["k"]
    assert v3 != v2

    mapper["k"] = b"1"
    v4 = mapper.get_versions(["k"])["k"]
    assert v4 not in (v1, v2, v3)  # delete+recreate never reproduces a token


def test_changeset_preconditions_pass(mapper):
    mapper["a"] = b"1"
    mapper["b"] = b"2"
    versions = mapper.get_versions(["a", "b", "c"])

    statuses = mapper.apply_changeset(
        {"a": b"10", "c": b"3"}, expected_versions=versions
    )

    assert statuses == {"a": "applied", "c": "applied"}
    assert mapper["a"] == b"10"
    assert mapper["c"] == b"3"


def test_changeset_conflict_applies_nothing_and_retry_needs_fresh_token(mapper):
    mapper["a"] = b"1"
    mapper["b"] = b"2"
    versions = mapper.get_versions(["a", "b"])

    # someone else writes "b" after we read it
    mapper["b"] = b"changed"

    with pytest.raises(ChangesetConflictError) as excinfo:
        mapper.apply_changeset({"a": b"10"}, expected_versions=versions)

    exc = excinfo.value
    assert set(exc.conflicts) == {"b"}
    assert exc.conflicts["b"]["expected"] == versions["b"]
    assert exc.conflicts["b"]["actual"] != versions["b"]
    assert exc.statuses == {"a": "skipped"}
    # all-or-nothing: the write to "a" never happened
    assert mapper["a"] == b"1"
    assert mapper["b"] == b"changed"

    # retrying with the same stale token fails again instead of silently
    # overwriting the newer value
    with pytest.raises(ChangesetConflictError):
        mapper.apply_changeset({"a": b"10"}, expected_versions=versions)
    assert mapper["b"] == b"changed"

    # re-reading and retrying with fresh tokens succeeds
    fresh = mapper.get_versions(["a", "b"])
    mapper.apply_changeset({"a": b"10"}, expected_versions=fresh)
    assert mapper["a"] == b"10"
    assert mapper["b"] == b"changed"


def test_changeset_conflict_on_deleted_key(mapper):
    mapper["a"] = b"1"
    versions = mapper.get_versions(["a"])
    del mapper["a"]

    with pytest.raises(ChangesetConflictError):
        mapper.apply_changeset({"b": b"2"}, expected_versions=versions)
    assert "b" not in mapper


def test_changeset_atomic_rollback(mapper, fs, monkeypatch):
    mapper["first"] = b"original"
    original = mapper.fs.store["/root/first"]

    def failing_pipe_file(path, value, *args, **kwargs):
        if path.endswith("boom"):
            raise OSError("simulated backend failure")
        return MemoryFileSystem.pipe_file(fs, path, value, *args, **kwargs)

    monkeypatch.setattr(fs, "pipe_file", failing_pipe_file)

    with pytest.raises(ChangesetError) as excinfo:
        mapper.apply_changeset({"first": b"updated", "boom": b"x", "third": b"y"})

    exc = excinfo.value
    assert exc.statuses == {"first": "skipped", "boom": "failed", "third": "skipped"}
    # the earlier write was rolled back: one visible change or none
    assert mapper["first"] == b"original"
    assert mapper.fs.store["/root/first"] is original
    assert "boom" not in mapper
    assert "third" not in mapper


def test_changeset_key_collision(mapper):
    with pytest.raises(ChangesetCollisionError) as excinfo:
        mapper.apply_changeset({"a": b"1", "a/": b"2"})

    exc = excinfo.value
    assert isinstance(exc, ValueError)
    assert exc.statuses == {"a": "skipped", "a/": "skipped"}
    assert {tuple(keys) for keys in exc.collisions.values()} == {("a", "a/")}
    assert "a" not in mapper


def test_changeset_collision_between_changes_and_preconditions(mapper):
    mapper["a"] = b"1"
    versions = mapper.get_versions(["a"])

    with pytest.raises(ChangesetCollisionError):
        mapper.apply_changeset({"a": b"2"}, expected_versions={"a/": versions["a"]})

    # the same key in both changes and preconditions is fine, though
    mapper.apply_changeset({"a": b"2"}, expected_versions=versions)
    assert mapper["a"] == b"2"


def test_changeset_atomic_unsupported_applies_nothing(notxn_mapper):
    assert not notxn_mapper.fs.supports_atomic_changeset

    with pytest.raises(ChangesetUnsupportedError) as excinfo:
        notxn_mapper.apply_changeset({"a": b"1", "b": None})

    exc = excinfo.value
    assert isinstance(exc, NotImplementedError)
    assert exc.statuses == {"a": "skipped", "b": "skipped"}
    # refused, not faked: nothing was written
    assert len(notxn_mapper) == 0


def test_changeset_non_atomic_best_effort(notxn_mapper):
    notxn_mapper["gone"] = b"bye"

    statuses = notxn_mapper.apply_changeset({"a": b"1", "gone": None}, atomic=False)

    assert statuses == {"a": "applied", "gone": "applied"}
    assert notxn_mapper["a"] == b"1"
    assert "gone" not in notxn_mapper


def test_changeset_non_atomic_preconditions(notxn_mapper):
    notxn_mapper["a"] = b"1"
    versions = notxn_mapper.get_versions(["a"])
    notxn_mapper["a"] = b"changed"

    with pytest.raises(ChangesetConflictError):
        notxn_mapper.apply_changeset(
            {"b": b"2"}, expected_versions=versions, atomic=False
        )
    assert "b" not in notxn_mapper

    fresh = notxn_mapper.get_versions(["a"])
    notxn_mapper.apply_changeset({"b": b"2"}, expected_versions=fresh, atomic=False)
    assert notxn_mapper["b"] == b"2"


def test_changeset_non_atomic_partial_failure_reports_statuses(
    notxn_mapper, notxn_fs, monkeypatch
):
    def failing_pipe_file(path, value, *args, **kwargs):
        if path.endswith("boom"):
            raise OSError("simulated backend failure")
        return MemoryFileSystem.pipe_file(notxn_fs, path, value, *args, **kwargs)

    monkeypatch.setattr(notxn_fs, "pipe_file", failing_pipe_file)

    with pytest.raises(ChangesetError) as excinfo:
        notxn_mapper.apply_changeset(
            {"first": b"1", "boom": b"x", "third": b"y"}, atomic=False
        )

    exc = excinfo.value
    assert exc.statuses == {"first": "applied", "boom": "failed", "third": "skipped"}
    # non-atomic mode is honest about the partial state it left behind
    assert notxn_mapper["first"] == b"1"
    assert "boom" not in notxn_mapper
    assert "third" not in notxn_mapper


def test_changeset_direct_fs_api_atomic_unsupported(notxn_fs):
    with pytest.raises(ChangesetUnsupportedError):
        notxn_fs.apply_changeset({"/root/a": b"1"})
    assert not notxn_fs.exists("/root/a")


def test_get_version_unsupported_by_default():
    fs = fsspec.filesystem("file")
    with pytest.raises(ChangesetUnsupportedError):
        fs.get_version("/some/path")


def test_changeset_accepts_array_like_values(mapper):
    import array

    mapper.apply_changeset({"arr": array.array("B", b"bytes")})
    assert mapper["arr"] == b"bytes"
