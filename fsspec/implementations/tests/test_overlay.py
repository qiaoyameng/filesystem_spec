import hashlib

import pytest

import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.overlay import OverlayFileSystem

LOWER_FILES = {
    "top.txt": b"top-lower",
    "shadowed.txt": b"lower-version",
    "dir/a.txt": b"aaa",
    "dir/b.txt": b"bbb",
    "dir/sub/deep.txt": b"deep",
}


def make_lower(root):
    lower_dir = root / "lower"
    for rel, data in LOWER_FILES.items():
        path = lower_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return lower_dir


def snapshot(root):
    """Content-addressed snapshot of every file and dir below root."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_file():
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            out[rel] = "dir"
    return out


@pytest.fixture(params=["local", "memory"])
def overlay(request, tmp_path):
    lower_dir = make_lower(tmp_path)
    if request.param == "local":
        upper = LocalFileSystem()
        upper_path = str(tmp_path / "upper")
    else:
        upper = MemoryFileSystem(global_store=False, skip_instance_cache=True)
        upper_path = None
    fs = OverlayFileSystem(
        lower=LocalFileSystem(),
        lower_path=str(lower_dir),
        upper=upper,
        upper_path=upper_path,
        skip_instance_cache=True,
    )
    return fs, lower_dir


def test_read_through_and_merged_listing(overlay):
    fs, _ = overlay
    fs.pipe_file("upper-only.txt", b"u1")
    fs.pipe_file("dir/upper2.txt", b"u2")
    fs.pipe_file("shadowed.txt", b"upper-version")

    # reads: upper wins, otherwise fall through to lower
    assert fs.cat_file("top.txt") == b"top-lower"
    assert fs.cat_file("upper-only.txt") == b"u1"
    assert fs.cat_file("shadowed.txt") == b"upper-version"

    # ls merges both layers
    assert set(fs.ls("", detail=False)) == {
        "top.txt",
        "shadowed.txt",
        "dir",
        "upper-only.txt",
    }
    assert set(fs.ls("dir", detail=False)) == {
        "dir/a.txt",
        "dir/b.txt",
        "dir/sub",
        "dir/upper2.txt",
    }

    # find/walk merge too
    found = fs.find("")
    assert "dir/sub/deep.txt" in found
    assert "dir/upper2.txt" in found
    assert "upper-only.txt" in found
    assert not any(".wh." in name for name in found)

    # info reflects the winning layer
    assert fs.info("shadowed.txt")["size"] == len(b"upper-version")
    assert fs.info("dir")["type"] == "directory"
    assert fs.isfile("top.txt")
    assert not fs.isfile("dir")


def test_writes_go_to_upper_only(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)
    fs.pipe_file("new.txt", b"hello")
    fs.pipe_file("dir/new2.txt", b"world")
    assert fs.cat_file("new.txt") == b"hello"
    assert fs.cat_file("dir/new2.txt") == b"world"
    assert fs.upper.exists("new.txt")
    assert fs.upper.exists("dir/new2.txt")
    assert snapshot(lower_dir) == before


def test_append_triggers_copy_up(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)
    with fs.open("top.txt", "ab") as f:
        f.write(b"+upper")
    assert fs.cat_file("top.txt") == b"top-lower+upper"
    # the lower file itself is unchanged
    assert snapshot(lower_dir) == before


def test_text_mode_write(overlay):
    fs, _ = overlay
    with fs.open("text.txt", "w") as f:
        f.write("hello text")
    assert fs.cat_file("text.txt") == b"hello text"


def test_failed_write_leaves_no_partial_file(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)

    with pytest.raises(RuntimeError):
        with fs.open("broken.txt", "wb") as f:
            f.write(b"partial")
            raise RuntimeError("boom")
    assert not fs.exists("broken.txt")
    assert "broken.txt" not in fs.ls("", detail=False)

    # failing while overwriting a lower file keeps the original visible
    with pytest.raises(RuntimeError):
        with fs.open("top.txt", "wb") as f:
            f.write(b"garbage")
            raise RuntimeError("boom")
    assert fs.cat_file("top.txt") == b"top-lower"

    # no staging/whiteout debris left behind in the upper layer
    assert not [p for p in fs.upper.find("") if ".wh." in p]
    assert snapshot(lower_dir) == before


def test_delete_lower_file_records_whiteout(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)
    fs.rm("top.txt")
    assert not fs.exists("top.txt")
    assert "top.txt" not in fs.ls("", detail=False)
    assert "top.txt" not in fs.find("")
    with pytest.raises(FileNotFoundError):
        fs.info("top.txt")

    # the whiteout marker exists in the upper layer only
    assert fs.upper.exists(".wh.top.txt")
    # ...but is invisible and cannot be confused with a user file
    assert not fs.exists(".wh.top.txt")
    assert not any(".wh." in name for name in fs.find(""))
    with pytest.raises(FileNotFoundError):
        fs.info(".wh.top.txt")

    # lower data is untouched, and double delete fails cleanly
    assert snapshot(lower_dir) == before
    with pytest.raises(FileNotFoundError):
        fs.rm("top.txt")


def test_whiteout_names_never_collide_with_user_files(overlay):
    fs, _ = overlay
    # reserved prefix is rejected for user writes
    with pytest.raises(ValueError):
        fs.pipe_file(".wh.sneaky", b"x")
    with pytest.raises(ValueError):
        fs.mkdir(".wh.sneaky-dir")

    # delete + recreate of the same name works; whiteout does not shadow
    # the new upper file
    fs.rm("shadowed.txt")
    assert fs.upper.exists(".wh.shadowed.txt")
    fs.pipe_file("shadowed.txt", b"recreated")
    assert fs.cat_file("shadowed.txt") == b"recreated"
    assert fs.ls("", detail=False).count("shadowed.txt") == 1


def test_removed_dir_recreated_empty(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)

    fs.rm("dir", recursive=True)
    assert not fs.exists("dir")
    assert "dir" not in fs.ls("", detail=False)

    fs.mkdir("dir")
    assert fs.isdir("dir")
    assert fs.ls("dir") == []

    fs.pipe_file("dir/fresh.txt", b"new")
    assert fs.ls("dir", detail=False) == ["dir/fresh.txt"]
    # lower descendants stay hidden behind the whiteout
    assert not fs.exists("dir/a.txt")
    assert not fs.exists("dir/sub/deep.txt")
    assert snapshot(lower_dir) == before


def test_file_and_directory_name_conflict(overlay):
    fs, lower_dir = overlay
    # a lower *file* and an upper *directory* with the same name
    (lower_dir / "conflict").write_bytes(b"lower-file")
    before = snapshot(lower_dir)
    fs.mkdir("conflict")
    assert fs.isdir("conflict")
    assert not fs.isfile("conflict")
    fs.pipe_file("conflict/inner.txt", b"inner")
    # the lower file is not listed as a child of the upper directory
    assert fs.ls("conflict", detail=False) == ["conflict/inner.txt"]
    # removing the directory hides the whole path via a whiteout
    fs.rm("conflict", recursive=True)
    assert not fs.exists("conflict")
    assert snapshot(lower_dir) == before

    # the reverse conflict (file over lower directory) is rejected
    with pytest.raises(IsADirectoryError):
        fs.pipe_file("dir", b"nope")


def test_mv_file_across_layers(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)

    # lower-only file -> new name: content copied up, source whiteouted
    fs.mv("top.txt", "moved.txt")
    assert not fs.exists("top.txt")
    assert fs.cat_file("moved.txt") == b"top-lower"
    assert fs.upper.exists("moved.txt")
    assert fs.upper.exists(".wh.top.txt")

    # upper file -> path occupied by a lower file (replace)
    fs.pipe_file("staging.txt", b"staged")
    fs.mv("staging.txt", "shadowed.txt")
    assert fs.cat_file("shadowed.txt") == b"staged"
    assert not fs.exists("staging.txt")

    # lower file -> directory that only exists in the upper layer
    fs.mkdir("newdir")
    fs.mv("dir/a.txt", "newdir/a.txt")
    assert fs.cat_file("newdir/a.txt") == b"aaa"
    assert not fs.exists("dir/a.txt")
    assert fs.cat_file("dir/b.txt") == b"bbb"

    assert snapshot(lower_dir) == before


def test_mv_directory_does_not_resurrect_deleted_descendants(overlay):
    fs, lower_dir = overlay
    before = snapshot(lower_dir)

    fs.rm("dir/a.txt")  # whiteout inside dir
    fs.mv("dir", "dir2", recursive=True)

    assert not fs.exists("dir")
    assert fs.isdir("dir2")
    assert fs.cat_file("dir2/b.txt") == b"bbb"
    assert fs.cat_file("dir2/sub/deep.txt") == b"deep"
    # the descendant deleted before the rename stays deleted
    assert not fs.exists("dir2/a.txt")
    assert not any(".wh." in name for name in fs.find(""))
    assert snapshot(lower_dir) == before


def test_rename_alias_handles_directories(overlay):
    fs, _ = overlay
    fs.rename("dir", "dir3")
    assert not fs.exists("dir")
    assert fs.cat_file("dir3/sub/deep.txt") == b"deep"


def test_walk_and_glob(overlay):
    fs, _ = overlay
    fs.rm("dir/a.txt")
    fs.pipe_file("dir/new.txt", b"n")

    walked = set()
    for root, dirs, files in fs.walk(""):
        walked.add(root)
    assert "dir" in walked
    assert "dir/sub" in walked

    assert set(fs.glob("dir/*.txt")) == {"dir/b.txt", "dir/new.txt"}
    assert set(fs.glob("dir/sub/*")) == {"dir/sub/deep.txt"}


def test_lower_layer_is_never_modified(tmp_path):
    lower_dir = make_lower(tmp_path)
    # make the base genuinely read-only to prove nothing writes to it
    for path in sorted(lower_dir.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    lower_dir.chmod(0o555)
    try:
        fs = OverlayFileSystem(
            lower=LocalFileSystem(),
            lower_path=str(lower_dir),
            upper=MemoryFileSystem(global_store=False, skip_instance_cache=True),
            skip_instance_cache=True,
        )
        fs.pipe_file("new.txt", b"x")
        fs.pipe_file("dir/more.txt", b"y")
        with fs.open("top.txt", "ab") as f:
            f.write(b"+")
        fs.rm("shadowed.txt")
        fs.rm("dir/a.txt")
        fs.mv("dir/b.txt", "b-moved.txt")
        fs.rm("dir/sub", recursive=True)

        assert fs.cat_file("top.txt") == b"top-lower+"
        assert not fs.exists("shadowed.txt")
        assert not fs.exists("dir/sub")
        assert fs.cat_file("b-moved.txt") == b"bbb"
    finally:
        for path in sorted(lower_dir.rglob("*"), reverse=True):
            path.chmod(0o644 if path.is_file() else 0o755)
        lower_dir.chmod(0o755)
    for rel, data in LOWER_FILES.items():
        assert (lower_dir / rel).read_bytes() == data


def test_registered_protocol(tmp_path):
    lower_dir = make_lower(tmp_path)
    fs = fsspec.filesystem(
        "overlay",
        lower_path=str(lower_dir),
        upper_protocol="memory",
        upper_options={"global_store": False, "skip_instance_cache": True},
        skip_instance_cache=True,
    )
    assert isinstance(fs, OverlayFileSystem)
    assert fs.cat_file("top.txt") == b"top-lower"
