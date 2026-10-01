import os

import pytest

import fsspec
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.implementations.overlay import TMP_ROOT, WHITEOUT_ROOT, OverlayFileSystem


@pytest.fixture
def lower_dir(tmp_path):
    base = tmp_path / "lower"
    (base / "dir" / "sub").mkdir(parents=True)
    (base / "top.txt").write_bytes(b"top-lower")
    (base / "shared.txt").write_bytes(b"shared-lower")
    (base / "dir" / "keep.txt").write_bytes(b"keep")
    (base / "dir" / "sub" / "deep.txt").write_bytes(b"deep")
    return base


def make_overlay(lower_dir, upper, tmp_path):
    if upper == "memory":
        upper = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    elif upper == "local":
        upper = str(tmp_path / "upper")
    return OverlayFileSystem(
        lower=str(lower_dir), upper=upper, skip_instance_cache=True
    )


@pytest.fixture(params=["memory", "local"])
def fs(request, lower_dir, tmp_path):
    return make_overlay(lower_dir, request.param, tmp_path)


def tree_snapshot(path):
    out = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            out[os.path.relpath(full, path)] = open(full, "rb").read()
    return out


def test_read_prefers_upper_falls_back_to_lower(fs):
    assert fs.cat_file("top.txt") == b"top-lower"
    fs.pipe_file("top.txt", b"top-upper")
    assert fs.cat_file("top.txt") == b"top-upper"
    # untouched files still come from the lower layer
    assert fs.cat_file("dir/keep.txt") == b"keep"


def test_ls_and_find_merge_layers(fs):
    fs.pipe_file("shared.txt", b"shared-upper")
    fs.pipe_file("dir/upper-only.txt", b"u")
    fs.pipe_file("new.txt", b"n")

    assert fs.ls("", detail=False) == ["dir", "new.txt", "shared.txt", "top.txt"]
    assert fs.ls("dir", detail=False) == [
        "dir/keep.txt",
        "dir/sub",
        "dir/upper-only.txt",
    ]
    found = fs.find("")
    assert "dir/keep.txt" in found
    assert "dir/sub/deep.txt" in found
    assert "dir/upper-only.txt" in found
    assert "new.txt" in found
    # merged view must not contain duplicates
    assert len(found) == len(set(found))
    # internal whiteout/temp trees never leak into listings
    assert all(not n.startswith(WHITEOUT_ROOT) for n in fs.ls("", detail=False))
    assert all(not n.startswith(TMP_ROOT) for n in fs.ls("", detail=False))


def test_delete_lower_file_records_whiteout(fs, lower_dir):
    fs.rm_file("top.txt")
    assert not fs.exists("top.txt")
    assert "top.txt" not in fs.ls("", detail=False)
    # the base layer is untouched
    assert (lower_dir / "top.txt").read_bytes() == b"top-lower"
    # a whiteout marker was recorded in the upper layer
    assert fs.upper.isfile(WHITEOUT_ROOT + "/top.txt")
    with pytest.raises(FileNotFoundError):
        fs.info("top.txt")
    with pytest.raises(FileNotFoundError):
        fs.rm_file("top.txt")


def test_delete_directory_then_recreate(fs, lower_dir):
    fs.rm("dir", recursive=True)
    assert not fs.exists("dir")
    assert fs.find("dir") == []
    # lower dataset is untouched
    assert (lower_dir / "dir" / "sub" / "deep.txt").read_bytes() == b"deep"

    fs.mkdir("dir")
    assert fs.isdir("dir")
    # the recreated directory is empty: old lower contents stay hidden
    assert fs.ls("dir", detail=False) == []
    assert fs.find("dir") == []

    fs.pipe_file("dir/fresh.txt", b"fresh")
    assert fs.ls("dir", detail=False) == ["dir/fresh.txt"]
    assert not fs.exists("dir/keep.txt")
    assert not fs.exists("dir/sub/deep.txt")


def test_file_directory_same_name_conflicts(fs):
    # lower has a file, upper creates a directory with the same name
    fs.mkdir("shared.txt")
    assert fs.isdir("shared.txt")
    assert fs.ls("shared.txt", detail=False) == []
    fs.pipe_file("shared.txt/inner.txt", b"inner")
    assert fs.cat_file("shared.txt/inner.txt") == b"inner"

    # lower has a directory, upper writes a file with the same name
    fs.pipe_file("dir", b"dir-is-now-a-file")
    assert fs.isfile("dir")
    assert fs.cat_file("dir") == b"dir-is-now-a-file"
    # lower contents below are unreachable through info, but base is intact
    assert fs.info("dir")["type"] == "file"

    # whiteout a lower file, then recreate a directory with the same name
    fs.rm_file("top.txt")
    fs.mkdir("top.txt")
    assert fs.isdir("top.txt")
    assert fs.ls("top.txt", detail=False) == []


def test_rename_directory_does_not_unhide_descendants(fs):
    fs.rm("dir/sub", recursive=True)
    assert fs.find("dir") == ["dir/keep.txt"]

    fs.mv("dir", "renamed")
    assert not fs.exists("dir")
    assert fs.find("renamed") == ["renamed/keep.txt"]
    assert fs.cat_file("renamed/keep.txt") == b"keep"
    # the whiteouted subtree must not reappear under the new name
    assert not fs.exists("renamed/sub")
    assert not fs.exists("renamed/sub/deep.txt")


def test_cross_layer_mv_file(fs, lower_dir):
    fs.mv("top.txt", "newdir/moved.txt")
    assert not fs.exists("top.txt")
    assert fs.cat_file("newdir/moved.txt") == b"top-lower"
    # source is hidden by a whiteout, base is untouched
    assert fs.upper.isfile(WHITEOUT_ROOT + "/top.txt")
    assert (lower_dir / "top.txt").read_bytes() == b"top-lower"

    # moving an upper-only file works too
    fs.pipe_file("u.txt", b"u")
    fs.mv("u.txt", "newdir/u.txt")
    assert not fs.exists("u.txt")
    assert fs.cat_file("newdir/u.txt") == b"u"


def test_copy_up_on_append_and_random_access(fs, lower_dir):
    with fs.open("top.txt", "ab") as f:
        f.write(b"+appended")
    assert fs.cat_file("top.txt") == b"top-lower+appended"
    # copy-up: the base file is not modified
    assert (lower_dir / "top.txt").read_bytes() == b"top-lower"

    with fs.open("top.txt", "r+b") as f:
        f.seek(0)
        f.write(b"TOP")
    assert fs.cat_file("top.txt") == b"TOP-lower+appended"
    assert (lower_dir / "top.txt").read_bytes() == b"top-lower"

    fs.pipe_file("top.txt", b"+more", mode="append")
    assert fs.cat_file("top.txt") == b"TOP-lower+appended+more"


def test_failed_write_leaves_no_visible_half_file(fs, lower_dir):
    with pytest.raises(RuntimeError, match="boom"):
        with fs.open("top.txt", "wb") as f:
            f.write(b"partial")
            raise RuntimeError("boom")
    # the original lower content is still served, nothing leaked upwards
    assert fs.cat_file("top.txt") == b"top-lower"
    assert not fs.upper.exists("top.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with fs.open("brand-new.txt", "wb") as f:
            f.write(b"partial")
            raise RuntimeError("boom")
    assert not fs.exists("brand-new.txt")


def test_failed_commit_leaves_no_visible_half_file(fs, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("commit failed")

    monkeypatch.setattr(fs.upper, "mv", boom)
    with pytest.raises(RuntimeError, match="commit failed"):
        with fs.open("top.txt", "wb") as f:
            f.write(b"partial")
    assert fs.cat_file("top.txt") == b"top-lower"
    # no temporary files are left behind in the upper layer
    try:
        left_over = fs.upper.find(TMP_ROOT)
    except FileNotFoundError:
        left_over = []
    assert left_over == []


def test_whiteout_names_cannot_collide_with_user_files(fs):
    # names resembling whiteout markers are ordinary user files
    fs.pipe_file(".wh.top.txt", b"legacy-style")
    fs.pipe_file(WHITEOUT_ROOT + ".txt", b"similar")
    assert fs.cat_file(".wh.top.txt") == b"legacy-style"
    assert fs.cat_file(WHITEOUT_ROOT + ".txt") == b"similar"

    fs.rm_file("top.txt")
    # the marker itself is not visible or addressable as a user file
    assert WHITEOUT_ROOT not in fs.ls("", detail=False)
    with pytest.raises(FileNotFoundError):
        fs.info(WHITEOUT_ROOT)
    with pytest.raises(FileNotFoundError):
        fs.info(WHITEOUT_ROOT + "/top.txt")
    with pytest.raises(ValueError):
        fs.pipe_file(WHITEOUT_ROOT + "/evil", b"x")
    with pytest.raises(ValueError):
        fs.mkdir(WHITEOUT_ROOT)
    # user files with marker-like names are unaffected by deletions
    assert fs.cat_file(".wh.top.txt") == b"legacy-style"


def test_lower_layer_is_never_modified(fs, lower_dir):
    before = tree_snapshot(lower_dir)
    fs.pipe_file("shared.txt", b"changed")
    fs.rm_file("top.txt")
    fs.rm("dir", recursive=True)
    fs.mkdir("dir")
    fs.mv("shared.txt", "elsewhere.txt")
    with fs.open("new.txt", "wb") as f:
        f.write(b"new")
    fs.copy("elsewhere.txt", "copied.txt")
    assert tree_snapshot(lower_dir) == before


def test_write_and_read_roundtrip(fs):
    with fs.open("a/b/c.bin", "wb") as f:
        f.write(b"payload")
    assert fs.isfile("a/b/c.bin")
    assert fs.cat_file("a/b/c.bin") == b"payload"

    with fs.open("a/b/c.bin", "rb") as f:
        assert f.read() == b"payload"

    fs.write_text("a/b/text.txt", "hello")
    assert fs.read_text("a/b/text.txt") == "hello"

    fs.touch("empty.txt")
    assert fs.cat_file("empty.txt") == b""

    with pytest.raises(FileExistsError):
        fs.pipe_file("a/b/c.bin", b"x", mode="create")


def test_rm_semantics(fs):
    fs.pipe_file("d1/f.txt", b"f")
    fs.makedirs("d1/empty")
    with pytest.raises(OSError):
        fs.rm("d1", recursive=False)
    fs.rm("d1/empty")
    assert not fs.exists("d1/empty")
    fs.rm("d1", recursive=True)
    assert not fs.exists("d1")
    with pytest.raises(FileNotFoundError):
        fs.rm("does-not-exist", recursive=True)


def test_info_and_type_checks(fs):
    assert fs.isdir("")
    assert fs.isdir("dir")
    assert fs.isfile("top.txt")
    assert fs.info("top.txt")["size"] == len(b"top-lower")
    fs.pipe_file("top.txt", b"12345")
    assert fs.info("top.txt")["size"] == 5
    assert not fs.exists("nope")


def test_protocol_registered(lower_dir, tmp_path):
    fs = fsspec.filesystem(
        "overlay",
        lower=str(lower_dir),
        upper=str(tmp_path / "upper"),
        skip_instance_cache=True,
    )
    assert isinstance(fs, OverlayFileSystem)
    assert fs.cat_file("top.txt") == b"top-lower"


def test_default_upper_is_independent_memory(lower_dir):
    fs = OverlayFileSystem(lower=str(lower_dir), skip_instance_cache=True)
    assert isinstance(fs.upper, MemoryFileSystem)
    fs.pipe_file("new.txt", b"n")
    assert fs.cat_file("new.txt") == b"n"
    # the global memory store is not polluted
    assert not MemoryFileSystem().exists("new.txt")


def test_memory_lower_layer(tmp_path):
    lower = MemoryFileSystem(global_store=False, skip_instance_cache=True)
    lower.pipe_file("base.txt", b"base")
    fs = OverlayFileSystem(
        lower=lower, upper=str(tmp_path / "upper"), skip_instance_cache=True
    )
    assert fs.cat_file("base.txt") == b"base"
    fs.rm_file("base.txt")
    assert not fs.exists("base.txt")
    assert lower.cat_file("base.txt") == b"base"
