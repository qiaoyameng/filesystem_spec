import uuid

from ..registry import filesystem
from ..spec import AbstractFileSystem

__all__ = ("OverlayFileSystem",)


class OverlayFileSystem(AbstractFileSystem):
    """Copy-on-write overlay of a read-only lower layer and a writable upper layer.

    Reads are served from the upper layer when a path exists there, and fall
    back to the lower (base) layer otherwise. All mutations go to the upper
    layer only; the lower layer is never written to:

    - writing to a path that only exists in the lower layer first copies the
      lower content up (copy-up), so appending preserves the original data;
    - deleting a path that exists in the lower layer records a *whiteout*
      marker file in the upper layer, which hides the lower entry from
      ``ls``/``find``/``info`` without touching the base data;
    - truncating writes (``"w"``/``"x"`` modes) are staged in a hidden
      temporary file in the upper layer and renamed into place on close, so
      a failed write never leaves a visible partial file.

    Whiteout and temporary files are named with the reserved ``.wh.`` prefix.
    They are filtered out of every listing, and user paths with that prefix
    are rejected on write, so markers can never be confused with real files.

    A directory removed from the lower layer and later re-created stays empty
    (its whiteout keeps hiding the lower descendants, like an opaque
    directory in Linux overlayfs), and renaming a directory never resurrects
    descendants that were deleted before the rename.
    """

    protocol = "overlay"
    whiteout_prefix = ".wh."
    _tmp_prefix = ".wh..tmp-"

    def __init__(
        self,
        lower=None,
        upper=None,
        lower_path=None,
        upper_path=None,
        lower_protocol=None,
        lower_options=None,
        upper_protocol=None,
        upper_options=None,
        **storage_options,
    ):
        """
        Parameters
        ----------
        lower: AbstractFileSystem
            Instantiated read-only base filesystem. Never modified.
        upper: AbstractFileSystem
            Instantiated writable filesystem receiving all changes.
            Defaults to an in-memory filesystem.
        lower_path, upper_path: str
            If given, restrict the respective layer to this directory by
            wrapping the filesystem in a DirFileSystem.
        lower_protocol, lower_options, upper_protocol, upper_options:
            Used to construct ``lower``/``upper`` when instances are not
            passed directly. ``lower`` defaults to the local filesystem,
            ``upper`` to an in-memory filesystem.
        """
        super().__init__(**storage_options)
        if lower is None:
            lower = filesystem(lower_protocol or "file", **(lower_options or {}))
        if upper is None:
            upper = filesystem(upper_protocol or "memory", **(upper_options or {}))
        if lower_path is not None:
            from .dirfs import DirFileSystem

            lower = DirFileSystem(path=lower_path, fs=lower)
        if upper_path is not None:
            from .dirfs import DirFileSystem

            upper = DirFileSystem(path=upper_path, fs=upper)
        self.lower = lower
        self.upper = upper
        # make sure the upper root exists so files can be staged at the
        # top level; some backends (e.g. plain local without a root path)
        # have no creatable root, which is fine to ignore
        try:
            self.upper.makedirs("", exist_ok=True)
        except OSError:
            pass

    # Path and whiteout helpers

    def _rel(self, path):
        """Normalize a user or backend path to the overlay namespace."""
        return self._strip_protocol(path).lstrip("/")

    def _whiteout(self, path):
        """Whiteout marker path in the upper layer for the given path."""
        parent, _, name = path.rpartition("/")
        marker = self.whiteout_prefix + name
        return f"{parent}/{marker}" if parent else marker

    def _is_whiteout_name(self, name):
        return name.startswith(self.whiteout_prefix)

    def _is_whiteout_path(self, path):
        return self._is_whiteout_name(path.rsplit("/", 1)[-1])

    def _check_writable_name(self, path):
        if self._is_whiteout_path(path):
            raise ValueError(
                f"names starting with {self.whiteout_prefix!r} are reserved"
                " for overlay whiteout markers"
            )

    def _has_whiteout(self, path):
        return self.upper.exists(self._whiteout(path))

    def _set_whiteout(self, path):
        parent = self._parent(path).lstrip("/")
        if parent:
            self.upper.makedirs(parent, exist_ok=True)
        self.upper.pipe_file(self._whiteout(path), b"")

    def _ancestor_whiteout(self, path):
        """Whether any strict ancestor of path is whiteouted in the upper."""
        parts = [p for p in path.split("/") if p]
        for i in range(1, len(parts)):
            if self.upper.exists(self._whiteout("/".join(parts[:i]))):
                return True
        return False

    def _lower_visible(self, path):
        """Whether path is reachable in the lower layer (not whiteouted)."""
        if not path:
            return True
        return (
            not self._ancestor_whiteout(path)
            and not self._has_whiteout(path)
            and self.lower.exists(path)
        )

    def _copy_up(self, path):
        """Copy a lower-layer file into the upper layer, if not already there."""
        if self.upper.exists(path):
            return
        parent = self._parent(path).lstrip("/")
        if parent:
            self.upper.makedirs(parent, exist_ok=True)
        self.upper.pipe_file(path, self.lower.cat_file(path))

    # Metadata

    def info(self, path, **kwargs):
        path = self._rel(path)
        if path == "":
            return {"name": "", "size": 0, "type": "directory"}
        if self._is_whiteout_path(path):
            raise FileNotFoundError(path)
        if self.upper.exists(path):
            out = self.upper.info(path).copy()
            out["name"] = path
            return out
        if self._lower_visible(path):
            out = self.lower.info(path).copy()
            out["name"] = path
            return out
        raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        path = self._rel(path)
        if self._is_whiteout_path(path):
            raise FileNotFoundError(path)
        up_exists = path != "" and self.upper.exists(path)
        up_isdir = up_exists and self.upper.isdir(path)
        low_visible = self._lower_visible(path)
        low_isdir = low_visible and self.lower.isdir(path)
        if path == "" or up_isdir or (not up_exists and low_isdir):
            # merged directory view; each layer only contributes when it
            # holds a directory here, so a same-named file in the other
            # layer is never listed as its own child
            out = {}
            if low_isdir:
                for entry in self.lower.ls(path, detail=True):
                    rel = self._rel(entry["name"])
                    if self._has_whiteout(rel):
                        continue
                    entry = entry.copy()
                    entry["name"] = rel
                    out[rel] = entry
            if up_isdir or (path == "" and self.upper.exists(path)):
                for entry in self.upper.ls(path, detail=True):
                    rel = self._rel(entry["name"])
                    if self._is_whiteout_path(rel):
                        continue
                    entry = entry.copy()
                    entry["name"] = rel
                    out[rel] = entry
            entries = [out[name] for name in sorted(out)]
        elif up_exists or low_visible:
            # ls on a file returns the file itself, like other fsspec backends
            entries = [self.info(path)]
        else:
            raise FileNotFoundError(path)
        if detail:
            return entries
        return [entry["name"] for entry in entries]

    # Directories

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._rel(path)
        self._check_writable_name(path)
        if create_parents:
            self.upper.makedirs(path, exist_ok=True)
        else:
            self.upper.mkdir(path, create_parents=False, **kwargs)
        self.invalidate_cache(path)

    def makedirs(self, path, exist_ok=False):
        path = self._rel(path)
        self._check_writable_name(path)
        if self.exists(path) and not exist_ok:
            raise FileExistsError(path)
        self.upper.makedirs(path, exist_ok=True)
        self.invalidate_cache(path)

    def rmdir(self, path):
        path = self._rel(path)
        if not self.isdir(path):
            raise FileNotFoundError(path)
        if self.ls(path):
            raise OSError(f"Directory not empty: {path}")
        # anything left in the upper dir can only be whiteout markers
        if self.upper.exists(path):
            self.upper.rm(path, recursive=True)
        if self.lower.exists(path):
            self._set_whiteout(path)
        self.invalidate_cache(self._parent(path))

    # Removal

    def rm(self, path, recursive=False, maxdepth=None):
        if isinstance(path, (list, tuple)):
            for p in path:
                self.rm(p, recursive=recursive, maxdepth=maxdepth)
            return
        path = self._rel(path)
        if not self.exists(path):
            raise FileNotFoundError(path)
        if self.isdir(path):
            if not recursive:
                raise IsADirectoryError(path)
            if self.upper.exists(path):
                self.upper.rm(path, recursive=True)
            if self.lower.exists(path):
                self._set_whiteout(path)
        else:
            if self.upper.exists(path):
                self.upper.rm_file(path)
            if self.lower.exists(path):
                self._set_whiteout(path)
        self.invalidate_cache(self._parent(path))

    def rm_file(self, path):
        self.rm(path, recursive=False)

    # Copy, move, rename

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._rel(path1)
        path2 = self._rel(path2)
        self._check_writable_name(path2)
        if not self.isfile(path1):
            raise FileNotFoundError(path1)
        parent = self._parent(path2).lstrip("/")
        if parent:
            self.upper.makedirs(parent, exist_ok=True)
        self.upper.pipe_file(path2, self.cat_file(path1))
        self.invalidate_cache(self._parent(path2))

    def _copy_tree_to_upper(self, src, dst):
        """Materialize the merged view of directory src into upper at dst."""
        self.upper.makedirs(dst, exist_ok=True)
        for name, info in self.find(src, withdirs=True, detail=True).items():
            rel = name[len(src) :].lstrip("/") if src else name
            if not rel:
                continue
            target = f"{dst}/{rel}" if dst else rel
            if info["type"] == "directory":
                self.upper.makedirs(target, exist_ok=True)
            else:
                self.upper.pipe_file(target, self.cat_file(name))

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        path1 = self._rel(path1)
        path2 = self._rel(path2)
        if path1 == path2:
            return
        self._check_writable_name(path2)
        if not self.exists(path1):
            raise FileNotFoundError(path1)
        if self.isdir(path1):
            if not recursive:
                raise ValueError("Cannot move directory without recursive=True")
            # copy the merged view so whiteouted descendants stay hidden
            self._copy_tree_to_upper(path1, path2)
            if self.upper.exists(path1):
                self.upper.rm(path1, recursive=True)
            if self.lower.exists(path1):
                self._set_whiteout(path1)
        else:
            parent = self._parent(path2).lstrip("/")
            if parent:
                self.upper.makedirs(parent, exist_ok=True)
            if self.upper.exists(path1):
                self.upper.mv(path1, path2)
            else:
                self.upper.pipe_file(path2, self.lower.cat_file(path1))
            if self.lower.exists(path1):
                self._set_whiteout(path1)
        self.invalidate_cache(self._parent(path1))
        self.invalidate_cache(self._parent(path2))

    def rename(self, path1, path2, **kwargs):
        self.mv(path1, path2, recursive=True, **kwargs)

    # File access

    def _open(self, path, mode="rb", **kwargs):
        path = self._rel(path)
        if "r" in mode and "+" not in mode:
            if self._is_whiteout_path(path):
                raise FileNotFoundError(path)
            if self.upper.exists(path):
                return self.upper.open(path, mode, **kwargs)
            if self._lower_visible(path):
                return self.lower.open(path, mode, **kwargs)
            raise FileNotFoundError(path)
        self._check_writable_name(path)
        if self.isdir(path):
            raise IsADirectoryError(path)
        if "x" in mode and self.exists(path):
            raise FileExistsError(path)
        if "w" not in mode and "x" not in mode:
            # append / read-write: preserve existing content via copy-up
            if not self.upper.exists(path) and self._lower_visible(path):
                self._copy_up(path)
            return self.upper.open(path, mode, **kwargs)
        # truncating write: stage in a hidden temp file, rename on commit
        return _OverlayCommitFile(self, path, mode, kwargs)


class _OverlayCommitFile:
    """Write-staging file: buffers into a hidden temp file in the upper layer
    and renames it over the target path on commit.

    If the file is discarded (exception in the ``with`` block, or an
    aborted transaction), the temp file is removed and no partial content
    ever becomes visible at the target path.
    """

    def __init__(self, fs, path, mode, kwargs):
        self.fs = fs
        self.path = path
        self._committed = False
        parent = fs._parent(path).lstrip("/")
        if parent:
            fs.upper.makedirs(parent, exist_ok=True)
        tmpname = fs._tmp_prefix + uuid.uuid4().hex
        self.tmp = f"{parent}/{tmpname}" if parent else tmpname
        kwargs = dict(kwargs)
        kwargs.pop("autocommit", None)
        self._file = fs.upper.open(self.tmp, mode, autocommit=True, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__dict__["_file"], name)

    def commit(self):
        if self._committed:
            return
        self._committed = True
        self._file.close()
        self.fs.upper.mv(self.tmp, self.path)
        self.fs.invalidate_cache(self.fs._parent(self.path))

    def discard(self):
        if self._committed:
            return
        self._committed = True
        try:
            self._file.close()
        finally:
            try:
                self.fs.upper.rm(self.tmp)
            except Exception:
                pass

    def close(self):
        self.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.commit()
        else:
            self.discard()
