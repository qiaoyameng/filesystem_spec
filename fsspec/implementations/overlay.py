"""Overlay (copy-on-write) filesystem.

Stacks a writable *upper* filesystem on top of a read-only *lower*
filesystem, similar to OverlayFS:

- reads are served from the upper layer when present, otherwise the lower
- writes always land in the upper layer (copy-up of lower content when a
  write needs the previous bytes, e.g. append or random-access modes)
- deletions record a *whiteout* marker in the upper layer; a whiteout on a
  directory hides the whole lower subtree beneath it
- listings merge both layers, with upper entries winning name conflicts
- the lower layer is never modified

Whiteouts and temporary files live in reserved top-level directories of the
upper filesystem (``.fsspec_overlay_whiteouts`` and ``.fsspec_overlay_tmp``)
which are hidden from the merged view, so marker names can never be
confused with user files.
"""

from __future__ import annotations

import errno
import logging
import os
import posixpath
import uuid

from ..spec import AbstractFileSystem
from ..utils import stringify_path

logger = logging.getLogger("fsspec.overlay")

WHITEOUT_ROOT = ".fsspec_overlay_whiteouts"
TMP_ROOT = ".fsspec_overlay_tmp"
_RESERVED = (WHITEOUT_ROOT, TMP_ROOT)


class _AtomicCommitFile:
    """File-like proxy writing to a temporary upper-layer path.

    On successful close the temporary file is renamed onto the final
    destination; on failure (exception in the with-block, or a failed
    commit) the temporary file is discarded, so a failed write never leaves
    a visible half-written file.
    """

    def __init__(self, fs, tmp, final, fileobj):
        self._fs = fs
        self._tmp = tmp
        self._final = final
        self._f = fileobj
        self._committed = False
        self._closed = False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_f"], name)

    @property
    def closed(self):
        return self._closed

    def _discard(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._f.close()
        finally:
            self._cleanup_tmp()

    def _cleanup_tmp(self):
        try:
            if self._fs.exists(self._tmp):
                self._fs.rm(self._tmp, recursive=True)
        except Exception:  # best effort cleanup
            logger.debug("failed to remove overlay temp file %s", self._tmp)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._f.close()
            parent = self._final.rsplit("/", 1)[0] if "/" in self._final else ""
            if parent:
                self._fs.makedirs(parent, exist_ok=True)
            self._fs.mv(self._tmp, self._final)
            self._committed = True
        except Exception:
            self._cleanup_tmp()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._discard()
        return False

    def __iter__(self):
        return iter(self._f)


class OverlayFileSystem(AbstractFileSystem):
    """Copy-on-write overlay of a writable upper fs over a read-only lower fs."""

    protocol = "overlay"

    def __init__(
        self,
        lower=None,
        upper=None,
        lower_protocol=None,
        lower_options=None,
        upper_protocol=None,
        upper_options=None,
        **storage_options,
    ):
        """
        Parameters
        ----------
        lower: AbstractFileSystem | str
            The read-only base layer. Either an instantiated filesystem or a
            local directory path.
        upper: AbstractFileSystem | str | None
            The writable layer. Either an instantiated filesystem (e.g.
            ``MemoryFileSystem``) or a local directory path. Defaults to a
            fresh independent in-memory filesystem.
        lower_protocol, lower_options, upper_protocol, upper_options:
            Alternative way to construct a layer from a protocol name and
            options, when the corresponding layer is not given directly.
        """
        super().__init__(**storage_options)
        self.lower = self._make_layer(lower, lower_protocol, lower_options, False)
        if self.lower is None:
            raise ValueError("a lower (read-only base) layer is required")
        self.upper = self._make_layer(upper, upper_protocol, upper_options, True)
        if self.upper is None:
            from .memory import MemoryFileSystem

            self.upper = MemoryFileSystem(global_store=False, skip_instance_cache=True)

    @staticmethod
    def _make_layer(obj, protocol, options, writable):
        if obj is None:
            if protocol is None:
                return None
            from ..registry import filesystem

            return filesystem(protocol, **(options or {}))
        if isinstance(obj, AbstractFileSystem):
            return obj
        if isinstance(obj, (str, os.PathLike)):
            from .dirfs import DirFileSystem
            from .local import LocalFileSystem

            return DirFileSystem(
                path=stringify_path(obj),
                fs=LocalFileSystem(auto_mkdir=writable),
            )
        raise TypeError(f"cannot use {obj!r} as an overlay layer")

    # Path handling -----------------------------------------------------

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith("overlay://"):
            path = path[len("overlay://") :]
        return path

    def _rel(self, path):
        """Normalise a user path to a relative posix path ("" is the root)."""
        path = self._strip_protocol(path)
        path = posixpath.normpath("/" + path).lstrip("/")
        return "" if path == "." else path

    @staticmethod
    def _is_reserved(rel):
        """Whether a relative path falls in a reserved (internal) namespace."""
        return rel.split("/", 1)[0] in _RESERVED

    def _check_reserved(self, rel):
        if self._is_reserved(rel):
            raise ValueError(
                f"{rel!r} uses a reserved overlay-internal name ({_RESERVED})"
            )

    @staticmethod
    def _wh(rel):
        """Upper-layer path of the whiteout marker for a relative path."""
        return WHITEOUT_ROOT + "/" + rel

    def _tmp_path(self):
        return f"{TMP_ROOT}/{uuid.uuid4().hex}"

    # Whiteout handling -------------------------------------------------

    def _lower_hidden(self, rel):
        """Whether a lower-layer path is hidden by a whiteout on itself or
        any of its ancestors."""
        if not rel:
            return False
        parts = rel.split("/")
        for i in range(len(parts), 0, -1):
            marker = WHITEOUT_ROOT + "/" + "/".join(parts[:i])
            if self.upper.isfile(marker):
                return True
        return False

    def _lower_visible(self, rel):
        return not self._lower_hidden(rel) and self.lower.exists(rel)

    def _set_whiteout(self, rel):
        """Record a whiteout for ``rel`` in the upper layer."""
        marker = self._wh(rel)
        if self.upper.isdir(marker):
            # descendant whiteouts are now redundant: the new marker covers
            # the whole subtree
            self.upper.rm(marker, recursive=True)
        self.upper.makedirs(marker.rsplit("/", 1)[0], exist_ok=True)
        self.upper.pipe_file(marker, b"")

    def _drop_whiteouts(self, rel):
        """Remove any whiteout state at or below ``rel`` (best effort)."""
        marker = self._wh(rel)
        try:
            if self.upper.exists(marker):
                self.upper.rm(marker, recursive=True)
        except Exception:  # best effort cleanup
            logger.debug("failed to drop whiteout state for %s", rel)

    # Layer resolution --------------------------------------------------

    def _read_fs(self, rel):
        """The layer that should serve reads for ``rel``."""
        if self.upper.exists(rel):
            return self.upper
        if self._lower_visible(rel):
            return self.lower
        raise FileNotFoundError(rel)

    # Metadata ----------------------------------------------------------

    def info(self, path, **kwargs):
        rel = self._rel(path)
        if rel == "":
            return {"name": "", "size": 0, "type": "directory"}
        if self._is_reserved(rel):
            raise FileNotFoundError(path)
        if self.upper.exists(rel):
            out = dict(self.upper.info(rel))
        elif self._lower_visible(rel):
            out = dict(self.lower.info(rel))
        else:
            raise FileNotFoundError(path)
        out["name"] = rel
        return out

    def ls(self, path, detail=True, **kwargs):
        rel = self._rel(path)
        if self._is_reserved(rel):
            raise FileNotFoundError(path)
        top = self.info(rel)
        if top["type"] != "directory":
            return [top] if detail else [top["name"]]
        merged = {}
        if self.upper.isdir(rel):
            for entry in self.upper.ls(rel, detail=True):
                name = self._rel(entry["name"])
                if self._is_reserved(name):
                    continue
                entry = dict(entry)
                entry["name"] = name
                merged[name] = entry
        if not self._lower_hidden(rel) and self.lower.isdir(rel):
            for entry in self.lower.ls(rel, detail=True):
                name = self._rel(entry["name"])
                if name in merged or self._lower_hidden(name):
                    continue
                entry = dict(entry)
                entry["name"] = name
                merged[name] = entry
        names = sorted(merged)
        if detail:
            return [merged[name] for name in names]
        return names

    def created(self, path):
        rel = self._rel(path)
        return self._read_fs(rel).created(rel)

    def modified(self, path):
        rel = self._rel(path)
        return self._read_fs(rel).modified(rel)

    # Reading -----------------------------------------------------------

    def cat_file(self, path, start=None, end=None, **kwargs):
        rel = self._rel(path)
        return self._read_fs(rel).cat_file(rel, start=start, end=end, **kwargs)

    # Writing -----------------------------------------------------------

    def _atomic_write(self, rel, data):
        """Write bytes to an upper-layer path, atomically."""
        tmp = self._tmp_path()
        self.upper.makedirs(TMP_ROOT, exist_ok=True)
        try:
            self.upper.pipe_file(tmp, data)
            parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
            if parent:
                self.upper.makedirs(parent, exist_ok=True)
            self.upper.mv(tmp, rel)
        except Exception:
            try:
                if self.upper.exists(tmp):
                    self.upper.rm(tmp, recursive=True)
            finally:
                raise

    def _copy_up(self, src_rel, dst_rel):
        """Copy the merged-view file at ``src_rel`` to ``dst_rel`` in upper."""
        self._atomic_write(dst_rel, self.cat_file(src_rel))

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        rel = self._rel(path)
        if "r" in mode and "+" not in mode:
            return self._read_fs(rel).open(rel, mode=mode, **kwargs)

        self._check_reserved(rel)
        if not rel:
            raise IsADirectoryError(errno.EISDIR, "cannot write to root", path)
        try:
            if self.info(rel)["type"] == "directory":
                raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
        except FileNotFoundError:
            pass

        if "x" in mode:
            raise FileExistsError(errno.EEXIST, "File exists", path)

        tmp = self._tmp_path()
        if "a" in mode or "+" in mode:
            # these modes need the previous bytes: copy-up first
            try:
                data = self.cat_file(rel)
            except FileNotFoundError:
                if "r" in mode:
                    raise
            else:
                self._atomic_write(tmp, data)
        f = self.upper.open(tmp, mode=mode, **kwargs)
        return _AtomicCommitFile(self.upper, tmp, rel, f)

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        rel = self._rel(path)
        self._check_reserved(rel)
        if mode == "create" and self.exists(rel):
            raise FileExistsError(errno.EEXIST, "File exists", path)
        if mode == "append":
            try:
                value = self.cat_file(rel) + value
            except FileNotFoundError:
                pass
        self._atomic_write(rel, value)

    def touch(self, path, truncate=True, **kwargs):
        rel = self._rel(path)
        self._check_reserved(rel)
        if truncate or not self.exists(rel):
            self._atomic_write(rel, b"")
        # truncate=False on an existing file: timestamp updates are not
        # supported uniformly by the backends, so this is a no-op

    def mkdir(self, path, create_parents=True, **kwargs):
        rel = self._rel(path)
        self._check_reserved(rel)
        if rel == "":
            return
        self.upper.mkdir(rel, create_parents=create_parents, **kwargs)

    def makedirs(self, path, exist_ok=False):
        rel = self._rel(path)
        self._check_reserved(rel)
        if rel == "":
            return
        self.upper.makedirs(rel, exist_ok=exist_ok)

    # Deletion ----------------------------------------------------------

    def rm_file(self, path, **kwargs):
        rel = self._rel(path)
        self._check_reserved(rel)
        info = self.info(rel)
        if info["type"] == "directory":
            raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
        if self.upper.isfile(rel):
            self.upper.rm_file(rel)
        if self._lower_visible(rel):
            self._set_whiteout(rel)

    def rm(self, path, recursive=False, maxdepth=None, **kwargs):
        if isinstance(path, (list, tuple)):
            for p in path:
                self.rm(p, recursive=recursive, maxdepth=maxdepth, **kwargs)
            return
        rel = self._rel(path)
        self._check_reserved(rel)
        info = self.info(rel)
        if info["type"] != "directory":
            self.rm_file(rel)
            return
        if not recursive and self.ls(rel):
            raise OSError(errno.ENOTEMPTY, "Directory not empty", path)
        if self.upper.exists(rel):
            self.upper.rm(rel, recursive=True)
        if self._lower_visible(rel):
            self._set_whiteout(rel)
        else:
            self._drop_whiteouts(rel)

    # Copy / move -------------------------------------------------------

    def cp_file(self, path1, path2, **kwargs):
        rel1, rel2 = self._rel(path1), self._rel(path2)
        self._check_reserved(rel2)
        self._copy_up(rel1, rel2)

    def _dst_for(self, rel1, rel2):
        """Final destination, nesting inside ``rel2`` if it is a directory."""
        try:
            is_dir = self.info(rel2)["type"] == "directory"
        except FileNotFoundError:
            is_dir = False
        if is_dir:
            base = rel1.rsplit("/", 1)[-1]
            return rel2 + "/" + base if rel2 else base
        return rel2

    def copy(
        self, path1, path2, recursive=False, maxdepth=None, on_error=None, **kwargs
    ):
        if isinstance(path1, (list, tuple)):
            paths2 = (
                path2 if isinstance(path2, (list, tuple)) else [path2] * len(path1)
            )
            for p1, p2 in zip(path1, paths2):
                self.copy(p1, p2, recursive=recursive, maxdepth=maxdepth, **kwargs)
            return
        rel1, rel2 = self._rel(path1), self._rel(path2)
        self._check_reserved(rel2)
        info = self.info(rel1)
        if info["type"] != "directory":
            self._copy_up(rel1, self._dst_for(rel1, rel2))
            return
        if not recursive:
            raise ValueError("recursive=False is required to copy a directory")
        root = self._dst_for(rel1, rel2)
        for name, entry in sorted(self.find(rel1, withdirs=True, detail=True).items()):
            sub = name[len(rel1) :].lstrip("/")
            target = root + "/" + sub if sub else root
            if entry["type"] == "directory":
                self.upper.makedirs(target, exist_ok=True)
            else:
                self._copy_up(name, target)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        rel1, rel2 = self._rel(path1), self._rel(path2)
        self._check_reserved(rel1)
        self._check_reserved(rel2)
        info = self.info(rel1)
        if info["type"] != "directory":
            self._copy_up(rel1, self._dst_for(rel1, rel2))
            self.rm_file(rel1)
            return
        # renaming a directory: copy the *visible* merged tree up, then
        # remove the source (which whiteouts the lower subtree, so hidden
        # descendants stay hidden)
        root = self._dst_for(rel1, rel2)
        for name, entry in sorted(self.find(rel1, withdirs=True, detail=True).items()):
            sub = name[len(rel1) :].lstrip("/")
            target = root + "/" + sub if sub else root
            if entry["type"] == "directory":
                self.upper.makedirs(target, exist_ok=True)
            else:
                self._copy_up(name, target)
        self.rm(rel1, recursive=True)
