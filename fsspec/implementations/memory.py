from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from errno import ENOTEMPTY
from io import BytesIO
from pathlib import PurePath, PureWindowsPath
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.exceptions import (
    ChangeSetPreconditionFailed,
    ChangeSetValidationError,
)
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import stringify_path, tokenize

logger = logging.getLogger("fsspec.memoryfs")


class MemoryFileSystem(AbstractFileSystem):
    """A filesystem based on a dict of BytesIO objects

    By default, instances share a global in-memory filesystem. Pass
    ``global_store=False, skip_instance_cache=True`` to create a new instance
    with an independent store instead.
    """

    store: dict[str, Any] = {}  # shared by default
    pseudo_dirs = [""]  # shared by default
    protocol = "memory"
    root_marker = "/"
    supports_atomic_changesets = True
    _changeset_lock = threading.RLock()
    _file_version_counter = 0

    def __init__(self, *args, global_store=True, **kwargs):
        """Create a memory filesystem.

        Parameters
        ----------
        global_store: bool
            Share files and directories with other default instances. If False,
            each instance starts with an empty, independent store. The normal
            instance cache still applies; pass ``skip_instance_cache=True``
            to create a new instance on every call. Pickling an
            independent filesystem copies its files and directories; pickling a
            global filesystem retains the reference to the global store.
        """
        super().__init__(*args, **kwargs)
        self.global_store = global_store
        if not global_store:
            self.store = {}
            self.pseudo_dirs = [""]
            self._changeset_lock = threading.RLock()
            self._file_version_counter = 0

    def _next_file_version(self):
        with self._changeset_lock:
            self._file_version_counter += 1
            return str(self._file_version_counter)

    def commit_changeset(self, writes=None, deletes=(), preconditions=None):
        writes = {self._strip_protocol(k): v for k, v in (writes or {}).items()}
        deletes = [self._strip_protocol(path) for path in deletes]
        preconditions = {
            self._strip_protocol(path): version
            for path, version in (preconditions or {}).items()
        }

        with self._changeset_lock:
            for path, required in preconditions.items():
                current = self.store[path].ukey if path in self.store else None
                if current != required:
                    raise ChangeSetPreconditionFailed(
                        f"precondition failed for {path!r}"
                    )

            try:
                new_store = self.store.copy()
                for path, value in writes.items():
                    if path in self.pseudo_dirs:
                        raise IsADirectoryError(path)
                    parent = self._parent(path)
                    while parent.strip("/"):
                        if parent in new_store:
                            raise NotADirectoryError(parent)
                        parent = self._parent(parent)
                    new_store[path] = MemoryFile(self, path, value)

                for path in deletes:
                    if path not in new_store:
                        raise FileNotFoundError(path)
                    del new_store[path]
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
            ) as exc:
                raise ChangeSetValidationError(str(exc)) from exc

            if self.global_store:
                type(self).store = new_store
            else:
                self.store = new_store

            result = {path: new_store[path].ukey for path in writes}
            result.update({path: None for path in deletes})
            return result

    @property
    def _fs_token(self):
        if self.global_store:
            return super()._fs_token
        return tokenize(super()._fs_token, id(self))

    def __reduce__(self):
        reduced = super().__reduce__()
        if self.global_store:
            return reduced
        factory, (cls, args, kwargs) = reduced
        # Restore into a new instance even when the source is cached.
        kwargs = {**kwargs, "skip_instance_cache": True}
        # Pickle's memo preserves the cycle through each MemoryFile.fs.
        return (
            factory,
            (cls, args, kwargs),
            {"store": self.store, "pseudo_dirs": self.pseudo_dirs},
        )

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, PurePath):
            if isinstance(path, PureWindowsPath):
                return LocalFileSystem._strip_protocol(path)
            else:
                path = stringify_path(path)

        path = path.removeprefix("memory://")
        if "::" in path or "://" in path:
            return path.rstrip("/")
        path = path.lstrip("/").rstrip("/")
        return "/" + path if path else ""

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        # The base implementation calls ls() once per directory, and each ls()
        # scans the whole store, giving O(n_dirs * n_entries) behaviour
        # for a tree. Since the store is a flat mapping of every path, the same
        # result can be produced with a single pass over it.
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")
        path = self._strip_protocol(path)
        if path in self.store:
            # path is itself a file
            if not detail:
                return [path]
            filelike = self.store[path]
            return {
                path: {
                    "name": path,
                    "size": filelike.size,
                    "type": "file",
                    "created": filelike.created.timestamp(),
                }
            }

        # Uniform prefix so that the search root "" (the filesystem root) and a
        # nested path are handled the same way; rel depth is rel.count("/") + 1.
        prefix = path + "/" if path else "/"
        out = {}
        dirs = {}

        def add_ancestor_dirs(name):
            # Register every directory implied between ``path`` and ``name`` that
            # is within maxdepth, mirroring how walk() surfaces implied dirs.
            idx = name.rfind("/")
            while idx > len(path):
                parent = name[:idx]
                if parent in dirs:
                    break
                rel = parent[len(prefix) :]
                if maxdepth is None or rel.count("/") + 1 <= maxdepth:
                    dirs[parent] = {"name": parent, "size": 0, "type": "directory"}
                idx = parent.rfind("/")

        # `store` may be shared by multiple MemoryFileSystem instances, so iterate a
        # snapshot: a concurrent create/delete would otherwise raise
        # "dictionary changed size during iteration". ls() does the same.
        for name, filelike in tuple(self.store.items()):
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if withdirs:
                add_ancestor_dirs(name)
            if maxdepth is not None and rel.count("/") + 1 > maxdepth:
                continue
            out[name] = {
                "name": name,
                "size": filelike.size,
                "type": "file",
                "created": filelike.created.timestamp(),
            }

        if withdirs:
            # Explicitly-created (possibly empty) directories live in pseudo_dirs.
            for pdir in self.pseudo_dirs:
                if pdir and pdir.startswith(prefix):
                    add_ancestor_dirs(pdir + "/")
            out.update(dirs)
            # Mirror the base find(): include the search root itself when it is
            # a directory (needed for posix glob compliance).
            if path != "" and self.isdir(path):
                out[path] = self.info(path)

        names = sorted(out)
        return {name: out[name] for name in names} if detail else names

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store:
            # there is a key with this exact name
            if not detail:
                return [path]
            return [
                {
                    "name": path,
                    "size": self.store[path].size,
                    "type": "file",
                    "created": self.store[path].created.timestamp(),
                }
            ]
        paths = set()
        starter = path + "/"
        out = []
        for p2 in tuple(self.store):
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child
                    out.append(
                        {
                            "name": p2,
                            "size": self.store[p2].size,
                            "type": "file",
                            "created": self.store[p2].created.timestamp(),
                        }
                    )
                elif len(p2) > len(starter):
                    # implied child directory
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out = out or []
                        out.append(
                            {
                                "name": ppath,
                                "size": 0,
                                "type": "directory",
                            }
                        )
                        paths.add(ppath)
        for p2 in self.pseudo_dirs:
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child pdir
                    if p2 not in paths:
                        out.append({"name": p2, "size": 0, "type": "directory"})
                        paths.add(p2)
                else:
                    # directory implied by deeper pdir
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out.append({"name": ppath, "size": 0, "type": "directory"})
                        paths.add(ppath)
        if not out:
            if path in self.pseudo_dirs:
                # empty dir
                return []
            raise FileNotFoundError(path)
        if detail:
            return out
        return sorted([f["name"] for f in out])

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store or path in self.pseudo_dirs:
            raise FileExistsError(path)
        if self._parent(path).strip("/") and self.isfile(self._parent(path)):
            raise NotADirectoryError(self._parent(path))
        if create_parents and self._parent(path).strip("/"):
            try:
                self.mkdir(self._parent(path), create_parents, **kwargs)
            except FileExistsError:
                pass
        if path and path not in self.pseudo_dirs:
            self.pseudo_dirs.append(path)

    def makedirs(self, path, exist_ok=False):
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if not exist_ok:
                raise

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        """Set the bytes of given file

        Avoids copies of the data if possible
        """
        mode = "xb" if mode == "create" else "wb"
        self.open(path, mode=mode, data=value)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        if path == "":
            # silently avoid deleting FS root
            return
        if path in self.pseudo_dirs:
            if not self.ls(path):
                self.pseudo_dirs.remove(path)
            else:
                raise OSError(ENOTEMPTY, "Directory not empty", path)
        else:
            raise FileNotFoundError(path)

    def info(self, path, **kwargs):
        logger.debug("info: %s", path)
        path = self._strip_protocol(path)
        if path in self.pseudo_dirs or any(
            p.startswith(path + "/") for p in list(self.store) + self.pseudo_dirs
        ):
            return {
                "name": path,
                "size": 0,
                "type": "directory",
            }
        elif path in self.store:
            filelike = self.store[path]
            return {
                "name": path,
                "size": filelike.size,
                "type": "file",
                "created": getattr(filelike, "created", None),
            }
        else:
            raise FileNotFoundError(path)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if "x" in mode and self.exists(path):
            raise FileExistsError
        if path in self.pseudo_dirs:
            raise IsADirectoryError(path)
        parent = path
        while len(parent) > 1:
            parent = self._parent(parent)
            if self.isfile(parent):
                raise FileExistsError(parent)
        if mode in ["rb", "ab", "r+b", "a+b"]:
            if path in self.store:
                f = self.store[path]
                if "a" in mode:
                    # position at the end of file
                    f.seek(0, 2)
                else:
                    # position at the beginning of file
                    f.seek(0)
                return f
            elif "a" in mode:
                # append modes create the file if it does not exist, matching
                # builtin open() and LocalFileSystem
                m = MemoryFile(self, path, kwargs.get("data"))
                if not self._intrans:
                    m.commit()
                # position at the end of file, like the existing-file path above
                m.seek(0, 2)
                return m
            else:
                raise FileNotFoundError(path)
        elif mode in {"wb", "w+b", "xb", "x+b"}:
            if "x" in mode and self.exists(path):
                raise FileExistsError
            m = MemoryFile(self, path, kwargs.get("data"))
            if not self._intrans:
                m.commit()
            return m
        else:
            name = self.__class__.__name__
            raise ValueError(f"unsupported file mode for {name}: {mode!r}")

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        with self._changeset_lock:
            if self.isfile(path1):
                self.store[path2] = MemoryFile(
                    self, path2, self.store[path1].getvalue()
                )  # implicit copy
            elif self.isdir(path1):
                if path2 not in self.pseudo_dirs:
                    self.pseudo_dirs.append(path2)
            else:
                raise FileNotFoundError(path1)

    def cat_file(self, path, start=None, end=None, **kwargs):
        logger.debug("cat: %s", path)
        path = self._strip_protocol(path)
        try:
            return bytes(self.store[path].getbuffer()[start:end])
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def ukey(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].ukey
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def _rm(self, path):
        path = self._strip_protocol(path)
        with self._changeset_lock:
            try:
                del self.store[path]
            except KeyError as e:
                raise FileNotFoundError(path) from e

    def modified(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].modified
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def created(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].created
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def isfile(self, path):
        path = self._strip_protocol(path)
        return path in self.store

    def rm(self, path, recursive=False, maxdepth=None):
        if isinstance(path, str):
            path = self._strip_protocol(path)
        else:
            path = [self._strip_protocol(p) for p in path]
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        # A non-recursive expand keeps a literal path whether or not it exists,
        # so record what exists before anything is deleted.
        existed = {p for p in paths if self.exists(p)}
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            elif not self.exists(p):
                # A directory that is not in self.pseudo_dirs only exists while it
                # has files under it, so deleting those files earlier in this loop
                # makes it vanish. That is expected; a path that never existed is
                # not, and raises like rm_file and the other filesystems do.
                if p not in existed:
                    raise FileNotFoundError(p)
                continue
            else:
                self.rmdir(p)


class MemoryFile(BytesIO):
    """A BytesIO which can't close and works as a context manager

    Can initialise with data. Each path should only be active once at any moment.

    No need to provide fs, path if auto-committing (default)
    """

    def __init__(self, fs=None, path=None, data=None):
        logger.debug("open file %s", path)
        self.fs = fs
        self.path = path
        self.created = datetime.now(tz=timezone.utc)
        self.modified = datetime.now(tz=timezone.utc)
        self.ukey = fs._next_file_version() if fs is not None else None
        if data:
            super().__init__(data)
            self.seek(0)

    @property
    def size(self):
        return self.getbuffer().nbytes

    def __enter__(self):
        return self

    def close(self):
        pass

    def discard(self):
        pass

    def commit(self):
        with self.fs._changeset_lock:
            self.fs.store[self.path] = self
            self.modified = datetime.now(tz=timezone.utc)
