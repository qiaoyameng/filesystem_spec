"""
fsspec user-defined exception classes
"""

import asyncio


class BlocksizeMismatchError(ValueError):
    """
    Raised when a cached file is opened with a different blocksize than it was
    written with
    """


class FSTimeoutError(asyncio.TimeoutError):
    """
    Raised when a fsspec function timed out occurs
    """


#: Outcome of a single changeset entry, as reported in ``ChangesetError.statuses``
#: and returned by ``apply_changeset`` on success.
#:
#: - "applied": the write/delete definitely took effect
#: - "failed": the operation was attempted and raised; it did not take effect
#: - "skipped": the operation did not take effect: it was never attempted
#:   (e.g. an earlier check or entry failed) or its effect was rolled back
#: - "unknown": the outcome could not be determined; the caller must re-read
#:   before retrying, so that a retry never silently overwrites a newer value
CHANGESET_STATUSES = frozenset({"applied", "failed", "skipped", "unknown"})


class ChangesetError(Exception):
    """A changeset could not be applied (completely or at all)

    Attributes
    ----------
    statuses: dict
        Maps each path (or key, when raised via ``FSMap``) of the changeset
        to one of "applied", "failed", "skipped" or "unknown", describing
        what is known about that entry after the error.
    """

    def __init__(self, message="", statuses=None):
        super().__init__(message)
        self.statuses = dict(statuses or {})


class ChangesetCollisionError(ChangesetError, ValueError):
    """Two or more changeset keys normalize to the same backend path

    Attributes
    ----------
    collisions: dict
        Maps the duplicated backend path to the list of original keys that
        normalized to it.
    """

    def __init__(self, message="", statuses=None, collisions=None):
        super().__init__(message, statuses=statuses)
        self.collisions = dict(collisions or {})


class ChangesetUnsupportedError(ChangesetError, NotImplementedError):
    """The backend cannot honour the requested changeset semantics

    Raised, for instance, when an atomic (all-or-nothing) changeset is
    requested from a filesystem that cannot provide one. Nothing was
    applied; the error is raised instead of pretending success.
    """


class ChangesetConflictError(ChangesetError):
    """One or more precondition versions did not match the current state

    Attributes
    ----------
    conflicts: dict
        Maps the path (or key, when raised via ``FSMap``) to a dict with
        "expected" and "actual" version tokens. Nothing was applied. A
        retry with the same stale token fails again rather than silently
        overwriting the newer value; re-read the current version first.
    """

    def __init__(self, message="", statuses=None, conflicts=None):
        super().__init__(message, statuses=statuses)
        self.conflicts = dict(conflicts or {})
