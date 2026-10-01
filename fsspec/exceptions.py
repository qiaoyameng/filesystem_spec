"""
fsspec user-defined exception classes
"""

import asyncio
from enum import Enum


class ChangeSetState(str, Enum):
    """Lifecycle state of a mapper change set."""

    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"
    FAILED = "failed"


class ChangeSetError(Exception):
    """Base class for change-set errors."""

    def __init__(self, *args, state=None, result=None):
        super().__init__(*args)
        self.state = state
        self.result = result


class ChangeSetValidationError(ChangeSetError, ValueError):
    """The change-set request is invalid."""


class ChangeSetUnsupported(ChangeSetError, NotImplementedError):
    """The backend cannot commit the change set atomically."""


class ChangeSetPreconditionFailed(ChangeSetError):
    """A mapper key is no longer at the required version."""


class ChangeSetCommitError(ChangeSetError):
    """The backend returned an error while committing a change set."""


class BlocksizeMismatchError(ValueError):
    """
    Raised when a cached file is opened with a different blocksize than it was
    written with
    """


class FSTimeoutError(asyncio.TimeoutError):
    """
    Raised when a fsspec function timed out occurs
    """
