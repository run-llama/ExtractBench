"""Execution budgets shared by ExtractBench and Pro; scoring cutoffs are separate."""

import asyncio
import threading
import time
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

EXTRACT_HARD_TIMEOUT_S = 3 * 60 * 60
DEFAULT_POLL_INTERVAL_S = 3.0
_Result = TypeVar("_Result")
_CURRENT_BUDGET: ContextVar["ExecutionBudget | None"] = ContextVar("extract_execution_budget", default=None)


class ExecutionBudget:
    """One monotonic deadline shared by a document's stages and retries.

    Cancellation wakes synchronous waits and cancels registered async tasks.
    HTTP callers use remaining() to avoid restarting a full timeout per poll.
    """

    def __init__(self, seconds: float = EXTRACT_HARD_TIMEOUT_S) -> None:
        self.deadline = time.monotonic() + seconds
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._tasks: dict[asyncio.Task[Any], asyncio.AbstractEventLoop] = {}

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise TimeoutError("Document execution deadline exceeded")
        return remaining

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            for task, loop in self._tasks.items():
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # A loop can close while its task is being unregistered.
                    # Other loops and provider cancellation still need signaling.
                    if not loop.is_closed():
                        raise

    def wait(self, seconds: float) -> None:
        self.cancelled.wait(min(seconds, self.remaining()))
        self.remaining()

    @contextmanager
    def activate(self) -> Generator[None]:
        self.remaining()
        token = _CURRENT_BUDGET.set(self)
        try:
            yield
        finally:
            _CURRENT_BUDGET.reset(token)

    async def run_async(self, coroutine: Coroutine[Any, Any, _Result]) -> _Result:
        task = asyncio.create_task(coroutine)
        with self._lock:
            self._tasks[task] = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(task, self.remaining())
        except asyncio.CancelledError:
            if self.cancelled.is_set():
                raise TimeoutError("Document execution deadline exceeded") from None
            raise
        finally:
            task.cancel()
            with self._lock:
                self._tasks.pop(task, None)


def execution_budget() -> ExecutionBudget:
    """Use the runner's budget, or a standalone budget for direct provider calls."""
    budget = _CURRENT_BUDGET.get()
    return ExecutionBudget() if budget is None else budget
