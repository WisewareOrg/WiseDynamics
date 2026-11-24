"""The Basilisk module that holds one Python thread state on a simulation's thread for its run."""

import ctypes
import threading

from Basilisk.architecture import sysModel

__all__ = ["ThreadStateHold"]

# PyDLL, not CDLL: ctypes keeps the GIL held across a PyDLL call, which these C-API calls require.
# Prototypes are set on a handle of this module's own to the interpreter's library, not on
# ctypes.pythonapi, whose function objects others share.
_PYTHON_API = ctypes.PyDLL(ctypes.pythonapi._name, handle=ctypes.pythonapi._handle)
_gil_ensure = _PYTHON_API.PyGILState_Ensure
_gil_ensure.restype = ctypes.c_int
_gil_release = _PYTHON_API.PyGILState_Release
_gil_release.argtypes = [ctypes.c_int]
_thread_state_get = _PYTHON_API.PyThreadState_Get
_thread_state_get.restype = ctypes.c_void_p
_thread_state_clear = _PYTHON_API.PyThreadState_Clear
_thread_state_clear.argtypes = [ctypes.c_void_p]


class ThreadStateHold(sysModel.SysModel):
    """Holds one Python thread state on the simulation's thread from its reset to its last call.

    Basilisk runs a simulation on a thread of its own, which has no Python thread state between
    calls into Python, so each call creates one and destroys it. `Reset` takes one more on that
    thread, so every later call reuses it, and `UpdateState`, called once at the run's final
    instant, gives it back. `abandon` clears what a run that raised still holds, and does nothing
    once the state was given back.
    """

    def __init__(self) -> None:
        """Hold nothing yet."""
        super().__init__()
        self._gil_state: int | None = None
        self._thread: int | None = None
        self._thread_state: int | None = None

    def Reset(self, CurrentSimNanos: int) -> None:
        """Take a Python thread state on the simulation's thread, once.

        Args:
            CurrentSimNanos: The reset time, in nanoseconds, as Basilisk passes it.
        """
        if self._gil_state is None:
            self._thread = threading.get_ident()
            self._gil_state = _gil_ensure()
            self._thread_state = _thread_state_get()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Give back the thread state `Reset` took.

        Args:
            CurrentSimNanos: The current simulation time, in nanoseconds, as Basilisk passes it.

        Raises:
            RuntimeError: Called on another thread than `Reset`, where giving it back would
                corrupt the interpreter.
        """
        if self._gil_state is None:
            return
        if threading.get_ident() != self._thread:
            raise RuntimeError("the thread state is given back on another thread than the one that took it")
        _gil_release(self._gil_state)
        self._gil_state = None

    def abandon(self) -> None:
        """Clear the thread state a run that raised still holds, so it keeps no object alive.

        That state still holds the error of the failed call, and so everything its traceback
        reaches. Clearing it from another thread is safe because nothing runs Python on it again:
        Basilisk's thread returns once a call on it raises, and a run that raised on the main thread
        never steps its simulation again. Only its own thread may delete it, so it stays allocated,
        in the interpreter's list of thread states, for the life of the process; a debug build of
        CPython, which asserts a thread state is cleared once, aborts at exit after such a run.
        """
        if self._gil_state is None:
            return
        _thread_state_clear(self._thread_state)
        self._gil_state = None
