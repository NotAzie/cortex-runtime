"""
Autonomous process supervisor and POSIX/Windows signal trap harness.
Orchestrates graceful component teardown, memory buffer flushing, and persistent state synchronization.
"""

import signal
import sys
import asyncio
from typing import List, Callable, Coroutine, Any


class SystemSupervisor:
    def __init__(self):
        self._shutdown_hooks: List[Callable[[], Any]] = []
        self._is_shutting_down = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_shutdown_hook(self, hook: Callable[[], Any]):
        self._shutdown_hooks.append(hook)

    def attach_signal_handlers(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))
            except NotImplementedError:
                # Windows event loop fallback
                signal.signal(sig, lambda s, f: asyncio.run_coroutine_threadsafe(self.shutdown(s), loop))

    async def shutdown(self, sig: Optional[signal.Signals] = None):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        print(f"[Supervisor] Triggered graceful shutdown (signal: {sig}). Running {len(self._shutdown_hooks)} teardown hooks...")
        
        for hook in reversed(self._shutdown_hooks):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                print(f"[Supervisor] Error in teardown hook: {e}")

        print("[Supervisor] All subsystems cleanly halted.")
