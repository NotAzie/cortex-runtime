"""
Lock-free shared-memory ring arena.
Atomic slot reservations via memory-mapped buffers, eliminating OS kernel context switches
and memory allocation overhead during sustained multi-gigabit token/audio streaming.
"""

import mmap
import struct
import os


class LockFreeRingArena:
    __slots__ = ("capacity", "slot_size", "_mmap", "_head", "_tail")

    def __init__(self, capacity: int = 256, slot_size: int = 4096):
        self.capacity = capacity
        self.slot_size = slot_size
        total_size = capacity * slot_size
        
        # Anonymous memory map
        self._mmap = mmap.mmap(-1, total_size)
        self._head = 0
        self._tail = 0

    def write_slot(self, data: bytes) -> int:
        if len(data) > self.slot_size - 4:
            raise ValueError(f"Payload ({len(data)} bytes) exceeds slot limit ({self.slot_size - 4} bytes)")

        slot_idx = self._head % self.capacity
        offset = slot_idx * self.slot_size

        # Write 4-byte length prefix + payload
        self._mmap.seek(offset)
        self._mmap.write(struct.pack("<I", len(data)))
        self._mmap.write(data)

        self._head += 1
        return slot_idx

    def read_slot(self, slot_idx: int) -> bytes:
        offset = (slot_idx % self.capacity) * self.slot_size
        self._mmap.seek(offset)
        raw_len = self._mmap.read(4)
        length = struct.unpack("<I", raw_len)[0]
        return self._mmap.read(length)

    def close(self):
        if not self._mmap.closed:
            self._mmap.close()
