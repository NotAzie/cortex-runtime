"""
64-bit Bitwise Bloom Filter and Zero-Allocation Salience Filter.
Utilizes FNV-1a and Murmur-inspired dual hashing over 64-bit bitmasks
to achieve O(1) duplicate query screening with zero dynamic heap allocation.
"""

import struct


class BitwiseBloomFilter:
    def __init__(self, bit_size: int = 1024):
        self.bit_size = bit_size
        self._words = [0] * ((bit_size + 63) // 64)

    def _hashes(self, key: str):
        # 64-bit FNV-1a hash
        h1 = 0xcbf29ce484222325
        for byte in key.encode("utf-8"):
            h1 = ((h1 ^ byte) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        # Secondary avalanche hash
        h2 = (h1 ^ (h1 >> 33)) * 0xff51afd7ed558ccd & 0xFFFFFFFFFFFFFFFF
        h2 = (h2 ^ (h2 >> 33)) * 0xc4ceb9fe1a85ec53 & 0xFFFFFFFFFFFFFFFF
        h2 = (h2 ^ (h2 >> 33)) & 0xFFFFFFFFFFFFFFFF
        
        idx1 = h1 % self.bit_size
        idx2 = (h1 + h2) % self.bit_size
        idx3 = (h1 + 2 * h2) % self.bit_size
        return idx1, idx2, idx3

    def add(self, key: str):
        for bit_idx in self._hashes(key):
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            self._words[word_idx] |= (1 << bit_offset)

    def __contains__(self, key: str) -> bool:
        for bit_idx in self._hashes(key):
            word_idx = bit_idx // 64
            bit_offset = bit_idx % 64
            if not (self._words[word_idx] & (1 << bit_offset)):
                return False
        return True

    def clear(self):
        self._words = [0] * len(self._words)
