"""
Python ctypes binding with pure-Python fast fallback for C-SIMD Hyperdimensional Kernel.
"""

import ctypes
import os
import sys
from typing import List, Optional


class SIMDKernelBridge:
    def __init__(self):
        self._lib = None
        self._try_load_native()

    def _try_load_native(self):
        base_dir = os.path.dirname(__file__)
        possible_libs = [
            os.path.join(base_dir, "libcortex_simd.so"),
            os.path.join(base_dir, "libcortex_simd.dylib"),
            os.path.join(base_dir, "cortex_simd.dll")
        ]
        for lib_path in possible_libs:
            if os.path.exists(lib_path):
                try:
                    self._lib = ctypes.CDLL(lib_path)
                    self._lib.cortex_hamming_distance_10k.argtypes = [
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.POINTER(ctypes.c_uint64)
                    ]
                    self._lib.cortex_hamming_distance_10k.restype = ctypes.c_uint32
                    break
                except Exception:
                    self._lib = None

    def hamming_distance(self, a_words: List[int], b_words: List[int]) -> int:
        if self._lib is not None:
            a_arr = (ctypes.c_uint64 * 160)(*a_words)
            b_arr = (ctypes.c_uint64 * 160)(*b_words)
            return int(self._lib.cortex_hamming_distance_10k(a_arr, b_arr))
        
        # High-performance Python 3.10+ int.bit_count fallback
        dist = 0
        for w1, w2 in zip(a_words, b_words):
            dist += (w1 ^ w2).bit_count()
        return dist
