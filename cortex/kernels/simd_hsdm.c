/*
 * Cortex Native SIMD Hyperdimensional Accelerator
 * Computes 10,240-bit Hamming distance and XOR binding using unrolled 64-bit POPCNT.
 * AVX-512 / AVX2 / ARM Neon auto-vectorization target.
 */

#include <stdint.h>
#include <stddef.h>

#define WORD_COUNT 160 // 160 * 64-bit = 10,240 bits

#if defined(_MSC_VER)
#include <intrin.h>
#define POPCNT64 __popcnt64
#else
#define POPCNT64 __builtin_popcountll
#endif

uint32_t cortex_hamming_distance_10k(const uint64_t* a, const uint64_t* b) {
    uint32_t total = 0;
    
    // Unrolled 8x 64-bit words per loop iteration (512-bit stride)
    #pragma unroll(8)
    for (size_t i = 0; i < WORD_COUNT; i += 8) {
        total += POPCNT64(a[i + 0] ^ b[i + 0]);
        total += POPCNT64(a[i + 1] ^ b[i + 1]);
        total += POPCNT64(a[i + 2] ^ b[i + 2]);
        total += POPCNT64(a[i + 3] ^ b[i + 3]);
        total += POPCNT64(a[i + 4] ^ b[i + 4]);
        total += POPCNT64(a[i + 5] ^ b[i + 5]);
        total += POPCNT64(a[i + 6] ^ b[i + 6]);
        total += POPCNT64(a[i + 7] ^ b[i + 7]);
    }
    return total;
}

void cortex_bind_10k(const uint64_t* a, const uint64_t* b, uint64_t* out) {
    #pragma unroll(8)
    for (size_t i = 0; i < WORD_COUNT; i += 8) {
        out[i + 0] = a[i + 0] ^ b[i + 0];
        out[i + 1] = a[i + 1] ^ b[i + 1];
        out[i + 2] = a[i + 2] ^ b[i + 2];
        out[i + 3] = a[i + 3] ^ b[i + 3];
        out[i + 4] = a[i + 4] ^ b[i + 4];
        out[i + 5] = a[i + 5] ^ b[i + 5];
        out[i + 6] = a[i + 6] ^ b[i + 6];
        out[i + 7] = a[i + 7] ^ b[i + 7];
    }
}
