/**********************************************************************
 * AES-128 GPU Encryption + Last-Round Leakage (Keyed, Attack-Ready)
 *********************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define AES_BLOCK 16

#define CUDA_CHECK(x) do {                                  \
    cudaError_t err = x;                                    \
    if (err != cudaSuccess) {                               \
        printf("CUDA error %s:%d: %s\n",                    \
               __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);                                            \
    }                                                       \
} while(0)

/* ================= HOST SBOX ================= */
static const uint8_t h_sbox[256] = {
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

/* ================= DEVICE CONSTANTS ================= */
__constant__ uint8_t d_sbox[256];
__constant__ uint8_t d_last_key[16];   // 🔑 LAST ROUND KEY

/* ================= DEVICE HELPERS ================= */
__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return (x << 1) ^ ((x & 0x80) ? 0x1b : 0);
}

__device__ void shift_rows(uint8_t *s) {
    uint8_t t;
    t=s[1];  s[1]=s[5];  s[5]=s[9];  s[9]=s[13]; s[13]=t;
    t=s[2];  s[2]=s[10]; s[10]=t;
    t=s[6];  s[6]=s[14]; s[14]=t;
    t=s[3];  s[3]=s[15]; s[15]=s[11]; s[11]=s[7]; s[7]=t;
}

__device__ void mix_columns(uint8_t *s) {
    for (int i = 0; i < 4; i++) {
        int c = 4*i;
        uint8_t a=s[c], b=s[c+1], c1=s[c+2], d=s[c+3];
        uint8_t e = a ^ b ^ c1 ^ d;
        s[c]   ^= e ^ xtime(a ^ b);
        s[c+1] ^= e ^ xtime(b ^ c1);
        s[c+2] ^= e ^ xtime(c1 ^ d);
        s[c+3] ^= e ^ xtime(d ^ a);
    }
}

/* ================= KERNEL ================= */
__global__ void aes_encrypt_kernel(
    uint8_t *pt,
    uint8_t *ct,
    uint64_t *time_last,
    int *bank_conflicts
) {
    int tid = threadIdx.x;
    if (tid >= WARP_SIZE) return;

    __shared__ uint8_t Te4[256];
    __shared__ int bank_hits[32];

    if (tid < 256) Te4[tid] = d_sbox[tid];
    if (tid < 32) bank_hits[tid] = 0;
    __syncthreads();

    uint8_t state[16];
    for (int i = 0; i < 16; i++)
        state[i] = pt[tid*16 + i];

    /* Rounds 1–9 (no key needed for last-round attack) */
    for (int r = 0; r < 9; r++) {
        for (int i = 0; i < 16; i++)
            state[i] = d_sbox[state[i]];
        shift_rows(state);
        mix_columns(state);
    }

    __syncwarp();
    uint64_t start = clock64();

    /* ===== FINAL ROUND (KEY USED HERE) ===== */
    for (int i = 0; i < 1; i++) {
        uint8_t idx = state[i] ^ d_last_key[i];   // 🔑 KEY XOR
        atomicAdd(&bank_hits[(idx >> 1) & 31], 1);
        state[i] = Te4[idx];
    }
    shift_rows(state);

    __syncwarp();
    uint64_t end = clock64();

    if (tid == 0) {
        int max_hits = 0;
        for (int i = 0; i < 32; i++) {
            if (bank_hits[i] > max_hits)
                max_hits = bank_hits[i];
        }
        bank_conflicts[0] = max_hits - 1;
        *time_last = end - start;
    }

    for (int i = 0; i < 16; i++)
        ct[tid*16 + i] = state[i];

    
}

/* ================= HOST ================= */
int main() {
    uint8_t pt[WARP_SIZE * AES_BLOCK];
    uint8_t ct[WARP_SIZE * AES_BLOCK];
    uint64_t time;
    int conflicts;

    uint8_t last_key[16] = {
        0x2b,0x7e,0x15,0x16,
        0x28,0xae,0xd2,0xa6,
        0xab,0xf7,0x15,0x88,
        0x09,0xcf,0x4f,0x3c
    };

    for (int i = 0; i < WARP_SIZE * AES_BLOCK; i++)
        pt[i] = rand() & 0xff;

    uint8_t *d_pt,*d_ct;
    uint64_t *d_time;
    int *d_conf;

    CUDA_CHECK(cudaMemcpyToSymbol(d_sbox, h_sbox, 256));
    CUDA_CHECK(cudaMemcpyToSymbol(d_last_key, last_key, 16));

    CUDA_CHECK(cudaMalloc(&d_pt,sizeof(pt)));
    CUDA_CHECK(cudaMalloc(&d_ct,sizeof(ct)));
    CUDA_CHECK(cudaMalloc(&d_time,sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_conf,sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pt,pt,sizeof(pt),cudaMemcpyHostToDevice));

    aes_encrypt_kernel<<<1,WARP_SIZE>>>(d_pt,d_ct,d_time,d_conf);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ct,d_ct,sizeof(ct),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&time,d_time,sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&conflicts,d_conf,sizeof(int),cudaMemcpyDeviceToHost));

    printf("Ciphertext (warp 0):\n");
for (int t = 0; t < WARP_SIZE; t++) {
    printf("T%02d: ", t);
    for (int i = 0; i < 16; i++) {
        printf("%02x ", ct[t*16 + i]);
    }
    printf("\n");
}

printf("\nLast-round execution time: %llu cycles\n",
       (unsigned long long)time);
printf("Total bank conflicts (last round): %d\n", conflicts);

}

