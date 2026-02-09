/**********************************************************************
 * AES-128 GPU Encryption + Last-Round Leakage (DoM Attack)
 * SAMPLE-BY-SAMPLE VERSION
 *
 * Changes from previous version:
 * ---------------------------------------------------------------
 * 1. Sample collection happens ONE SAMPLE AT A TIME.
 *    - Exactly one warp (32 threads) processes one sample.
 *    - Kernel loops over NUM_SAMPLES internally.
 *
 * 2. Attack phase also runs SAMPLE-BY-SAMPLE.
 *    - One block per key guess.
 *    - Each block has 32 threads (one warp).
 *    - Warp collaboratively computes bank conflicts
 *      for each sample sequentially.
 *
 * This removes cross-sample interference and makes timing
 * behaviour much more stable for GPU side-channel analysis.
 *********************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define AES_BLOCK 16
#define NUM_SAMPLES 2000000
#define ATTACK_BYTE 0

#define CUDA_CHECK(x) do {                                  \
    cudaError_t err = x;                                    \
    if (err != cudaSuccess) {                               \
        printf("CUDA error %s:%d: %s\n",                  \
               __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(1);                                            \
    }                                                       \
} while(0)

/* ================= SAMPLE STRUCT ================= */

typedef struct {
    uint8_t cipher[WARP_SIZE][AES_BLOCK];
    uint64_t time;
    uint8_t conflicts[256];
} Sample;

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

static uint8_t hinv_sbox[256];

/* ================= DEVICE CONSTANTS ================= */

__constant__ uint8_t d_sbox[256];
__constant__ uint8_t dinv_sbox[256];
__constant__ uint8_t d_last_key[16];

/* ================= AES HELPERS ================= */

__device__ __forceinline__ uint8_t xtime(uint8_t x){
    return (x<<1)^((x&0x80)?0x1b:0);
}

__device__ void shift_rows(uint8_t *s){
    uint8_t t;
    t=s[1]; s[1]=s[5]; s[5]=s[9]; s[9]=s[13]; s[13]=t;
    t=s[2]; s[2]=s[10]; s[10]=t;
    t=s[6]; s[6]=s[14]; s[14]=t;
    t=s[3]; s[3]=s[15]; s[15]=s[11]; s[11]=s[7]; s[7]=t;
}

__device__ void mix_columns(uint8_t *s){
    for(int i=0;i<4;i++){
        int c=4*i;
        uint8_t a=s[c],b=s[c+1],c1=s[c+2],d=s[c+3];
        uint8_t e=a^b^c1^d;
        s[c]^=e^xtime(a^b);
        s[c+1]^=e^xtime(b^c1);
        s[c+2]^=e^xtime(c1^d);
        s[c+3]^=e^xtime(d^a);
    }
}

/* ================= ENCRYPTION KERNEL ================= */

__global__ void aes_encrypt_kernel(
    uint8_t *pt,
    uint8_t *ct,
    uint64_t *time_last)
{
    int tid = threadIdx.x;   // lane id (0..31)

    __shared__ uint8_t Te4[256];

    for(int i=tid;i<256;i+=blockDim.x)
        Te4[i]=d_sbox[i];
    __syncthreads();

    // ----- PROCESS ONE SAMPLE AT A TIME -----
    for(int sample=0; sample<NUM_SAMPLES; sample++){

        int offset = sample*WARP_SIZE*AES_BLOCK + tid*AES_BLOCK;

        uint8_t state[16];
        for(int i=0;i<16;i++)
            state[i]=pt[offset+i];

        for(int r=0;r<9;r++){
            for(int i=0;i<16;i++)
                state[i]=d_sbox[state[i]];
            shift_rows(state);
            mix_columns(state);
        }
        
        __syncwarp();
        uint64_t start=clock64();

        for(int i=0;i<16;i++){
            uint8_t idx = state[i];
            uint8_t val = Te4[idx];
            state[i] = val ^ d_last_key[ATTACK_BYTE];
        }
        __syncwarp();
        uint64_t end=clock64();
        
        if(tid==0)
            time_last[sample]=end-start;

        shift_rows(state);

        for(int i=0;i<16;i++)
            ct[offset+i]=state[i];
    }
}

/* ================= ATTACK KERNEL ================= */

__global__ void attack_kernel(
    Sample *samples,
    float *sumA,
    float *sumB,
    int *countA,
    int *countB,
    int num_samples)
{
    int guess = blockIdx.x;   // one block per key guess
    int lane  = threadIdx.x;  // 0..31

    __shared__ int bank_hits[32];

    int inv_shift[16]={0,13,10,7,4,1,14,11,8,5,2,15,12,9,6,3};

    float localA=0, localB=0;
    int localCA=0, localCB=0;

    // ---- PROCESS SAMPLE BY SAMPLE ----
    for(int s=0; s<num_samples; s++){

        if(lane<32)
            bank_hits[lane]=0;
        __syncthreads();

        uint8_t c = samples[s].cipher[lane][inv_shift[ATTACK_BYTE]];

        uint8_t idx = dinv_sbox[c^guess];
        int bank = (idx>>1)&31;

        atomicAdd(&bank_hits[bank],1);
        __syncthreads();

        int max_hits=0;
        if(lane==0){
            for(int b=0;b<32;b++)
                if(bank_hits[b]>max_hits)
                    max_hits=bank_hits[b];

            int conflicts=max_hits-1;
            float time=(float)samples[s].time;
            samples[s].conflicts[guess]=conflicts;

            if(conflicts == 2){ localA+=time; localCA++; }
            else if(conflicts == 4){ localB+=time; localCB++; }
        }

        __syncthreads();
    }

    if(lane==0){
        atomicAdd(&sumA[guess],localA);
        atomicAdd(&sumB[guess],localB);
        atomicAdd(&countA[guess],localCA);
        atomicAdd(&countB[guess],localCB);
    }
}

/* ================= HOST MAIN ================= */

int main(){

    for(int i=0;i<256;i++)
        hinv_sbox[h_sbox[i]]=i;

    CUDA_CHECK(cudaMemcpyToSymbol(d_sbox,h_sbox,256));
    CUDA_CHECK(cudaMemcpyToSymbol(dinv_sbox,hinv_sbox,256));

    uint8_t last_key[16]={
        0x2b,0x7e,0x15,0x16,
        0x28,0xae,0xd2,0xa6,
        0xab,0xf7,0x15,0x88,
        0x09,0xcf,0x4f,0x3c
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_last_key,last_key,16));

    size_t pt_size=NUM_SAMPLES*WARP_SIZE*AES_BLOCK;

    uint8_t *pt=(uint8_t*)malloc(pt_size);
    uint8_t *ct=(uint8_t*)malloc(pt_size);
    uint64_t *time_last=(uint64_t*)malloc(NUM_SAMPLES*sizeof(uint64_t));

    for(size_t i=0;i<pt_size;i++)
        pt[i]=rand()&0xff;

    uint8_t *d_pt,*d_ct;
    uint64_t *d_time;

    CUDA_CHECK(cudaMalloc(&d_pt,pt_size));
    CUDA_CHECK(cudaMalloc(&d_ct,pt_size));
    CUDA_CHECK(cudaMalloc(&d_time,NUM_SAMPLES*sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_pt,pt,pt_size,cudaMemcpyHostToDevice));

    aes_encrypt_kernel<<<1,WARP_SIZE>>>(d_pt,d_ct,d_time);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ct,d_ct,pt_size,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(time_last,d_time,
                NUM_SAMPLES*sizeof(uint64_t),
                cudaMemcpyDeviceToHost));

    Sample *samples=(Sample*)malloc(NUM_SAMPLES*sizeof(Sample));

    for(int s=0;s<NUM_SAMPLES;s++){
        samples[s].time = time_last[s];

        for(int t=0;t<WARP_SIZE;t++)
            for(int i=0;i<16;i++)
                samples[s].cipher[t][i]=
                  ct[s*WARP_SIZE*16+t*16+i];
    }

    Sample *d_samples;
    float *d_sumA,*d_sumB;
    int *d_countA,*d_countB;

    CUDA_CHECK(cudaMalloc(&d_samples,
        NUM_SAMPLES*sizeof(Sample)));
    CUDA_CHECK(cudaMemcpy(d_samples,samples,
        NUM_SAMPLES*sizeof(Sample),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_sumA,256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sumB,256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_countA,256*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_countB,256*sizeof(int)));

    CUDA_CHECK(cudaMemset(d_sumA,0,256*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sumB,0,256*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_countA,0,256*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_countB,0,256*sizeof(int)));

    attack_kernel<<<256,32>>>(
        d_samples,d_sumA,d_sumB,
        d_countA,d_countB,NUM_SAMPLES);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(samples, d_samples, NUM_SAMPLES*sizeof(Sample), cudaMemcpyDeviceToHost));

    float sumA[256],sumB[256];
    int countA[256],countB[256];

    CUDA_CHECK(cudaMemcpy(sumA,d_sumA,256*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sumB,d_sumB,256*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(countA,d_countA,256*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(countB,d_countB,256*sizeof(int),cudaMemcpyDeviceToHost));

    float best_score=-1e30;
    int best_key=0;
    float all_scores[256];

    for(int k=0;k<256;k++){
        if(countA[k]==0||countB[k]==0){
            all_scores[k] = 0.0f;
            continue;
        } 

        float meanA=sumA[k]/countA[k];
        float meanB=sumB[k]/countB[k];
        float score=fabs(meanA-meanB);
        all_scores[k] = score;

        if(score>best_score){
            best_score=score;
            best_key=k;
        }
    }

    FILE *fp = fopen("scores.txt","w");

    for(int k=0;k<256;k++){
        fprintf(fp,"%d %f\n", k, all_scores[k]);
    }

    fclose(fp);


    printf("Recovered key byte = %02x\n",best_key);
    printf("Score = %f\n",best_score);
    printf("Actual key byte = %02x\n",last_key[ATTACK_BYTE]);

    printf("\nSample timing dump:\n");
    for(int s = 0; s < 32; s++) {
        if(samples[s].conflicts[best_key] == 2 || samples[s].conflicts[best_key] == 4){
            printf("Sample %d : ", s);
            printf("%llu, %llu ",(unsigned long long)time_last[s], (unsigned long long)samples[s].conflicts[best_key]);
            printf("\n");
        }
        
    }
}
