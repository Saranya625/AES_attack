
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>
#include "tables.h"

__global__ void test()
{
	uint32_t a = 0xfffefd01, b = 0x00000000, c = 0x00000000;
	b = __byte_perm(a, b, 0x1456);
	// BytePerm(a, 21554);
	printf("------------------------------------\n");
	printf("%08x %08x\n", a, b);
}

void testAES(char* keyBuf)
{
	
    double kernelSpeed2 = 0;
    cudaEvent_t start, stop;
    float miliseconds = 0;

    uint32_t *gpuBuf, *outBuf, *inBuf;
    uint32_t *dev_outBuf, *dev_rk, *dev_inBuf;
    uint32_t *d_timer;
    uint32_t *h_timer;

    char* m_EncryptKey = (char*)malloc(16 * 11 * sizeof(char));

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocHost((void**)&gpuBuf, msgSize * sizeof(uint32_t));
    cudaMallocHost((void**)&outBuf, msgSize * sizeof(uint32_t));
    cudaMallocHost((void**)&inBuf, msgSize * sizeof(uint32_t));

    cudaMalloc((void**)&dev_outBuf, msgSize * sizeof(uint32_t));
    cudaMalloc((void**)&dev_inBuf, msgSize * sizeof(uint32_t));
    cudaMalloc((void**)&dev_rk, 60 * sizeof(uint32_t));
    cudaMalloc((void**)&d_timer, msgSize * sizeof(uint32_t));

    h_timer = (uint32_t*)malloc(msgSize * sizeof(uint32_t));

    memset(outBuf, 0, msgSize * sizeof(uint32_t));
    cudaMemset(dev_outBuf, 0, msgSize * sizeof(uint32_t));
    cudaMemset(d_timer, 0, msgSize * sizeof(uint32_t));

    for (int i = 0; i < 11 * 16; i++)
        m_EncryptKey[i] = 0;

    AESPrepareKey(m_EncryptKey, keyBuf, 128);

    /* ================= AES TABLES ================= */

    uint8_t *SAES_d;
    uint32_t *t0, *t1, *t2, *t3;
    uint32_t *dt0, *dt1, *dt2, *dt3;
    uint32_t *pret2, *pret3;
    uint32_t *dev_pret2, *dev_pret3;

    cudaMallocHost((void**)&t0, TABLE_SIZE * sizeof(uint32_t));
    cudaMallocHost((void**)&t1, TABLE_SIZE * sizeof(uint32_t));
    cudaMallocHost((void**)&t2, TABLE_SIZE * sizeof(uint32_t));
    cudaMallocHost((void**)&t3, TABLE_SIZE * sizeof(uint32_t));
    cudaMallocHost((void**)&pret2, pret2Size * sizeof(uint32_t));
    cudaMallocHost((void**)&pret3, pret3Size * sizeof(uint32_t));

    cudaMalloc((void**)&dt0, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&dt1, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&dt2, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&dt3, TABLE_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&dev_pret2, pret2Size * sizeof(uint32_t));
    cudaMalloc((void**)&dev_pret3, pret3Size * sizeof(uint32_t));

    cudaMallocManaged(&SAES_d, 256 * sizeof(uint8_t));

    for (int i = 0; i < TABLE_SIZE; i++) {
        t0[i] = T0[i];
        t1[i] = T1[i];
        t2[i] = T2[i];
        t3[i] = T3[i];
    }

    for (int i = 0; i < 256; i++)
        SAES_d[i] = SAES[i];

    /* ================= TRACE FILE ================= */

    FILE* traceFile = fopen("aes_traces.csv", "w");
    fprintf(traceFile, "plaintext,timer\n");

    srand(time(NULL));

    /* ================= MAIN LOOP ================= */

    for (int iter = 0; iter < ITERATION; iter++)
    {
        /* Random plaintext */
        for (int i = 0; i < msgSize; i++)
            inBuf[i] = ((uint32_t)rand() << 16) ^ rand();

        cudaMemcpy(dev_rk, m_EncryptKey, 60 * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_inBuf, inBuf, msgSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dt0, t0, TABLE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dt1, t1, TABLE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dt2, t2, TABLE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dt3, t3, TABLE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pret2, pret2, pret2Size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_pret3, pret3, pret3Size * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaMemset(d_timer, 0, msgSize * sizeof(uint32_t));
        cudaMemset(dev_outBuf, 0, msgSize * sizeof(uint32_t));

        cudaEventRecord(start);

        encGPUshared<<<gridSize / REPEAT, threadSize>>>(
            dev_outBuf,
            dev_rk,
            dev_inBuf,
            dev_pret3,
            d_timer
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaMemcpy(gpuBuf, dev_outBuf, msgSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_timer, d_timer, msgSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_timer, d_timer, msgSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&miliseconds, start, stop);
        kernelSpeed2 += 8 * (4 * (msgSize / 1024)) / miliseconds;

        /* Store traces */
        for (int i = 0; i < msgSize; i++)
            fprintf(traceFile, "%08x,%u\n", inBuf[i], h_timer[i]);
    }

    fclose(traceFile);

    printf("\nAES GPU (one-T): %u MB, Kernel %.4f Gbps\n",
           4 * (msgSize / 1024 / 1024),
           kernelSpeed2 / 1024 / ITERATION);

    printf("GPU Output (first 8 words):\n");
    printf("%x %x %x %x\n", gpuBuf[0], gpuBuf[1], gpuBuf[2], gpuBuf[3]);
    printf("%x %x %x %x\n", gpuBuf[4], gpuBuf[5], gpuBuf[6], gpuBuf[7]);

    printf("Traces saved to aes_traces.csv\n");
}


int main(int argc, char** argv)
{
	int i, j;
	char* user_key = (char*) malloc(16*sizeof(char));
	printf("<------ TESTING AES-128 CTR Mode ------>\n");

	if(argc==1)
	{
		printf("Use Default Key:\n");
		//key for test vector, FIPS-197 0x000102030405060708090A0B0C0D0E0F
		for(j=0; j<16; j++) user_key[j] = j;
	}
	else if(argc==2)
	{
		printf("New User Key:\n");
		strcpy(user_key, argv[1]);
  		for(j=0; j<16; j++) printf("%c ", user_key[j]);
  	}
  	else
  	{
  		printf("Wrong Arguments!\n");
  		return 0;
  	}


	cudaSharedMemConfig pConfig;
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);// Avoid bank conflict for 64 bit access. 
	cudaDeviceGetSharedMemConfig(&pConfig);
	//printf("Share mem config: %d\n", pConfig);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("\nGPU Compute Capability = [%d.%d], clock: %d asynCopy: %d MapHost: %d SM: %d\n",
		deviceProp.major, deviceProp.minor, deviceProp.clockRate, deviceProp.asyncEngineCount, deviceProp.canMapHostMemory, deviceProp.multiProcessorCount);
	printf("msgSize: %lu MB\t counter blocks: %u M Block\n", msgSize * 4 / 1024 / 1024, msgSize / 1024 / 1024);
	printf("%u blocks and %u threads\n", gridSize, threadSize);
	testAES(user_key);
	// cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();


	return 0;

}
