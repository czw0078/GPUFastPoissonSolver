#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <thrust/functional.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#define PI 3.14159265

float * fastPoissonSolver(float * input, int M, int N, int P);

// cwang, run the following command in terminal
// nvcc fastPoissonSolver.cu -run -lcusparse -lcufft
int main()
{
  
  // --- Test input
  float input[27] = {
  3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0,
  3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0,
  3.0, 5.0, 1.0, 3.0, 5.0, 1.0, 3.0, 5.0, 1.0};
  int M = 3;
  int N = 3;
  int P = 3;

  // --- run the function fastPoissonSolver()
  float *result = fastPoissonSolver(input,M,N,P);

  // --- out put the result
  for(int p=0; p<P; p++)
    for(int n=0; n<N; n++)
      for(int m=0; m<M; m++)
      {
        std::cout << result[p*M*N + n*M + m] << std::endl;
      }
}

float * fastPoissonSolver(float * input, int M, int N, int P)
{
  // 1 --- Set up FFT handler
  //

  int DATASIZE = (M+1)*2;
  int BATCH = N*P;
  float scale = sqrt(DATASIZE);

  cufftHandle handle;
  int rank = 1;
  int n[] = { DATASIZE };
  int istride = 1, ostride = 1;
  int idist = DATASIZE, odist = DATASIZE;
  int inembed[] = { 0 };
  int onembed[] = { 0 };
  int batch = BATCH;
  cufftPlanMany(&handle, rank, n,
    inembed, istride, idist,
    onembed, ostride, odist, CUFFT_C2C, batch);

  // 2 --- Set up the dignal matrix
  //

  float *eta = (float*)malloc(M*sizeof(float));
  for (int m=0; m<M; m++){
    eta[m] = -600 + 120*cos((m+1)*PI/(M+1));
  }
  float *alpha = (float*)malloc(M*sizeof(float));
  for (int m=0; m<M; m++){
    alpha[m] = 60 + 36*cos((m+1)*PI/(M+1));
  }
  float *beta = (float*)malloc(M*sizeof(float));
  for (int m=0; m<M; m++){
    beta[m] = 18 + 6*cos((m+1)*PI/(M+1));
  }

  float *lambda = (float*)malloc(M*sizeof(float));
  for (int m=0; m<M; m++){
    lambda[m] = eta[m] + 2*alpha[m]*cos((m+1)*PI/(M+1));
  }
  float *miu = (float*)malloc(M*sizeof(float));
  for (int m=0; m<M; m++){
    miu[m] = alpha[m] + 2*beta[m]*cos((m+1)*PI/(M+1));
  }

  // lambda re-arrange to P*M*N
  float *h_d = (float*)malloc(P*M*N*sizeof(float));
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
      for (int p=0; p<P; p++)
      {
        h_d[n*M*P + m*P + p] = lambda[m];
      }
  // miu re-arrange to P*M*N,lower diagnol first must be zero 
  float *h_ld = (float*)malloc(P*M*N*sizeof(float));
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
    {
      h_ld[n*M*P + m*P + 0] = 0.0;
      for (int p=1; p<P; p++)
      {
        h_ld[n*M*P + m*P + p] = miu[m];
      }
    }
  // miu re-arrange to P*M*N, upper diagnol last must be zero
  float *h_ud = (float*)malloc(P*M*N*sizeof(float));
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
    {
      h_ud[n*M*P + m*P + P-1] = 0.0;
      for (int p=0; p<P-1; p++)
      {
        h_ud[n*M*P + m*P + p] = miu[m];
      }
    }

  free(eta);
  free(alpha);
  free(beta);
  free(lambda);
  free(miu);

  // send the h_d, h_ld, h_ud into cuda
  float *d_d;
  cudaMalloc((void**)&d_d, P*M*N*sizeof(float));
  cudaMemcpy(d_d, h_d, P*M*N*sizeof(float), cudaMemcpyHostToDevice);
  float *d_ld;
  cudaMalloc((void**)&d_ld, P*M*N*sizeof(float));
  cudaMemcpy(d_ld, h_ld, P*M*N*sizeof(float), cudaMemcpyHostToDevice);
  float *d_ud;
  cudaMalloc((void**)&d_ud, P*M*N*sizeof(float));
  cudaMemcpy(d_ud, h_ud, P*M*N*sizeof(float), cudaMemcpyHostToDevice);

  // 3 --- Set up Host input for first FFT
  //

  cufftComplex *hostFFTData = (cufftComplex*)malloc(DATASIZE*BATCH*sizeof(cufftComplex));

  for (int i=0; i<BATCH; i++)
    for (int j=0; j < DATASIZE; j++){
      hostFFTData[i*DATASIZE + j] = make_cuComplex(0.0f, 0.0f);
    }

  float each = 0.0f;
  for (int i=0; i<BATCH; i++)
    for (int m=0; m < M; m++){
      each = input[i*M + m];
      // hostFFT DATASISE(m)*BATCH
      hostFFTData[i*DATASIZE + m + 1 ] = make_cuComplex(0.0f, each);
      hostFFTData[i*DATASIZE + DATASIZE - 1 - m] = make_cuComplex(0.0f, - each);
    }

  // device
  cufftComplex *deviceFFTData;
  cudaMalloc((void**)&deviceFFTData, DATASIZE * BATCH * sizeof(cufftComplex));
  cudaMemcpy(deviceFFTData, hostFFTData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyHostToDevice);
  
  // Batched 1D FFTs
  cufftExecC2C(handle, deviceFFTData, deviceFFTData, CUFFT_FORWARD);

  // Send the result back
  cudaMemcpy(hostFFTData, deviceFFTData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  // Get real part from host FFT DATASIZE(M)*N*P result and rearrange into P*M*N
  float *hostData = (float*)malloc(P*M*N*sizeof(float));
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
      for (int p=0; p<P; p++)
      {
        hostData[n*M*P + m*P + p] = hostFFTData[p*DATASIZE*N + n*DATASIZE + m + 1].x/scale;
      }

  // 4 --- Start solving trignal system
  //

  // Device send hostData to deviceData
  float *deviceData;
  cudaMalloc((void**)&deviceData, P*M*N * sizeof(float));
  cudaMemcpy(deviceData, hostData, P*M*N * sizeof(float), cudaMemcpyHostToDevice);
  
  // Initialize cuSPASE
  cusparseHandle_t handle2;
  cusparseCreate(&handle2);

  // Solve Batched Trignal System
  cusparseSgtsvStridedBatch(handle2, P, d_ld, d_d, d_ud, deviceData, M*N, P);

  // send the device Data back to host
  cudaMemcpy(hostData, deviceData, P*M*N * sizeof(float), cudaMemcpyDeviceToHost);

  // 5 --- Set up Host input for second time FFT
  //

  // init 
  for (int i=0; i<BATCH; i++)
    for (int j=0; j < DATASIZE; j++){
      hostFFTData[i*DATASIZE + j] = make_cuComplex(0.0f, 0.0f);
    }

  // current host is in P*M*N shape
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
      for (int p=0; p<P; p++)
      {
        each = hostData[n*P*M + m*P + p];
        // the hostFFTData is DATASIZE(M)*N*P
        hostFFTData[p*DATASIZE*N + n*DATASIZE + m + 1 ] = make_cuComplex(0.0f, each);
        hostFFTData[p*DATASIZE*N + n*DATASIZE + DATASIZE - 1 - m] = make_cuComplex(0.0f, - each);
      }

  // send them into device
  cudaMemcpy(deviceFFTData, hostFFTData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyHostToDevice);

  // sencond time 1-D batched FFT
  cufftExecC2C(handle, deviceFFTData, deviceFFTData, CUFFT_FORWARD); //No need to use CUFFT_INVERSE

  // again Send the result back
  cudaMemcpy(hostFFTData, deviceFFTData, DATASIZE * BATCH * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

  // Get real part from host FFT DATASIZE(M)*N*P result and rearrange into M*N*P
  // final result:
  for (int n=0; n<N; n++)
    for (int m=0; m<M; m++)
      for (int p=0; p<P; p++)
      {
        hostData[p*M*N + n*M + m] = hostFFTData[p*DATASIZE*N + n*DATASIZE + m + 1].x/scale;
      }

  free(hostFFTData);
  free(h_d);
  free(h_ld);
  free(h_ud);

  cufftDestroy(handle);
  cusparseDestroy(handle2);
  cudaFree(deviceFFTData);
  cudaFree(deviceData);
  cudaFree(d_d);
  cudaFree(d_ld);
  cudaFree(d_ud);

  return hostData;
}












