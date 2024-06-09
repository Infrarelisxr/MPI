#include <iostream>
#include <Windows.h>
#include <sys/utime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <mpi.h>
#include <random>
#include <omp.h>

using namespace std;
#define TIME_UTC 1
#define NUM_THREADS 4
#define SIMD_WIDTH 8 // AVX支持8个单精度浮点数并行计算
int N = 2500;
float** m;
void m_reset(float** m)
{
    for (int i = 0; i < N; i++)
    {
        m[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        m[i][i] = 1.0;
        for (int j = i+1; j < N; j++)
        {
            m[i][j] = rand() % 5000;
        }
    }

    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                m[i][j] += m[k][j];
                m[i][j] = (int)m[i][j] % 5000;
            }
        }
    }

}
void m_reset0(float** m) {
    m = new float* [N];
    for (int i = 0; i < N; i++) {
        m[i] = new float[N];
        memset(m[i], 0, N * sizeof(float));
    }

}

void LU(float** m)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
void LU_mpi(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;
    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    if (rank == 0) 
    {
        for (j = 1; j < numprocs; j++) 
        {
            int start = j * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            for (i = start; i < end; i++) 
            {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);

            }

        }
    }
    else {
        m_reset0(m);
        for (i = start_row; i < end_row; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  
    // 对各自的子矩阵执行LU分解
    for (k = 0; k < N; k++) {
        if (start_row <= k && k < end_row) {
            for (j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
            for (i = 0; i < numprocs; i++) {
                if (i != rank) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int src = k < N / numprocs * numprocs ? (k / rows_per_proc): (numprocs-1);
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(start_row, k + 1); i < end_row; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);	

    MPI_Finalize();
}
void LU_mpi_simd(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;

    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    // 分发子矩阵
    if (rank == 0) {
        for (j = 1; j < numprocs; j++) {
            int start = j * rows_per_proc;
            int end = (j == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            for (i = start; i < end; i++) {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        m_reset0(m);
        for (i = start_row; i < end_row; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 对各自的子矩阵执行LU分解
    for (k = 0; k < N; k++) {
        if (start_row <= k && k < end_row) {
            // 利用SIMD并行计算除法
            __m256 div = _mm256_set1_ps(m[k][k]);
            for (j = k + 1; j + SIMD_WIDTH < N; j += SIMD_WIDTH) {
                __m256 vec = _mm256_loadu_ps(&m[k][j]);
                vec = _mm256_div_ps(vec, div);
                _mm256_storeu_ps(&m[k][j], vec);
            }
            m[k][k] = 1.0;
            for (i = 0; i < numprocs; i++) {
                if (i != rank) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int src = k < N / numprocs * numprocs ? (k / rows_per_proc) : (numprocs - 1);
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }

        for (i = max(start_row, k + 1); i < end_row; i++) {
            __m256 vec_ik = _mm256_broadcast_ss(&m[i][k]);
            for (j = k + 1; j + SIMD_WIDTH < N; j += SIMD_WIDTH) {
                __m256 vec_kj = _mm256_loadu_ps(&m[k][j]);
                __m256 vec_ij = _mm256_loadu_ps(&m[i][j]);
                __m256 tmp = _mm256_mul_ps(vec_ik, vec_kj);
                vec_ij = _mm256_sub_ps(vec_ij, tmp);
                _mm256_storeu_ps(&m[i][j], vec_ij);
            }
            m[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

void LU_mpi_omp(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;
    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    if (rank == 0)
    {
        for (j = 1; j < numprocs; j++)
        {
            int start = j * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            for (i = start; i < end; i++)
            {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);

            }

        }
    }
    else {
        m_reset0(m);
        for (i = start_row; i < end_row; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // 对各自的子矩阵执行LU分解
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        if (start_row <= k && k < end_row) {
            for (j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
            for (i = 0; i < numprocs; i++) {
                if (i != rank) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            int src = k < N / numprocs * numprocs ? (k / rows_per_proc) : (numprocs - 1);
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
#pragma omp for schedule(guided)
        for (i = max(start_row, k + 1); i < end_row; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
void LU_mpi_optimize(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;
    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    if (rank == 0)
    {
        for (j = 1; j < numprocs; j++)
        {
            int start = j * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            for (i = start; i < end; i++)
            {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);

            }

        }
    }
    else {
        m_reset0(m);
        for (i = start_row; i < end_row; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // 对各自的子矩阵执行LU分解
    for (k = 0; k < N; k++) {
        if (start_row <= k && k < end_row) {
            for (j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
            for (i = rank+1; i < numprocs; i++) {

                MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            if (k == end_row - 1) {
                break;
            }
        }
        else {
            int src = k < N / numprocs * numprocs ? (k / rows_per_proc) : (numprocs - 1);
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(start_row, k + 1); i < end_row; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}

void LU_mpi_circular(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Status status;
    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    if (rank == 0)
    {
        for (j = 1; j < numprocs; j++)
        {
            for (i = j; i < N; i++)
            {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);

            }

        }
    }
    else {
        m_reset0(m);
        for (i = start_row; i < N; i+= numprocs) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // 对各自的子矩阵执行LU分解
    for (k = 0; k < N; k++) {
        if (k % numprocs == rank) {

            for (j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
            for (i = 0; i < numprocs; i++)
            {
                if (i != rank) {
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }

            }
        }
        else
        {
            int src = k % numprocs;
            MPI_Recv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int start = k;
        while (start % numprocs != rank) {
            start++;
        }
        for (i = start; i < N; i += numprocs) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
void LU_mpi_async(float** m, int argc, char* argv[])
{
    int rank = 0, numprocs = 0;
    int i = 0, j = 0, k = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    if (rank == 0)
    {
        MPI_Request* request = new MPI_Request[N - end_row];
        for (j = 1; j < numprocs; j++)
        {
            int start = j * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            for (i = start; i < end; i++)
            {
                MPI_Isend(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, & request[i - end_row]);

            }
        }
        MPI_Waitall(N - end_row, request, MPI_STATUS_IGNORE); //等待传递
    }
    else {
        m_reset0(m);
        MPI_Request* request = new MPI_Request[end_row - start_row];
        for (i = start_row; i < end_row; i++) {
            MPI_Irecv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - start_row]);
        }
        MPI_Waitall(end_row - start_row, request, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // 对各自的子矩阵执行LU分解
    for (k = 0; k < N; k++) {
        if (start_row <= k && k < end_row) {
            for (j = k + 1; j < N; j++) {
                m[k][j] /= m[k][k];
            }
            m[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[numprocs - 1 - rank];  //非阻塞传递
            for (i = rank + 1; i < numprocs; i++) {
                MPI_Isend(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request[i - rank - 1]);
            }
            MPI_Waitall(numprocs - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end_row - 1) {
                break;
            }
        }
        else {
            int src = k / rows_per_proc;
            MPI_Request request;
            MPI_Irecv(&m[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        for (i = max(start_row, k + 1); i < end_row; i++) {
            for (j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
void LU_mpi_simd_circular(float** m, int argc, char* argv[]) {
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    // 分发子矩阵
    if (rank == 0) {
        for (int j = 1; j < numprocs; j++) {
            int start = j * rows_per_proc;
            int end = (j == numprocs - 1) ? N : start + rows_per_proc;
            for (int i = start; i < end; i++) {
                MPI_Send(&m[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        m_reset0(m);
        for (int i = start_row; i < end_row; i++) {
            MPI_Recv(&m[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 对各自的子矩阵执行LU分解
    for (int k = 0; k < N; k++) {
        int proc_owner = k % numprocs;
        if (proc_owner == rank) {
            // 利用SIMD并行计算除法
            __m256 div = _mm256_set1_ps(m[k][k]);
            for (int j = k + 1; j + SIMD_WIDTH < N; j += SIMD_WIDTH) {
                __m256 vec = _mm256_loadu_ps(&m[k][j]);
                vec = _mm256_div_ps(vec, div);
                _mm256_storeu_ps(&m[k][j], vec);
            }
            m[k][k] = 1.0;

            // 非阻塞发送
            MPI_Request* requests = new MPI_Request[numprocs - 1];
            int req_count = 0;
            for (int i = 0; i < numprocs; i++) {
                if (i != rank) {
                    MPI_Isend(&m[k][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[req_count++]);
                }
            }
            MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
            delete[] requests;
        }
        else {
            // 非阻塞接收
            MPI_Request req;
            MPI_Irecv(&m[k][0], N, MPI_FLOAT, proc_owner, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }

        int start = (k + 1) % numprocs == rank ? k + 1 : start_row;
        for (int i = start; i < end_row; i += numprocs) {
            __m256 vec_ik = _mm256_broadcast_ss(&m[i][k]);
            for (int j = k + 1; j + SIMD_WIDTH < N; j += SIMD_WIDTH) {
                __m256 vec_kj = _mm256_loadu_ps(&m[k][j]);
                __m256 vec_ij = _mm256_loadu_ps(&m[i][j]);
                __m256 tmp = _mm256_mul_ps(vec_ik, vec_kj);
                vec_ij = _mm256_sub_ps(vec_ij, tmp);
                _mm256_storeu_ps(&m[i][j], vec_ij);
            }
            m[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
int main(int argc, char* argv[])
{
    //cout << "串行" << endl;
    //while (N <= 3000)
    //{
    //    m = new float* [N];
    //    m_reset(m);
    //    struct timespec sts, ets;
    //    timespec_get(&sts, TIME_UTC);
    //    LU_mpi(m,argc,argv);


    //    timespec_get(&ets, TIME_UTC);
    //    time_t dsec = ets.tv_sec - sts.tv_sec;
    //    long dnsec = ets.tv_nsec - sts.tv_nsec;
    //    if (dnsec < 0)
    //    {
    //        dsec--;
    //        dnsec += 1000000000ll;
    //    }
    //    printf("%lld.%09lld\n", dsec, dnsec);
    //    // 清理内存
    //    for (int i = 0; i < N; i++) {
    //        delete[] m[i];
    //    }
    //    delete[] m;
    //    N += 100;
    //}

    m = new float* [N];
    m_reset(m);
    struct timespec sts, ets;
    timespec_get(&sts, TIME_UTC);
    LU_mpi_circular(m,argc,argv);


    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec - sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0)
    {
        dsec--;
        dnsec += 1000000000ll;
    }
    printf("%lld.%09lld\n", dsec, dnsec);

    
    // 清理内存
    for (int i = 0; i < N; i++) {
        delete[] m[i];
    }
    delete[] m;
    

    //m = new float* [N];
    //m_reset(m);




    return 0;

}