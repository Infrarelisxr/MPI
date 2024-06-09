#include <iostream>
#include <Windows.h>
#include <sys/utime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <mpi.h>
using namespace std;
#define TIME_UTC 1
int N = 300;
float **m;
void m_reset(float **m)
{
    for (int i = 0; i < N; i++)
    {
        m[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            m[i][j] = rand();
        }
        for (int k = 0; k < N; k++)
        {
            for (int i = k + 1; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    m[i][j] += m[k][j];
                }
            }
        }
    }
}
void LU(float **m)
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
void LU_mpi(float **m, int argc, char* argv[])
{
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 计算每个进程负责的行数
    int rows_per_proc = N / numprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == numprocs - 1) ? N : start_row + rows_per_proc;

    // 将矩阵划分并发送给各个进程
    float** sub_m = new float*[rows_per_proc];
    if (rank == 0) {
        for (int i = 0; i < numprocs; i++) {
            int start = i * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            float** sub_data = new float*[rows];
            for (int j = 0; j < rows; j++) {
                sub_data[j] = new float[N];
                memcpy(sub_data[j], m[start + j], N * sizeof(float));
            }
            if (i == 0) {
                sub_m = sub_data;
            } else {
                MPI_Send(sub_data[0], rows * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                for (int j = 0; j < rows; j++) {
                    delete[] sub_data[j];
                }
                delete[] sub_data;
            }
        }
    } else {
        float* buffer = new float[rows_per_proc * N];
        MPI_Recv(buffer, rows_per_proc * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < rows_per_proc; i++) {
            sub_m[i] = new float[N];
            memcpy(sub_m[i], buffer + i * N, N * sizeof(float));
        }
        delete[] buffer;
    }

    // 对各自的子矩阵执行LU分解
    for (int k = 0; k < N; k++) {
        if (start_row <= k && k < end_row) {
            for (int j = k + 1; j < N; j++) {
                sub_m[k - start_row][j] /= sub_m[k - start_row][k];
            }
            sub_m[k - start_row][k] = 1.0;
            for (int i = 0; i < numprocs; i++) {
                if (i != rank) {
                    MPI_Send(sub_m[k - start_row], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
        } else {
            float* buffer = new float[N];
            int src = k / rows_per_proc;
            MPI_Recv(buffer, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = max(start_row, k + 1); i < end_row; i++) {
                for (int j = k + 1; j < N; j++) {
                    sub_m[i - start_row][j] -= buffer[j] * sub_m[i - start_row][k];
                }
                sub_m[i - start_row][k] = 0;
            }
            delete[] buffer;
        }
    }

    // 将结果收集到根进程
    if (rank == 0) {
        for (int i = 0; i < rows_per_proc; i++) {
            memcpy(m[start_row + i], sub_m[i], N * sizeof(float));
        }
        for (int i = 1; i < numprocs; i++) {
            int start = i * rows_per_proc;
            int end = (i == numprocs - 1) ? N : start + rows_per_proc;
            int rows = end - start;
            float* buffer = new float[rows * N];
            MPI_Recv(buffer, rows * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < rows; j++) {
                memcpy(m[start + j], buffer + j * N, N * sizeof(float));
            }
            delete[] buffer;
        }
    } else {
        MPI_Send(sub_m[0], rows_per_proc * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    // 清理内存
    for (int i = 0; i < rows_per_proc; i++) {
        delete[] sub_m[i];
    }
    delete[] sub_m;

    MPI_Finalize();
}
int main(int argc, char* argv[])
{

    m = new float *[N];
    m_reset(m);

    LU_mpi(m, argc, argv);

    // 输出结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }

    // 清理内存
    for (int i = 0; i < N; i++) {
        delete[] m[i];
    }
    delete[] m;

    return 0;

}
