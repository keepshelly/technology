#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cblas.h>
using namespace std;
using namespace std::chrono;
template <typename T>
void my_symm(int M, int N, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T sum = 0;
            for (int k = 0; k < M; ++k) {
                int row = (i >= k) ? i : k;
                int col = (i >= k) ? k : i;
                sum += A[row + col * lda] * B[k + j * ldb];
            }
            if (beta == 0) {
                C[i + j * ldc] = alpha * sum;
            } else {
                C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
            }
        }
    }
}
double geometric_mean(const vector<double>& data) {
    double log_sum = 0;
    for (double x : data) log_sum += log(x);
    return exp(log_sum / data.size());
}
template <typename T>
void run_test(int M, int N) {
    string type_name = (sizeof(T) == 4 ? "Float (Single Precision)" : "Double (Double Precision)");
    cout << "\n========================================" << endl;
    cout << "ТЕСТ ДЛЯ ТИПА: " << type_name << endl;
    cout << "Размер матриц: " << M << "x" << N << endl;
    cout << "========================================" << endl;
    vector<T> A(M * M, 1.1f), B(M * N, 2.2f), C_my(M * N, 0.0f), C_blas(M * N, 0.0f);
    T alpha = 1.0, beta = 0.0;
    vector<double> times_my, times_blas;
    for (int iter = 0; iter < 10; ++iter) {
        cout << "Итерация " << iter + 1 << "/10... " << flush;
        auto start = high_resolution_clock::now();
        my_symm(M, N, alpha, A.data(), M, B.data(), M, beta, C_my.data(), M);
        auto end = high_resolution_clock::now();
        double d_my = duration<double>(end - start).count();
        times_my.push_back(d_my);
        start = high_resolution_clock::now();
        if constexpr (sizeof(T) == 4)
            cblas_ssymm(CblasColMajor, CblasLeft, CblasLower, M, N, alpha, A.data(), M, B.data(), M, beta, C_blas.data(), M);
        else
            cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, M, N, alpha, A.data(), M, B.data(), M, beta, C_blas.data(), M);
        end = high_resolution_clock::now();
        double d_blas = duration<double>(end - start).count();
        times_blas.push_back(d_blas);

        cout << "OK (My: " << fixed << setprecision(2) << d_my << "s, BLAS: " << d_blas << "s)" << endl;
    }
    cout << "\nРезультаты для " << type_name << ":" << endl;
    cout << "| № | My Time (s) | BLAS Time (s) | Perf (%) |" << endl;
    cout << "|---|-------------|---------------|----------|" << endl;
    for (int i = 0; i < 10; ++i) {
        double perf = (times_blas[i] / times_my[i]) * 100.0;
        cout << "| " << setw(1) << i + 1 
             << " | " << setw(11) << fixed << setprecision(4) << times_my[i] 
             << " | " << setw(13) << times_blas[i] 
             << " | " << setw(7) << perf << "% |" << endl;
    }
    cout << "\nСреднее геометрическое(My): " << geometric_mean(times_my) << " s" << endl;
    cout << "Среднее геометрическое(BLAS): " << geometric_mean(times_blas) << " s" << endl;
}
int main() {
    int size = 3000; 
    run_test<float>(size, size);
    run_test<double>(size, size);

    return 0;
}