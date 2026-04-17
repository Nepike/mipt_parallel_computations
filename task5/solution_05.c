#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* Подынтегральная функция f(x,y,z) = x^2 + y^2 + z^2 */
double f(double x, double y, double z) {
    return x * x + y * y + z * z;
}

/* Автоматическое определение сбалансированной 3D-решетки процессоров */
void get_grid_3d(int np, int *p1, int *p2, int *p3) {
    *p1 = *p2 = *p3 = 1;
    int temp = np;
    for (int i = 2; i * i <= temp; i++) {
        while (temp % i == 0) {
            if (*p1 <= *p2 && *p1 <= *p3) (*p1) *= i;
            else if (*p2 <= *p1 && *p2 <= *p3) (*p2) *= i;
            else (*p3) *= i;
            temp /= i;
        }
    }
    if (temp > 1) {
        if (*p1 <= *p2 && *p1 <= *p3) (*p1) *= temp;
        else if (*p2 <= *p1 && *p2 <= *p3) (*p2) *= temp;
        else (*p3) *= temp;
    }
}

int main(int argc, char *argv[]) {
    int np, mp;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    /* Параметры интегрирования: куб [0,1]x[0,1]x[0,1], сетка 1000^3 */
    double a = 0.0, b = 1.0, c = 0.0, d = 1.0, e = 0.0, g = 1.0;
    int Nx = 1000, Ny = 1000, Nz = 1000;

    /* Определяем размеры решетки процессоров */
    int p1, p2, p3;
    get_grid_3d(np, &p1, &p2, &p3);

    /* Логические координаты текущего процесса в 3D-решетке */
    int m1 = mp / (p2 * p3);
    int m2 = (mp / p3) % p2;
    int m3 = mp % p3;

    /* Границы подсетки для текущего процесса */
    int i_start = m1 * (Nx / p1);
    int i_end   = (m1 == p1 - 1) ? Nx : i_start + Nx / p1;
    int j_start = m2 * (Ny / p2);
    int j_end   = (m2 == p2 - 1) ? Ny : j_start + Ny / p2;
    int k_start = m3 * (Nz / p3);
    int k_end   = (m3 == p3 - 1) ? Nz : k_start + Nz / p3;

    double hx = (b - a) / Nx;
    double hy = (d - c) / Ny;
    double hz = (g - e) / Nz;
    double hvol = hx * hy * hz;

    /* Локальное суммирование */
    double local_sum = 0.0;
    double t1 = MPI_Wtime();
    for (int k = k_start; k < k_end; k++) {
        double z = e + (k + 0.5) * hz;
        for (int j = j_start; j < j_end; j++) {
            double y = c + (j + 0.5) * hy;
            for (int i = i_start; i < i_end; i++) {
                double x = a + (i + 0.5) * hx;
                local_sum += f(x, y, z);
            }
        }
    }
    local_sum *= hvol;
    t1 = MPI_Wtime() - t1;

    /* Коллективное суммирование на процессе 0 */
    double total_sum = 0.0;
    double t2 = MPI_Wtime();
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    t2 = MPI_Wtime() - t2;

    fprintf(stderr, "mp=%d t1=%10lf t2=%10lf t3=%10lf int=%22.15le\n",
            mp, t1, t2, t1 + t2, (mp == 0) ? total_sum : local_sum);

    MPI_Finalize();
    return 0;
}