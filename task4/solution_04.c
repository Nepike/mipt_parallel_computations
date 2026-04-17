#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define TAG 777

/* Интегрируемая функция */
double f1(double x) { return 4.0 / (1.0 + x * x); }

/* Метод средних прямоугольников с OpenMP-редукцией */
double integrate(double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += f1(a + h * (i + 0.5));
    }
    return h * s;
}

int main(int argc, char *argv[]) {
    int np, mp, nt = 1;
    if (argc > 1) nt = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);
    omp_set_num_threads(nt);

    const int TOTAL_N = 1000000000;
    double a = 0.0, b = 1.0;
    double h_proc = (b - a) / np;
    double a1 = h_proc * mp;
    double b1 = (mp == np - 1) ? b : a1 + h_proc;
    int n1 = TOTAL_N / np;

    /* Замер времени вычислений */
    double t_start = MPI_Wtime();
    double my_sum = integrate(a1, b1, n1);
    double t_comp = MPI_Wtime() - t_start;

    /* Замер времени редукции методом сдваивания */
    t_start = MPI_Wtime();
    int step = 1;
    double recv_val;
    MPI_Status status;

    while (step < np) {
        if (mp % (2 * step) == 0) {
            if (mp + step < np) {
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, mp + step, TAG, MPI_COMM_WORLD, &status);
                my_sum += recv_val;
            }
        } else {
            if (mp - step >= 0) {
                MPI_Send(&my_sum, 1, MPI_DOUBLE, mp - step, TAG, MPI_COMM_WORLD);
                break;
            }
        }
        step *= 2;
    }
    double t_comm = MPI_Wtime() - t_start;
    double t_total = t_comp + t_comm;

    fprintf(stderr, "mp=%d t1=%10lf t2=%10lf t3=%10lf int=%22.15le\n",
            mp, t_comp, t_comm, t_total, my_sum);

    MPI_Finalize();
    return 0;
}