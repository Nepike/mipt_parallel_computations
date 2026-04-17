#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* === Тестовая функция и коэффициенты (Задание 7а) === */
double u_exact(double x) { return x; }
double k(double x) { return 1.0 + exp(5.0 * x); }
double kp(double x) { return 5.0 * exp(5.0 * x); }
double q(double x) { return 1.0 - 0.5 * sin(8.0 * M_PI * x); }
/* f(x) = q(x)*u - k(x)*u'' - k'(x)*u'. Для u(x)=x => u'=1, u''=0 */
double f_rhs(double x) { return q(x) * x - kp(x); }

/* Последовательная правая прогонка для трёхдиагональной СЛАУ */
int prog_right(int n, double *a, double *b, double *c, double *f, double *y) {
    double *alpha = (double*)malloc((n + 1) * sizeof(double));
    double *beta  = (double*)malloc((n + 1) * sizeof(double));
    if (!alpha || !beta) return -1;

    alpha[0] = 0.0; 
    beta[0]  = f[0] / c[0];
    for (int i = 0; i < n - 1; i++) {
        double denom = c[i] - a[i] * alpha[i];
        if (fabs(denom) < 1e-15) { free(alpha); free(beta); return -2; }
        alpha[i+1] = b[i] / denom;
        beta[i+1]  = (f[i] - a[i] * beta[i]) / denom;
    }
    y[n-1] = (f[n-1] - a[n-1] * beta[n-1]) / (c[n-1] - a[n-1] * alpha[n-1]);
    for (int i = n - 2; i >= 0; i--) y[i] = alpha[i+1] * y[i+1] + beta[i+1];
    
    free(alpha); free(beta);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int np, mp;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    int nx = (argc > 1) ? atoi(argv[1]) : 100000;
    if (nx < np * 2) nx = np * 2; /* Защита от слишком мелких подобластей */

    double xa = 0.0, xb = 1.0;
    double ua = 0.0, ub = 1.0; /* u(0)=0, u'(1)=1 */

    double hx = (xb - xa) / nx;
    double hx2 = hx * hx;

    /* Декомпозиция области */
    int i1 = mp * (nx / np);
    int i2 = (mp == np - 1) ? nx : i1 + (nx / np);
    int nc = i2 - i1;
    int ncm = nc - 1;
    int ncp = 2 * (np - 1);

    /* Выделение памяти */
    double *xx = (double*)malloc(nc * sizeof(double));
    double *aa = (double*)calloc(nc, sizeof(double));
    double *bb = (double*)calloc(nc, sizeof(double));
    double *cc = (double*)calloc(nc, sizeof(double));
    double *ff = (double*)calloc(nc, sizeof(double));
    double *y1 = (double*)malloc(nc * sizeof(double));

    /* Построение сетки */
    #pragma omp parallel for
    for (int i = 0; i < nc; i++) xx[i] = xa + hx * (i1 + i);

    /* Заполнение коэффициентов разностной схемы */
    #pragma omp parallel for
    for (int i = 0; i < nc; i++) {
        int idx = i1 + i;
        double xi = xx[i];
        double km = 0.5 * (k(xi) + (idx > 0 ? k(xi - hx) : k(xa)));
        double kp2 = 0.5 * (k(xi) + (idx < nx ? k(xi + hx) : k(xb)));
        aa[i] = km / hx;
        bb[i] = kp2 / hx;
        cc[i] = aa[i] + bb[i] + hx2 * q(xi);
        ff[i] = hx2 * f_rhs(xi);
    }

    /* Граничные условия */
    if (mp == 0) { /* Дирихле слева */
        aa[0] = 0.0; bb[0] = 0.0; cc[0] = 1.0; ff[0] = ua;
    }
    if (mp == np - 1) { /* Нейман справа */
        double kb = k(xb);
        aa[ncm] = 0.5 * (k(xx[ncm]) + k(xx[ncm] - hx));
        bb[ncm] = 0.0;
        cc[ncm] = 0.5 * hx2 * q(xx[ncm]) + aa[ncm];
        ff[ncm] = 0.5 * hx2 * f_rhs(xx[ncm]) + hx * ub * kb;
    }

    double t1 = MPI_Wtime();
    double t2 = 0.0;

    if (np == 1) {
        prog_right(nc, aa, bb, cc, ff, y1);
    } else {
        /* Массивы для базисных функций и короткой системы */
        double *y2 = (double*)malloc(nc * sizeof(double));
        double *y3 = (double*)malloc(nc * sizeof(double));
        double *y4 = (double*)malloc(ncp * sizeof(double));
        double *dd = (double*)calloc(4 * ncp, sizeof(double));
        double *ee = (double*)calloc(4 * ncp, sizeof(double));

        double a0 = aa[0], b0 = bb[0], c0 = cc[0], f0 = ff[0];
        double a1 = aa[ncm], b1 = bb[ncm], c1 = cc[ncm], f1 = ff[ncm];

        /* Решение трёх локальных задач для базисных функций */
        if (mp == 0) {
            cc[ncm]=1.0; bb[ncm]=0.0; aa[ncm]=0.0; ff[ncm]=0.0;
            prog_right(nc, aa, bb, cc, ff, y1);
            for (int i = 0; i < ncm; i++) ff[i] = 0.0; ff[ncm] = 1.0;
            prog_right(nc, aa, bb, cc, ff, y2);
        } else if (mp < np - 1) {
            aa[0]=0.0; bb[0]=0.0; cc[0]=1.0; ff[0]=0.0;
            aa[ncm]=0.0; bb[ncm]=0.0; cc[ncm]=1.0; ff[ncm]=0.0;
            prog_right(nc, aa, bb, cc, ff, y1);
            for (int i = 0; i < ncm; i++) ff[i] = 0.0; ff[ncm] = 1.0;
            prog_right(nc, aa, bb, cc, ff, y2);
            ff[0] = 1.0; for (int i = 1; i <= ncm; i++) ff[i] = 0.0;
            prog_right(nc, aa, bb, cc, ff, y3);
        } else {
            aa[0]=0.0; bb[0]=0.0; cc[0]=1.0; ff[0]=0.0;
            prog_right(nc, aa, bb, cc, ff, y1);
            ff[0] = 1.0; for (int i = 1; i <= ncm; i++) ff[i] = 0.0;
            prog_right(nc, aa, bb, cc, ff, y3);
        }

        /* Формирование коэффициентов короткой системы */
        if (mp == 0) {
            c1 = c1 - a1 * y2[ncm - 1]; f1 = f1 + a1 * y1[ncm - 1]; a1 = 0.0;
            dd[0]=a1; dd[1]=b1; dd[2]=c1; dd[3]=f1;
        } else if (mp < np - 1) {
            c0 = c0 - b0 * y3[1]; f0 = f0 + b0 * y1[1]; b0 = b0 * y2[1];
            c1 = c1 - a1 * y2[ncm - 1]; f1 = f1 + a1 * y1[ncm - 1]; a1 = a1 * y3[ncm - 1];
            int idx = mp * 8 - 4;
            dd[idx]=a0; dd[idx+1]=b0; dd[idx+2]=c0; dd[idx+3]=f0;
            dd[idx+4]=a1; dd[idx+5]=b1; dd[idx+6]=c1; dd[idx+7]=f1;
        } else {
            c0 = c0 - b0 * y3[1]; f0 = f0 + b0 * y1[1]; b0 = 0.0;
            int idx = mp * 8 - 4;
            dd[idx]=a0; dd[idx+1]=b0; dd[idx+2]=c0; dd[idx+3]=f0;
        }

        t2 = MPI_Wtime();
        MPI_Allreduce(dd, ee, 4 * ncp, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t2 = MPI_Wtime() - t2;

        for (int i = 0; i < ncp; i++) {
            aa[i] = ee[4*i]; bb[i] = ee[4*i+1]; cc[i] = ee[4*i+2]; ff[i] = ee[4*i+3];
        }
        prog_right(ncp, aa, bb, cc, ff, y4);

        /* Восстановление полного решения (OpenMP) */
        #pragma omp parallel for
        if (mp == 0) {
            double c1_val = y4[0];
            for (int i = 0; i < nc; i++) y1[i] = y1[i] + c1_val * y2[i];
        } else if (mp < np - 1) {
            double a1_val = y4[2*mp-1];
            double b1_val = y4[2*mp];
            for (int i = 0; i < nc; i++) y1[i] = y1[i] + a1_val * y3[i] + b1_val * y2[i];
        } else {
            double a1_val = y4[2*mp-1];
            for (int i = 0; i < nc; i++) y1[i] = y1[i] + a1_val * y3[i];
        }

        free(y2); free(y3); free(y4); free(dd); free(ee);
    }
    t1 = MPI_Wtime() - t1;

    /* Оценка максимальной погрешности (OpenMP + MPI) */
    double dmax_local = 0.0;
    #pragma omp parallel for reduction(max:dmax_local)
    for (int i = 0; i < nc; i++) {
        double err = fabs(u_exact(xx[i]) - y1[i]);
        if (err > dmax_local) dmax_local = err;
    }
    double dmax_global = 0.0;
    MPI_Allreduce(&dmax_local, &dmax_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mp == 0) {
        printf("nx=%d np=%d t_comp=%.6lf t_comm=%.6lf dmax=%.4le\n", 
               nx, np, t1, t2, dmax_global);
    }

    free(xx); free(aa); free(bb); free(cc); free(ff); free(y1);
    MPI_Finalize();
    return 0;
}