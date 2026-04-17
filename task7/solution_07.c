#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double A = 0.0, B = 1.0;
const double U_A = 0.0, U_B = 1.0;

double u_exact(double x) { return x; }
double k(double x) { return 1.0 + exp(5.0 * x); }
double kp(double x) { return 5.0 * exp(5.0 * x); }
double q(double x) { return 1.0 - 0.5 * sin(8.0 * M_PI * x); }
double f_rhs(double x) {
    double u = u_exact(x), up = 1.0, u2 = 0.0;
    return q(x)*u - k(x)*u2 - kp(x)*up;
}

/* Последовательная правая прогонка */
void local_sweep(int n, double *a, double *b, double *c, double *f, double *y) {
    if (n <= 0) return;
    double *alpha = malloc((n+1)*sizeof(double));
    double *beta  = malloc((n+1)*sizeof(double));
    alpha[0] = 0.0; beta[0] = 0.0;
    for(int i=0; i<n; i++) {
        double d = c[i] - a[i]*alpha[i];
        alpha[i+1] = b[i]/d;
        beta[i+1]  = (f[i] - a[i]*beta[i])/d;
    }
    y[n-1] = beta[n];
    for(int i=n-2; i>=0; i--) y[i] = alpha[i+1]*y[i+1] + beta[i+1];
    free(alpha); free(beta);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, mp;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    int nx = (argc > 1) ? atoi(argv[1]) : 100000;
    if(nx < np) nx = np * 100;

    double hx = (B - A) / nx;
    int nc = nx / np;
    if(mp == np - 1) nc += nx % np;

    /* Локальная сетка */
    double *x = malloc(nc * sizeof(double));
    #pragma omp parallel for
    for(int i=0; i<nc; i++) x[i] = A + (mp*nx/np + i)*hx;

    /* Коэффициенты СЛАУ */
    double *aa = calloc(nc, sizeof(double));
    double *bb = calloc(nc, sizeof(double));
    double *cc = calloc(nc, sizeof(double));
    double *ff = calloc(nc, sizeof(double));

    #pragma omp parallel for
    for(int i=0; i<nc; i++) {
        double xi = x[i];
        double km = 0.5*(k(xi) + (i>0 ? k(xi-hx) : k(xi)));
        double kp2 = 0.5*(k(xi) + (i<nc-1 ? k(xi+hx) : k(xi)));
        aa[i] = km / hx;
        bb[i] = kp2 / hx;
        cc[i] = aa[i] + bb[i] + hx*hx * q(xi);
        ff[i] = hx*hx * f_rhs(xi);
    }

    /* Граничные условия */
    if(mp == 0) { cc[0] = 1.0; bb[0] = 0.0; ff[0] = U_A; }
    if(mp == np - 1) {
        cc[nc-1] -= bb[nc-1];
        ff[nc-1] += hx * k(B) * U_B;
        bb[nc-1] = 0.0;
    }

    double t_start = MPI_Wtime();
    double *y1 = malloc(nc*sizeof(double));
    double *y2 = NULL, *y3 = NULL, *y = NULL;

    /* 1. Базис y1 */
    local_sweep(nc, aa, bb, cc, ff, y1);

    if (np > 1) {
        y2 = malloc(nc*sizeof(double));
        y3 = malloc(nc*sizeof(double));

        /* Сохраняем граничные коэффициенты */
        double a0=aa[0], b0=bb[0], c0=cc[0], f0=ff[0];
        double a1=aa[nc-1], b1=bb[nc-1], c1=cc[nc-1], f1=ff[nc-1];

        /* 2. Базис y2 (левая граница = 1) */
        if(mp == 0) { cc[0]=1.0; bb[0]=0.0; ff[0]=1.0; }
        else { ff[0]=0.0; aa[0]=0.0; cc[0]=1.0; }
        if(mp == np-1) { cc[nc-1]-=bb[nc-1]; ff[nc-1]=0.0; bb[nc-1]=0.0; }
        local_sweep(nc, aa, bb, cc, ff, y2);

        /* 3. Базис y3 (правая граница = 1) */
        aa[0]=a0; bb[0]=b0; cc[0]=c0; ff[0]=f0;
        aa[nc-1]=a1; bb[nc-1]=b1; cc[nc-1]=c1; ff[nc-1]=f1;
        if(mp == 0) { cc[0]=1.0; bb[0]=0.0; ff[0]=0.0; }
        else { ff[0]=0.0; aa[0]=0.0; cc[0]=1.0; }
        if(mp == np-1) { cc[nc-1]=1.0; aa[nc-1]=0.0; bb[nc-1]=0.0; ff[nc-1]=1.0; }
        local_sweep(nc, aa, bb, cc, ff, y3);

        /* Восстановление исходных коэффициентов */
        aa[0]=a0; bb[0]=b0; cc[0]=c0; ff[0]=f0;
        aa[nc-1]=a1; bb[nc-1]=b1; cc[nc-1]=c1; ff[nc-1]=f1;
        if(mp == 0) { cc[0]=1.0; bb[0]=0.0; ff[0]=U_A; }
        if(mp == np-1) { cc[nc-1]-=bb[nc-1]; ff[nc-1]+=hx*k(B)*U_B; bb[nc-1]=0.0; }

        /* Формирование короткой системы */
        int ncp = 2 * (np - 1);
        int size_dd = 8 * (np - 1);
        double *dd = calloc(size_dd, sizeof(double));
        double *ee = calloc(size_dd, sizeof(double));

        if(mp == 0) {
            c1 -= a1 * y2[nc-2]; f1 += a1 * y1[nc-2]; a1 = 0.0;
            dd[0]=a1; dd[1]=b1; dd[2]=c1; dd[3]=f1;
        } else if(mp == np - 1) {
            int i = mp * 8 - 4;
            c0 -= b0 * y3[1]; f0 += b0 * y1[1]; b0 = 0.0;
            dd[i]=a0; dd[i+1]=b0; dd[i+2]=c0; dd[i+3]=f0;
        } else {
            int i = mp * 8 - 4;
            c0 -= b0 * y3[1]; f0 += b0 * y1[1]; b0 *= y2[1];
            c1 -= a1 * y2[nc-2]; f1 += a1 * y1[nc-2]; a1 *= y3[nc-2];
            dd[i]=a0; dd[i+1]=b0; dd[i+2]=c0; dd[i+3]=f0;
            dd[i+4]=a1; dd[i+5]=b1; dd[i+6]=c1; dd[i+7]=f1;
        }

        double t_comm_start = MPI_Wtime();
        MPI_Allreduce(dd, ee, size_dd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t_comm = MPI_Wtime() - t_comm_start;

        /* Решение короткой системы */
        double *a_s = malloc(ncp*sizeof(double));
        double *b_s = malloc(ncp*sizeof(double));
        double *c_s = malloc(ncp*sizeof(double));
        double *f_s = malloc(ncp*sizeof(double));
        double *y_s = malloc(ncp*sizeof(double));

        for(int i=0; i<ncp; i++) {
            a_s[i] = ee[4*i];
            b_s[i] = ee[4*i+1];
            c_s[i] = ee[4*i+2];
            f_s[i] = ee[4*i+3];
        }
        a_s[0] = 0.0; b_s[ncp-1] = 0.0;
        local_sweep(ncp, a_s, b_s, c_s, f_s, y_s);

        /* Восстановление решения */
        y = malloc(nc*sizeof(double));
        if(mp == 0) {
            double c1_val = y_s[0];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + c1_val * y2[i];
        } else if(mp == np-1) {
            double a1_val = y_s[ncp-1];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + a1_val * y3[i];
        } else {
            double a1_val = y_s[2*mp-2];
            double c1_val = y_s[2*mp-1];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + a1_val*y3[i] + c1_val*y2[i];
        }

        free(a_s); free(b_s); free(c_s); free(f_s); free(y_s);
        free(dd); free(ee);
    } else {
        y = malloc(nc*sizeof(double));
        #pragma omp parallel for
        for(int i=0; i<nc; i++) y[i] = y1[i];
    }

    /* Оценка погрешности */
    double dmax = 0.0;
    #pragma omp parallel for reduction(max:dmax)
    for(int i=0; i<nc; i++) {
        double err = fabs(u_exact(x[i]) - y[i]);
        if(err > dmax) dmax = err;
    }
    double dmax_glob = 0.0;
    MPI_Allreduce(&dmax, &dmax_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    if(mp == 0) {
        printf("nx=%d np=%d t_total=%.4lf dmax=%.4e\n",
               nx, np, t_end - t_start, dmax_glob);
    }

    free(x); free(aa); free(bb); free(cc); free(ff);
    free(y1); if(y2) free(y2); if(y3) free(y3); free(y);
    MPI_Finalize();
    return 0;
}