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
const int NX_DEF = 1000000;

double u_exact(double x) { return x; }
double k(double x) { return 1.0 + exp(5.0 * x); }
double kp(double x) { return 5.0 * exp(5.0 * x); }
double q(double x) { return 1.0 - 0.5 * sin(8.0 * M_PI * x); }

double f_rhs(double x) {
    double u = u_exact(x), up = 1.0, u2 = 0.0;
    return q(x)*u - k(x)*u2 - kp(x)*up;
}

void local_sweep(int n, double *a, double *b, double *c, double *f, double *y, double left_val) {
    double *alpha = malloc((n+1) * sizeof(double));
    double *beta  = malloc((n+1) * sizeof(double));
    alpha[0] = 0.0;
    beta[0]  = left_val;

    for (int i = 0; i < n; i++) {
        double denom = c[i] - a[i] * alpha[i];
        if (fabs(denom) < 1e-14) {
            printf("Zero denom at i=%d\n", i);
            exit(1);
        }
        alpha[i+1] = b[i] / denom;
        beta[i+1]  = (f[i] - a[i] * beta[i]) / denom;
    }

    y[n-1] = beta[n];
    for (int i = n-2; i >= 0; i--) {
        y[i] = alpha[i+1] * y[i+1] + beta[i+1];
    }

    free(alpha); free(beta);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, mp;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    int nx = (argc > 1) ? atoi(argv[1]) : NX_DEF;
    if (nx < np) nx = np * 100;

    double hx = (B - A) / nx;
    double hx2 = hx * hx;

    int nc = nx / np;
    if (mp == np - 1) nc += nx % np;

    double *x = malloc(nc * sizeof(double));
    double *aa = calloc(nc, sizeof(double));
    double *bb = calloc(nc, sizeof(double));
    double *cc = calloc(nc, sizeof(double));
    double *ff = calloc(nc, sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < nc; i++) {
        int gi = mp * (nx / np) + i;
        x[i] = A + gi * hx;
    }

    /* Левая граница */
    if (mp == 0) {
        aa[0]=0.0; bb[0]=0.0; cc[0]=1.0; ff[0]=U_A;
    } else {
        double s0 = k(x[0]);
        double s1 = k(x[0]-hx);
        double s2 = k(x[0]+hx);
        aa[0] = 0.5*(s0+s1);
        bb[0] = 0.5*(s0+s2);
        cc[0] = hx2*q(x[0]) + aa[0] + bb[0];
        ff[0] = hx2*f_rhs(x[0]);
    }

    for (int i = 1; i < nc-1; i++) {
        double s0 = k(x[i]);
        double s1 = k(x[i-1]);
        double s2 = k(x[i+1]);
        aa[i] = 0.5*(s0+s1);
        bb[i] = 0.5*(s0+s2);
        cc[i] = hx2*q(x[i]) + aa[i] + bb[i];
        ff[i] = hx2*f_rhs(x[i]);
    }

    /* Правая граница */
    if (mp == np-1) {
        int i = nc-1;
        double s0 = k(x[i]);
        double s1 = k(x[i-1]);
        aa[i] = 0.5*(s0+s1);
        bb[i] = 0.0;
        cc[i] = 0.5*hx2*q(x[i]) + aa[i];
        ff[i] = 0.5*hx2*f_rhs(x[i]) + hx*U_B*s0;
    } else {
        int i = nc-1;
        double s0 = k(x[i]);
        double s1 = k(x[i]-hx);
        double s2 = k(x[i]+hx);
        aa[i] = 0.5*(s0+s1);
        bb[i] = 0.5*(s0+s2);
        cc[i] = hx2*q(x[i]) + aa[i] + bb[i];
        ff[i] = hx2*f_rhs(x[i]);
    }

    double t_start = MPI_Wtime();
    double *y1 = malloc(nc * sizeof(double));
    local_sweep(nc, aa, bb, cc, ff, y1, U_A);

    double *y = malloc(nc * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < nc; i++) y[i] = y1[i];

    double dmax = 0.0;
    #pragma omp parallel for reduction(max:dmax)
    for (int i = 0; i < nc; i++) {
        double err = fabs(u_exact(x[i]) - y[i]);
        if (err > dmax) dmax = err;
    }

    double dmax_glob = 0.0;
    MPI_Allreduce(&dmax, &dmax_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    if (mp == 0) {
        printf("nx=%d np=%d t=%.4lf dmax=%.4e\n",
               nx, np, t_end - t_start, dmax_glob);
    }

    free(x); free(aa); free(bb); free(cc); free(ff);
    free(y1); free(y);
    MPI_Finalize();
    return 0;
}