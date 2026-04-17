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
const double U_A = 0.0, U_B = 1.0; /* u(0)=0, u'(1)=1 */
const int NX_DEF = 1000000;

double u_exact(double x) { return x; }
double k(double x) { return 1.0 + exp(5.0 * x); }
double kp(double x) { return 5.0 * exp(5.0 * x); }
double q(double x) { return 1.0 - 0.5 * sin(8.0 * M_PI * x); }
double f_rhs(double x) {
    double u = u_exact(x), up = 1.0, u2 = 0.0;
    return q(x)*u - k(x)*u2 - kp(x)*up;
}

/* Последовательная правая прогонка (рекуррентные зависимости запрещают OpenMP) */
void local_sweep(int n, double *a, double *b, double *c, double *f, double *y, double left_val) {
    double *alpha = malloc((n+1)*sizeof(double));
    double *beta  = malloc((n+1)*sizeof(double));
    alpha[0] = 0.0; beta[0] = left_val;
    for(int i=0; i<n; i++) {
        double d = c[i] - a[i]*alpha[i];
        alpha[i+1] = b[i]/d;
        beta[i+1]  = (f[i] - a[i]*beta[i])/d;
    }
    y[n-1] = beta[n-1];
    for(int i=n-2; i>=0; i--) y[i] = alpha[i+1]*y[i+1] + beta[i+1];
    free(alpha); free(beta);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, mp;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    int nx = (argc > 1) ? atoi(argv[1]) : NX_DEF;
    if(nx < np) nx = np * 100;

    double hx = (B - A) / nx;
    int nc = nx / np;
    if(mp == np - 1) nc += nx % np;

    double *x = malloc(nc * sizeof(double));
    #pragma omp parallel for
    for(int i=0; i<nc; i++) x[i] = A + (mp*nx/np + i)*hx;

    double *aa = calloc(nc, sizeof(double));
    double *bb = calloc(nc, sizeof(double));
    double *cc = calloc(nc, sizeof(double));
    double *ff = calloc(nc, sizeof(double));

    /* Сборка коэффициентов СЛАУ (параллелизуется по узлам) */
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
    if(mp == 0) { cc[0] = 1.0; aa[0] = 0.0; bb[0] = 0.0; ff[0] = U_A; }
    if(mp == np - 1) {
        cc[nc-1] -= bb[nc-1];
        ff[nc-1] += hx * k(B) * U_B; /* ИСПРАВЛЕНО: знак + соответствует u'(b)=U_B */
        bb[nc-1] = 0.0;
    }

    double t_start = MPI_Wtime();
    double *y1 = malloc(nc*sizeof(double));
    local_sweep(nc, aa, bb, cc, ff, y1, 0.0);

    double *y2 = NULL, *y3 = NULL, *y = NULL;

    if (np > 1) {
        y2 = malloc(nc*sizeof(double));
        y3 = malloc(nc*sizeof(double));

        /* Базис y2 (левая граница подрегиона) */
        double c0_save=cc[0], a0_save=aa[0], b0_save=bb[0], f0_save=ff[0];
        double cN_save=cc[nc-1], bN_save=bb[nc-1], fN_save=ff[nc-1];
        
        if(mp == 0) { cc[0]=1.0; aa[0]=0.0; bb[0]=0.0; ff[0]=1.0; }
        else { ff[0]=0.0; aa[0]=bb[0]=0.0; cc[0]=1.0; }
        if(mp == np-1) { cc[nc-1]=cN_save-bb[nc-1]; ff[nc-1]=0.0; bb[nc-1]=0.0; }
        local_sweep(nc, aa, bb, cc, ff, y2, 0.0);

        /* Базис y3 (правая граница подрегиона) */
        if(mp == 0) { cc[0]=1.0; aa[0]=0.0; bb[0]=0.0; ff[0]=0.0; }
        else { ff[0]=0.0; aa[0]=bb[0]=0.0; cc[0]=1.0; }
        if(mp == np-1) { cc[nc-1]=1.0; aa[nc-1]=0.0; bb[nc-1]=0.0; ff[nc-1]=1.0; }
        local_sweep(nc, aa, bb, cc, ff, y3, 0.0);

        /* Восстановление исходных коэффициентов для короткой системы */
        cc[0]=c0_save; aa[0]=a0_save; bb[0]=b0_save; ff[0]=f0_save;
        cc[nc-1]=cN_save; bb[nc-1]=bN_save; ff[nc-1]=fN_save;
        if(mp == np-1) { cc[nc-1]-=bb[nc-1]; ff[nc-1]+=hx*k(B)*U_B; bb[nc-1]=0.0; }

        /* Формирование и сборка короткой системы */
        int n_short = 2 * (np - 1);
        int n_coeffs = 4 * n_short;
        double *d_short = calloc(n_coeffs, sizeof(double));
        double *e_short = calloc(n_coeffs, sizeof(double));
        int base = (mp == 0) ? 0 : 4 + 8 * (mp - 1);

        if(mp == 0) {
            d_short[0] = aa[nc-1]*y2[nc-2] - cc[nc-1];
            d_short[1] = bb[nc-1];
            d_short[2] = aa[nc-1]*y1[nc-2] - ff[nc-1];
            d_short[3] = 0.0;
        } else if(mp == np - 1) {
            d_short[base]   = -cc[0];
            d_short[base+1] = bb[0]*y3[1];
            d_short[base+2] = -ff[0];
            d_short[base+3] = 0.0;
        } else {
            d_short[base]   = aa[0]*y2[1] - cc[0];
            d_short[base+1] = bb[0]*y3[1];
            d_short[base+2] = aa[0]*y1[1] - ff[0];
            d_short[base+3] = 0.0;
            d_short[base+4] = aa[nc-1]*y2[nc-2];
            d_short[base+5] = bb[nc-1] - cc[nc-1];
            d_short[base+6] = aa[nc-1]*y1[nc-2] - ff[nc-1];
            d_short[base+7] = 0.0;
        }

        double t_comm_start = MPI_Wtime();
        MPI_Allreduce(d_short, e_short, n_coeffs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t_comm = MPI_Wtime() - t_comm_start;

        double *a_s = malloc(n_short*sizeof(double));
        double *b_s = malloc(n_short*sizeof(double));
        double *c_s = malloc(n_short*sizeof(double));
        double *f_s = malloc(n_short*sizeof(double));
        double *y_s = malloc(n_short*sizeof(double));

        for(int i=0; i<n_short; i++) {
            a_s[i] = e_short[4*i];
            b_s[i] = e_short[4*i+1];
            c_s[i] = e_short[4*i+2];
            f_s[i] = e_short[4*i+3];
        }
        a_s[0] = 0.0; b_s[n_short-1] = 0.0;
        local_sweep(n_short, a_s, b_s, c_s, f_s, y_s, 0.0);

        y = malloc(nc*sizeof(double));
        if(mp == 0) {
            double c1 = y_s[0];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + c1*y2[i];
        } else if(mp == np-1) {
            double a1 = y_s[n_short-1];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + a1*y3[i];
        } else {
            double a1 = y_s[2*mp-2];
            double c1 = y_s[2*mp-1];
            #pragma omp parallel for
            for(int i=0; i<nc; i++) y[i] = y1[i] + a1*y3[i] + c1*y2[i];
        }
        free(a_s); free(b_s); free(c_s); free(f_s); free(y_s);
        free(d_short); free(e_short);
    } else {
        y = malloc(nc*sizeof(double));
        #pragma omp parallel for
        for(int i=0; i<nc; i++) y[i] = y1[i];
    }

    /* Оценка погрешности  */
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
        printf("nx=%d np=%d t_comp=%.4lf t_comm=%.4lf dmax=%.4e\n",
               nx, np, t_end - t_start, 0.0, dmax_glob);
    }

    free(x); free(aa); free(bb); free(cc); free(ff);
    free(y1); if(y2) free(y2); if(y3) free(y3); free(y);
    MPI_Finalize();
    return 0;
}