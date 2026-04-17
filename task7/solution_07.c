#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Глобальные параметры области */
const double XA = 0.0, XB = 1.0;
double UA, UB; /* Граничные значения, вычисляются из тестовой функции */
double px, px2;

/* Тестовая функция и её производные */
double u(double x)   { return cos(px*(x-XA)) + sin(px*(x-XA)); }
double u1(double x)  { return px*(-sin(px*(x-XA)) + cos(px*(x-XA))); }
double u2(double x)  { return -px2*(cos(px*(x-XA)) + sin(px*(x-XA))); }

/* Коэффициенты уравнения и правая часть */
double k(double x)   { return 1.0 + exp(5.0*(x-XA)/(XB-XA)); }
double kp(double x)  { return 5.0/(XB-XA) * exp(5.0*(x-XA)/(XB-XA)); }
double q(double x)   { return 1.0 - 0.5*sin(8.0*M_PI*(x-XA)/(XB-XA)); }
double f_rhs(double x){ return q(x)*u(x) - k(x)*u2(x) - kp(x)*u1(x); }

/* Локальная правая прогонка для системы: a*y_{i-1} - c*y_i + b*y_{i+1} = -f */
void sweep_right(int n, double *a, double *c, double *b, double *f, double *y) {
    double *alpha = malloc((n+1)*sizeof(double));
    double *beta  = malloc((n+1)*sizeof(double));
    alpha[0] = 0.0; beta[0] = 0.0;
    for(int i=0; i<n; i++) {
        double d = c[i] - a[i]*alpha[i];
        alpha[i+1] = b[i] / d;
        beta[i+1]  = (a[i]*beta[i] - f[i]) / d;
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

    /* Граничные условия согласованы с тестовой функцией */
    px = 0.5*M_PI/(XB-XA); px2 = px*px;
    UA = u(XA);
    UB = u1(XB);

    double hx = (XB - XA) / nx;
    double hx2 = hx * hx;

    /* Декомпозиция области: равномерное разбиение [0, nx] на np частей */
    int i1, i2, nc;
    i1 = mp * (nx / np);
    nc = (mp < np - 1) ? nx / np : nx - i1;
    i2 = i1 + nc - 1;

    /* Выделение памяти */
    double *xx = malloc(nc * sizeof(double));
    double *aa = calloc(nc, sizeof(double)); /* a[i] */
    double *bb = calloc(nc, sizeof(double)); /* b[i] */
    double *cc = calloc(nc, sizeof(double)); /* c[i] */
    double *ff = calloc(nc, sizeof(double)); /* f[i] */
    double *y1 = malloc(nc * sizeof(double));

    #pragma omp parallel for
    for(int i=0; i<nc; i++) xx[i] = XA + hx*(i1 + i);

    /* Заполнение коэффициентов СЛАУ */
    #pragma omp parallel for
    for(int i=0; i<nc; i++) {
        double xi = xx[i];
        double km = 0.5*(k(xi) + (i>0 ? k(xi-hx) : k(xi)));
        double kp2 = 0.5*(k(xi) + (i<nc-1 ? k(xi+hx) : k(xi)));
        aa[i] = km / hx;
        bb[i] = kp2 / hx;
        cc[i] = aa[i] + bb[i] + hx * q(xi);
        ff[i] = hx * f_rhs(xi);
    }

    /* Граничные условия */
    if(mp == 0) { /* Дирихле слева: u(a) = UA */
        aa[0] = 0.0; bb[0] = 0.0; cc[0] = 1.0; ff[0] = UA;
    }
    if(mp == np - 1) { /* Нейман справа: u'(b) = UB */
        cc[nc-1] -= bb[nc-1];
        ff[nc-1] += bb[nc-1] * hx * UB;
        bb[nc-1] = 0.0;
    }

    double t_start = MPI_Wtime();
    double t_comp = 0.0, t_comm = 0.0;

    if(np < 2) {
        sweep_right(nc, aa, cc, bb, ff, y1);
    } else {
        /* Параллельная прогонка: базисные функции y1, y2, y3 */
        double *y2 = malloc(nc*sizeof(double));
        double *y3 = malloc(nc*sizeof(double));
        double *y4 = malloc((2*(np-1))*sizeof(double));
        double *dd = calloc(4*(2*(np-1)), sizeof(double));
        double *ee = calloc(4*(2*(np-1)), sizeof(double));

        /* Сохраняем коэффициенты границ */
        double a0=aa[0], b0=bb[0], c0=cc[0], f0=ff[0];
        double a1=aa[nc-1], b1=bb[nc-1], c1=cc[nc-1], f1=ff[nc-1];

        /* 1. Решение для правой части (y1) */
        if(mp==0) { aa[nc-1]=0; bb[nc-1]=0; cc[nc-1]=1; ff[nc-1]=0; }
        else      { aa[0]=0; bb[0]=0; cc[0]=1; ff[0]=0; aa[nc-1]=0; bb[nc-1]=0; cc[nc-1]=1; ff[nc-1]=0; }
        sweep_right(nc, aa, cc, bb, ff, y1);

        /* 2. Базис для левой границы подрегиона (y2) */
        if(mp<np-1) {
            for(int i=0;i<nc-1;i++) ff[i]=0; ff[nc-1]=1.0;
            sweep_right(nc, aa, cc, bb, ff, y2);
        }
        /* 3. Базис для правой границы подрегиона (y3) */
        if(mp>0) {
            ff[0]=1.0; for(int i=1;i<nc;i++) ff[i]=0.0;
            sweep_right(nc, aa, cc, bb, ff, y3);
        }

        /* 4. Формирование короткой системы */
        int idx = mp * 8 - 4;
        if(mp == 0) {
            c1 -= a1*y2[nc-2]; f1 += a1*y1[nc-2];
            dd[0]=0; dd[1]=b1; dd[2]=c1; dd[3]=f1;
        } else if(mp == np-1) {
            c0 -= b0*y3[1]; f0 += b0*y1[1]; b0=0;
            dd[idx]=a0; dd[idx+1]=b0; dd[idx+2]=c0; dd[idx+3]=f0;
        } else {
            c0 -= b0*y3[1]; f0 += b0*y1[1]; b0 *= y2[1];
            c1 -= a1*y2[nc-2]; f1 += a1*y1[nc-2]; a1 *= y3[nc-2];
            dd[idx]=a0; dd[idx+1]=b0; dd[idx+2]=c0; dd[idx+3]=f0;
            dd[idx+4]=a1; dd[idx+5]=b1; dd[idx+6]=c1; dd[idx+7]=f1;
        }

        /* 5. Сборка и решение короткой системы */
        t_comm = MPI_Wtime();
        MPI_Allreduce(dd, ee, 4*(2*(np-1)), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t_comm = MPI_Wtime() - t_comm;

        for(int i=0; i<2*(np-1); i++) {
            aa[i]=ee[4*i]; bb[i]=ee[4*i+1]; cc[i]=ee[4*i+2]; ff[i]=ee[4*i+3];
        }
        /* Коррекция границ короткой системы */
        aa[0]=0; bb[2*(np-1)-1]=0;
        sweep_right(2*(np-1), aa, cc, bb, ff, y4);

        /* 6. Восстановление полного решения */
        #pragma omp parallel for
        for(int i=0; i<nc; i++) {
            double val = y1[i];
            if(mp>0)     val += y4[2*mp-2] * y3[i];
            if(mp<np-1)  val += y4[2*mp-1] * y2[i];
            y1[i] = val;
        }
        free(y2); free(y3); free(y4); free(dd); free(ee);
    }
    t_comp = MPI_Wtime() - t_start;

    /* Оценка максимальной погрешности */
    double dmax = 0.0;
    #pragma omp parallel for reduction(max:dmax)
    for(int i=0; i<nc; i++) {
        double err = fabs(u(xx[i]) - y1[i]);
        if(err > dmax) dmax = err;
    }
    double dmax_glob = 0.0;
    MPI_Allreduce(&dmax, &dmax_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if(mp == 0) {
        printf("nx=%d np=%d t_comp=%.4lf t_comm=%.4lf dmax=%.4e\n",
               nx, np, t_comp, t_comm, dmax_glob);
    }

    free(xx); free(aa); free(bb); free(cc); free(ff); free(y1);
    MPI_Finalize();
    return 0;
}