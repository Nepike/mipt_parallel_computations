#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
#include "mycom.h"
#include "mynet.h"
#include "myprog.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int np, mp, nl, ier, lp;
char pname[MPI_MAX_PROCESSOR_NAME];
char sname[14] = "task7a.p00";
MPI_Status status;
union_t buf;
double tick, t1, t2, t3;
FILE *Fi = NULL;
FILE *Fo = NULL;

int nx;
double xa, xb, ua, ub, px, px2;

/* Коэффициенты из задания 7a */
double k(double x) {
    double t = (x - xa) / (xb - xa);
    return 1.0 + exp(-5.0 * t);
}
double k1(double x) {
    double t = (x - xa) / (xb - xa);
    return -5.0/(xb-xa) * exp(-5.0 * t);
}
double q(double x) {
    double t = (x - xa) / (xb - xa);
    return 1.0 - 0.5 * sin(8.0 * t);
}

/* Тестовая функция (выбрана самостоятельно) */
double u(double x) {
    double t = (x - xa) / (xb - xa);
    return cos(px * (x-xa)) + sin(px * (x-xa));
}
double u1(double x) {
    return px * (-sin(px*(x-xa)) + cos(px*(x-xa)));
}
double u2(double x) {
    return -px*px * (cos(px*(x-xa)) + sin(px*(x-xa)));
}

/* Правая часть */
double f(double x) {
    return -k1(x)*u1(x) - k(x)*u2(x) + q(x)*u(x);
}

int main(int argc, char *argv[])
{
    int i, j, i1, i2, nc, ncm, ncp, ncx;
    double hx, hx2, s0, s1, s2, a0, b0, c0, f0, a1, b1, c1, f1;
    double *xx, *aa, *bb, *cc, *dd, *ee, *ff, *al, *y1, *y2, *y3, *y4;

    MyNetInit(&argc, &argv, &np, &mp, &nl, pname, &tick);
    fprintf(stderr,"Netsize: %d, process: %d, system: %s\n", np, mp, pname);
    sleep(1);

    sprintf(sname+10, "%02d", mp);  // имя файла протокола
    ier = fopen_m(&Fo, sname, "wt");
    if (ier != 0) mpierr("Protocol file not opened", 1);

    if (mp == 0) {
        ier = fopen_m(&Fi, "task7a.d", "rt");
        if (ier != 0) mpierr("Data file not opened", 2);
        i = fscanf(Fi, "xa=%le\n", &xa);
        i = fscanf(Fi, "xb=%le\n", &xb);
        i = fscanf(Fi, "ua=%le\n", &ua);
        i = fscanf(Fi, "ub=%le\n", &ub);
        i = fscanf(Fi, "nx=%d\n",  &nx);
        i = fscanf(Fi, "lp=%d\n",  &lp);
        fclose_m(&Fi);
        if (argc > 1) sscanf(argv[1], "%d", &nx);
    }

    if (np > 1) {
        if (mp == 0) {
            buf.ddata[0] = xa; buf.ddata[1] = xb;
            buf.ddata[2] = ua; buf.ddata[3] = ub;
            buf.idata[8] = nx; buf.idata[9] = lp;
        }
        MPI_Bcast(buf.ddata, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (mp > 0) {
            xa = buf.ddata[0]; xb = buf.ddata[1];
            ua = buf.ddata[2]; ub = buf.ddata[3];
            nx = buf.idata[8]; lp = buf.idata[9];
        }
    }

    fprintf(Fo, "xa=%le xb=%le ua=%le ub=%le nx=%d lp=%d\n", xa, xb, ua, ub, nx, lp);

    t1 = MPI_Wtime();
    px = 0.5*M_PI/(xb-xa); px2 = px*px;
    hx = (xb-xa)/nx; hx2 = hx*hx;

    MyRange(np, mp, 0, nx, &i1, &i2, &nc);
    ncm = nc - 1; ncp = 2*(np-1); ncx = imax(nc, ncp);

    fprintf(Fo, "i1=%d i2=%d nc=%d\n", i1, i2, nc);

    xx = (double*)malloc(sizeof(double)*nc);
    aa = (double*)malloc(sizeof(double)*ncx);
    bb = (double*)malloc(sizeof(double)*ncx);
    cc = (double*)malloc(sizeof(double)*ncx);
    ff = (double*)malloc(sizeof(double)*ncx);
    al = (double*)malloc(sizeof(double)*ncx);
    y1 = (double*)malloc(sizeof(double)*nc);

    for (i = 0; i < nc; i++)
        xx[i] = xa + hx*(i1 + i);

    /* === Заполнение коэффициентов СЛАУ (распараллеливание OpenMP) === */
    #pragma omp parallel for private(i,s0,s1,s2) if(nc>1000)
    for (i = 0; i < nc; i++) {
        if (i == 0) {
            if (mp == 0) {
                aa[i] = 0.0; bb[i] = 0.0; cc[i] = 1.0; ff[i] = ua;
            } else {
                s0 = k(xx[i]); s1 = k(xx[i]-hx); s2 = k(xx[i]+hx);
                aa[i] = 0.5*(s0+s1); bb[i] = 0.5*(s0+s2);
                cc[i] = hx2*q(xx[i]) + aa[i] + bb[i];
                ff[i] = hx2*f(xx[i]);
            }
        }
        else if (i == ncm) {
            if (mp == np-1) {
                // Нейман на правой границе: u'(b)=ub
                s0 = k(xx[i]); s1 = k(xx[i-1]);
                aa[i] = 0.5*(s0+s1); bb[i] = 0.0;
                cc[i] = 0.5*hx2*q(xx[i]) + aa[i];
                ff[i] = 0.5*hx2*f(xx[i]) + hx*ub*s0;
            } else {
                s0 = k(xx[i]); s1 = k(xx[i]-hx); s2 = k(xx[i]+hx);
                aa[i] = 0.5*(s0+s1); bb[i] = 0.5*(s0+s2);
                cc[i] = hx2*q(xx[i]) + aa[i] + bb[i];
                ff[i] = hx2*f(xx[i]);
            }
        }
        else {
            s0 = k(xx[i]); s1 = k(xx[i-1]); s2 = k(xx[i+1]);
            aa[i] = 0.5*(s0+s1); bb[i] = 0.5*(s0+s2);
            cc[i] = hx2*q(xx[i]) + aa[i] + bb[i];
            ff[i] = hx2*f(xx[i]);
        }
    }

    /* === Решение методом правой параллельной прогонки === */
    if (np < 2) {
        ier = prog_right(nc, aa, bb, cc, ff, al, y1);
        if (ier != 0) mpierr("Bad solution 1", 1);
        t2 = 0.0;
    }
    else {
        double *y2, *y3, *y4, *dd2, *ee2;
        y2  = (double*)malloc(sizeof(double)*nc);
        y3  = (double*)malloc(sizeof(double)*nc);
        y4  = (double*)malloc(sizeof(double)*ncp);
        dd2 = (double*)malloc(sizeof(double)*4*ncp);
        ee2 = (double*)malloc(sizeof(double)*4*ncp);

        a0 = aa[0]; b0 = bb[0]; c0 = cc[0]; f0 = ff[0];
        a1 = aa[ncm]; b1 = bb[ncm]; c1 = cc[ncm]; f1 = ff[ncm];

        if (mp == 0) {
            aa[ncm]=0; bb[ncm]=0; cc[ncm]=1; ff[ncm]=0;
            ier = prog_right(nc,aa,bb,cc,ff,al,y1);
            for (i=0;i<ncm;i++) ff[i]=0; ff[ncm]=1;
            ier = prog_right(nc,aa,bb,cc,ff,al,y2);
        }
        else if (mp < np-1) {
            aa[0]=0; bb[0]=0; cc[0]=1; ff[0]=0;
            aa[ncm]=0; bb[ncm]=0; cc[ncm]=1; ff[ncm]=0;
            ier = prog_right(nc,aa,bb,cc,ff,al,y1);
            for (i=0;i<ncm;i++) ff[i]=0; ff[ncm]=1;
            ier = prog_right(nc,aa,bb,cc,ff,al,y2);
            ff[0]=1; for (i=1;i<=ncm;i++) ff[i]=0;
            ier = prog_right(nc,aa,bb,cc,ff,al,y3);
        }
        else {
            aa[0]=0; bb[0]=0; cc[0]=1; ff[0]=0;
            ier = prog_right(nc,aa,bb,cc,ff,al,y1);
            ff[0]=1; for (i=1;i<=ncm;i++) ff[i]=0;
            ier = prog_right(nc,aa,bb,cc,ff,al,y3);
        }

        for (i=0; i<4*ncp; i++) dd2[i]=0;
        for (i=0; i<4*ncp; i++) ee2[i]=0;

        if (mp == 0) {
            c1 = c1 - a1*y2[ncm-1]; f1 = f1 + a1*y1[ncm-1]; a1 = 0;
            dd2[0]=a1; dd2[1]=b1; dd2[2]=c1; dd2[3]=f1;
        }
        else if (mp < np-1) {
            c0 = c0 - b0*y3[1]; f0 = f0 + b0*y1[1]; b0 = b0*y2[1];
            c1 = c1 - a1*y2[ncm-1]; f1 = f1 + a1*y1[ncm-1]; a1 = a1*y3[ncm-1];
            i = mp*8 - 4;
            dd2[i]=a0; dd2[i+1]=b0; dd2[i+2]=c0; dd2[i+3]=f0;
            dd2[i+4]=a1; dd2[i+5]=b1; dd2[i+6]=c1; dd2[i+7]=f1;
        }
        else {
            c0 = c0 - b0*y3[1]; f0 = f0 + b0*y1[1]; b0 = 0;
            i = mp*8 - 4;
            dd2[i]=a0; dd2[i+1]=b0; dd2[i+2]=c0; dd2[i+3]=f0;
        }

        t2 = MPI_Wtime();
        MPI_Allreduce(dd2, ee2, 4*ncp, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t2 = MPI_Wtime() - t2;

        for (i=0; i<ncp; i++) {
            j = 4*i;
            aa[i]=ee2[j]; bb[i]=ee2[j+1]; cc[i]=ee2[j+2]; ff[i]=ee2[j+3];
        }
        ier = prog_right(ncp, aa, bb, cc, ff, al, y4);

        if (mp == 0) {
            b1 = y4[0];
            #pragma omp parallel for if(nc>1000)
            for (i=0; i<nc; i++) y1[i] = y1[i] + b1*y2[i];
        }
        else if (mp < np-1) {
            a1 = y4[2*mp-1]; b1 = y4[2*mp];
            #pragma omp parallel for if(nc>1000)
            for (i=0; i<nc; i++) y1[i] = y1[i] + a1*y3[i] + b1*y2[i];
        }
        else {
            a1 = y4[2*mp-1];
            #pragma omp parallel for if(nc>1000)
            for (i=0; i<nc; i++) y1[i] = y1[i] + a1*y3[i];
        }

        free(y2); free(y3); free(y4); free(dd2); free(ee2);
    }

    t1 = MPI_Wtime() - t1;

    /* Оценка погрешности (максимальная норма) */
    double dmax_local = 0.0;
    #pragma omp parallel for reduction(max:dmax_local) if(nc>1000)
    for (i = 0; i < nc; i++) {
        double u_exact = u(xx[i]);
        double diff = fabs(u_exact - y1[i]);
        if (diff > dmax_local) dmax_local = diff;
    }
    double dmax_global = dmax_local;
    if (np > 1) {
        MPI_Allreduce(&dmax_local, &dmax_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (mp == 0) {
        fprintf(stderr,"nx=%d t1=%le t2=%le dmax=%le\n", nx, t1, t2, dmax_global);
        printf("Результаты:\n");
        printf("  nx = %d\n", nx);
        printf("  MPI процессов = %d\n", np);
        #ifdef _OPENMP
        printf("  OpenMP потоков на процесс = %d\n", omp_get_max_threads());
        #else
        printf("  OpenMP не используется\n");
        #endif
        printf("  Время счёта = %le сек\n", t1);
        printf("  Время обменов MPI = %le сек\n", t2);
        printf("  Макс. погрешность = %le\n", dmax_global);
    }
    fprintf(Fo,"t1=%le t2=%le dmax=%le\n", t1, t2, dmax_global);

    ier = fclose_m(&Fo);
    MPI_Finalize();
    return 0;
}