#include <stdio.h>
#include <math.h>
#include <omp.h>

int main() {
    long nc = 1000000000;
    double t1, t2, s1 = 0.0, s2 = 0.0;
    double a = 1.23456789, b = 0.987654321;

    /* Деление */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += a / (b + i);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += a / i;
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec '/' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Умножение */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += a * (b + i);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += a * i;
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec '*' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Сложение */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += a + b + i;
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += a + i;
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec '+' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Вычитание */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += a - b - i;
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += a - i;
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec '-' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Возведение в степень */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += pow(a, b);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += pow(a, i * 1e-9);
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec 'pow' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Экспонента */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += exp(b);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += exp(i * 1e-9);
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec 'exp' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Логарифм */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += log(b + 1.0);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += log(i * 1e-9 + 1.0);
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec 'log' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Синус */
    t1 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s1 += sin(b);
    t1 = omp_get_wtime() - t1;
    t2 = omp_get_wtime();
    for(long i = 1; i <= nc; i++) s2 += sin(i * 1e-9);
    t2 = omp_get_wtime() - t2;
    printf("Time: %.6lf %.6lf sec 'sin' perf.: %e GFlops\n", t1, t2, 1.0 / fabs(t2 - t1));

    /* Защита от оптимизации компилятором */
    if (s1 + s2 < -1e300) printf("");
    return 0;
}