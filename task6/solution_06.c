#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

/* Компаратор для qsort: возвращает -1, 0 или 1 без риска переполнения */
int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int np, mp;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &mp);

    int ns = (argc > 1) ? atoi(argv[1]) : 10000;
    if (ns < np) ns = np;

    /* Равномерное распределение: остаток отдается последнему процессу */
    int base_nc = ns / np;
    int rem = ns % np;
    int nc = base_nc + (mp == np - 1 ? rem : 0);

    /* Выделение локального буфера */
    double *a1 = (double*)malloc(nc * sizeof(double));
    if (!a1) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Генерация случайных чисел (разные seed для каждого процесса) */
    srand(time(NULL) + mp);
    for (int i = 0; i < nc; i++) a1[i] = (double)rand() / RAND_MAX * 32768.0;

    double t1 = MPI_Wtime();

    /* Локальная сортировка каждого фрагмента */
    qsort(a1, nc, sizeof(double), cmp_double);

    /* Буферы для обмена и временного слияния */
    int max_nc = base_nc + (rem > 0 ? 1 : 0);
    double *a2 = (double*)malloc(max_nc * sizeof(double));
    double *tmp = (double*)malloc((nc + max_nc) * sizeof(double));

    MPI_Status status;
    /* Четно-нечетная перестановочная сортировка: np фаз гарантируют глобальную упорядоченность */
    for (int phase = 0; phase < np; phase++) {
        int partner = -1;
        if ((phase + mp) % 2 == 0 && mp + 1 < np) partner = mp + 1;
        else if ((phase + mp) % 2 != 0 && mp - 1 >= 0) partner = mp - 1;

        if (partner != -1) {
            int partner_nc = base_nc + (partner == np - 1 ? rem : 0);

            /* Атомарный обмен с соседом */
            MPI_Sendrecv(a1, nc, MPI_DOUBLE, partner, 0,
                         a2, max_nc, MPI_DOUBLE, partner, 0,
                         MPI_COMM_WORLD, &status);

            int recv_cnt;
            MPI_Get_count(&status, MPI_DOUBLE, &recv_cnt);

            /* Слияние двух отсортированных массивов во временный буфер */
            int i = 0, j = 0, k = 0;
            while (i < nc && j < recv_cnt) {
                tmp[k++] = (a1[i] < a2[j]) ? a1[i++] : a2[j++];
            }
            while (i < nc) tmp[k++] = a1[i++];
            while (j < recv_cnt) tmp[k++] = a2[j++];

            /* Распределение: процесс с меньшим рангом забирает меньшие элементы */
            if (mp < partner) {
                for (int x = 0; x < nc; x++) a1[x] = tmp[x];
            } else {
                int start = recv_cnt;
                for (int x = 0; x < nc; x++) a1[x] = tmp[start + x];
            }
        }
    }

    double t2 = MPI_Wtime();
    double amin = a1[0];
    double amax = a1[nc - 1];

    fprintf(stderr, "mp=%d ns=%d i1=%d i2=%d nc=%d amin=%le amax=%le t1=%le\n",
            mp, ns, 0, ns-1, nc, amin, amax, t2 - t1);

    free(a1); free(a2); free(tmp);
    MPI_Finalize();
    return 0;
}