#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <pthread.h>
#include <omp.h>

#define SHM_ID 2008
#define SEM_ID 2007
#define PERMS 0600

typedef struct { double sum; int count; } shared_data_t;

typedef struct {
    int mp, mt, nt, n;
    double a, b, local_sum;
    pthread_mutex_t *mut;
    double *proc_sum;
} thread_arg_t;

double f(double x) { return 4.0 / (1.0 + x * x); }

double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += func(a + h * (i + 0.5));
    return h * s;
}

void* thread_func(void* arg) {
    thread_arg_t* td = (thread_arg_t*)arg;
    double h_th = (td->b - td->a) / td->nt;
    double a1 = td->a + h_th * td->mt;
    double b1 = (td->mt == td->nt - 1) ? td->b : a1 + h_th;
    int n1 = td->n / td->nt;
    
    td->local_sum = integrate(f, a1, b1, n1);
    fprintf(stderr, "mp=%d mt=%d a1=%le b1=%le n1=%d s1=%le\n",
            td->mp, td->mt, a1, b1, n1, td->local_sum);
    
    pthread_mutex_lock(td->mut);
    *(td->proc_sum) += td->local_sum;
    pthread_mutex_unlock(td->mut);
    return NULL;
}

#ifdef _SEM_SEMUN_UNDEFINED
union semun { int val; struct semid_ds* buf; unsigned short* array; };
#endif

int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Usage: %s <np> <nt>\n", argv[0]); return 1; }
    int np = atoi(argv[1]), nt = atoi(argv[2]);
    if (np < 1) np = 1; if (nt < 1) nt = 1;
    
    setbuf(stderr, NULL); // Мгновенный вывод stderr без буферизации
    double t_start = omp_get_wtime();
    
    /* 1. Создание и инициализация ресурсов ДО ветвления */
    int shmid = shmget(SHM_ID, sizeof(shared_data_t), PERMS | IPC_CREAT);
    int semid = semget(SEM_ID, 1, PERMS | IPC_CREAT);
    if (shmid < 0 || semid < 0) { perror("shmget/semget"); exit(1); }
    
    shared_data_t* shm = (shared_data_t*)shmat(shmid, NULL, 0);
    if (shm == (void*)-1) { perror("shmat"); exit(1); }
    shm->sum = 0.0; shm->count = 0;
    
    union semun arg;
    arg.val = 1;
    semctl(semid, 0, SETVAL, arg);
    
    /* 2. Создание тяжелых процессов */
    int mp = 0; pid_t spid = 0;
    for (int i = 1; i < np; i++) {
        if (spid == 0) { mp = i; spid = fork(); if (spid < 0) { perror("fork"); exit(1); } if (spid == 0) break; }
    }
    
    /* 3. Локальные переменные процесса */
    double h_proc = 1.0 / np;
    double a_proc = h_proc * mp;
    double b_proc = (mp == np - 1) ? 1.0 : a_proc + h_proc;
    int total_n = 1000000000;
    double proc_sum = 0.0;
    pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
    
    /* 4. Создание и ожидание легких процессов */
    pthread_t* threads = malloc(nt * sizeof(pthread_t));
    thread_arg_t* targs = malloc(nt * sizeof(thread_arg_t));
    for (int t = 0; t < nt; t++) {
        targs[t] = (thread_arg_t){mp, t, nt, total_n, a_proc, b_proc, 0.0, &mut, &proc_sum};
        pthread_create(&threads[t], NULL, thread_func, &targs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(threads[t], NULL);
    free(threads); free(targs);
    
    /* 5. Агрегация в общую память под семафором */
    struct sembuf lock = {0, -1, 0}, unlock = {0, +1, 0};
    semop(semid, &lock, 1);
    shm->sum += proc_sum;
    shm->count++;
    semop(semid, &unlock, 1);
    
    /* 6. Мастер ждет, дети выходят */
    if (mp == 0) {
        while (shm->count < np) usleep(100);
        double t_elapsed = omp_get_wtime() - t_start;
        printf("time=%lf sum= %le\n", t_elapsed, shm->sum);
        shmdt(shm); shmctl(SHM_ID, IPC_RMID, NULL); semctl(semid, 0, IPC_RMID);
    } else {
        shmdt(shm); exit(0);
    }
    return 0;
}