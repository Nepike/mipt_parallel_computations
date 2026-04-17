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

typedef struct {
    double sum;
    int ready;
} shared_data_t;

typedef struct {
    int mt, nt;
    double a, b;
    int n;
    double local_sum;
    pthread_mutex_t* mut;
    double* proc_sum;
} thread_arg_t;

double f(double x) { return 4.0 / (1.0 + x * x); }

double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a + h * (i + 0.5);
        s += func(x);
    }
    return h * s;
}

void* myjob(void* arg) {
    thread_arg_t* td = (thread_arg_t*)arg;
    double h_th = (td->b - td->a) / td->nt;
    double a1 = td->a + h_th * td->mt;
    double b1 = (td->mt == td->nt - 1) ? td->b : a1 + h_th;
    int n1 = td->n / td->nt;
    
    td->local_sum = integrate(f, a1, b1, n1);
    fprintf(stderr, "mp=%d mt=%d a1=%le b1=%le n1=%d s1=%le\n",
            td->mt, td->mt, a1, b1, n1, td->local_sum);
    
    pthread_mutex_lock(td->mut);
    *(td->proc_sum) += td->local_sum;
    pthread_mutex_unlock(td->mut);
    return NULL;
}

union semun { int val; struct semid_ds* buf; unsigned short* array; };

int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Usage: %s <np> <nt>\n", argv[0]); return 1; }
    
    int np = atoi(argv[1]), nt = atoi(argv[2]);
    if (np < 1) np = 1; if (nt < 1) nt = 1;
    
    double t_start = omp_get_wtime();
    int mp = 0;
    pid_t spid = 0;
    double proc_sum = 0.0;
    pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
    
    /* Создаём ресурсы ДО fork, чтобы все процессы видели одно и то же */
    int shmid = shmget(SHM_ID, sizeof(shared_data_t), PERMS | IPC_CREAT);
    int semid = semget(SEM_ID, 1, PERMS | IPC_CREAT);
    if (shmid < 0 || semid < 0) { perror("shmget/semget"); exit(1); }
    
    shared_data_t* shm = (shared_data_t*)shmat(shmid, NULL, 0);
    if (shm == (void*)-1) { perror("shmat"); exit(1); }
    
    union semun arg;
    arg.val = 1;
    semctl(semid, 0, SETVAL, arg);  /* Инициализация семафора = 1 */
    
    if (mp == 0) { shm->sum = 0.0; shm->ready = 0; }  /* Только мастер инициализирует */
    
    /* Fork: создаём np-1 потомков */
    for (int i = 1; i < np; i++) {
        if (spid == 0) {
            mp = i;
            spid = fork();
            if (spid < 0) { perror("fork"); exit(1); }
            if (spid == 0) break;
        }
    }
    
    /* Распределение работы */
    double h_proc = 1.0 / np;
    double a_proc = h_proc * mp;
    double b_proc = (mp == np - 1) ? 1.0 : a_proc + h_proc;
    int total_n = 1000000000;
    
    /* Запуск потоков */
    pthread_t* threads = malloc(nt * sizeof(pthread_t));
    thread_arg_t* targs = malloc(nt * sizeof(thread_arg_t));
    
    for (int t = 0; t < nt; t++) {
        targs[t].mt = t; targs[t].nt = nt;
        targs[t].a = a_proc; targs[t].b = b_proc;
        targs[t].n = total_n;
        targs[t].mut = &mut; targs[t].proc_sum = &proc_sum;
        pthread_create(&threads[t], NULL, myjob, &targs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(threads[t], NULL);
    free(threads); free(targs);
    
    /* Агрегация в общую память — атомарно через семафор */
    struct sembuf lock = {0, -1, 0}, unlock = {0, +1, 0};
    
    semop(semid, &lock, 1);      /* P-операция: захват */
    shm->sum += proc_sum;
    shm->ready++;
    semop(semid, &unlock, 1);    /* V-операция: освобождение */
    
    /* Мастер ждёт, пока все процессы завершат агрегацию */
    if (mp == 0) {
        while (shm->ready < np) usleep(100);  /* Активное ожидание без захвата семафора! */
        
        double t_elapsed = omp_get_wtime() - t_start;
        printf("time=%lf sum= %le\n", t_elapsed, shm->sum);
        
        /* Очистка ресурсов */
        shmdt(shm);
        shmctl(SHM_ID, IPC_RMID, NULL);
        semctl(semid, 0, IPC_RMID);
    } else {
        shmdt(shm);
        exit(0);
    }
    
    return 0;
}