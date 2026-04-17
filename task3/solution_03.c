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

/* Структура для общей памяти: сумма и счётчик готовых процессов */
typedef struct {
    double sum;
    int ready;
} shared_data_t;

/* Параметры для потока */
typedef struct {
    int mt;           /* номер потока внутри процесса */
    int nt;           /* всего потоков в процессе */
    double a, b;      /* границы интегрирования для процесса */
    int n;            /* общее число узлов */
    double local_sum; /* результат потока */
    pthread_mutex_t* mut; /* мьютекс процесса */
    double* proc_sum; /* указатель на сумму процесса */
} thread_arg_t;

/* Интегрируемая функция: 4/(1+x²) -> интеграл = π */
double f(double x) { return 4.0 / (1.0 + x * x); }

/* Метод средних прямоугольников */
double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a + h * (i + 0.5);
        s += func(x);
    }
    return h * s;
}

/* Функция потока: вычисляет часть интеграла */
void* myjob(void* arg) {
    thread_arg_t* td = (thread_arg_t*)arg;
    
    /* Разбиваем отрезок [a,b] процесса на части для потоков */
    double h_th = (td->b - td->a) / td->nt;
    double a1 = td->a + h_th * td->mt;
    double b1 = (td->mt == td->nt - 1) ? td->b : a1 + h_th;
    int n1 = td->n / (td->nt);
    
    td->local_sum = integrate(f, a1, b1, n1);
    
    fprintf(stderr, "mp=%d mt=%d a1=%le b1=%le n1=%d s1=%le\n",
            td->mt, td->mt, a1, b1, n1, td->local_sum);
    
    /* Критическая секция: добавляем результат потока в сумму процесса */
    pthread_mutex_lock(td->mut);
    *(td->proc_sum) += td->local_sum;
    pthread_mutex_unlock(td->mut);
    
    return NULL;
}

/* Инициализация семафора и общей памяти (только мастер) */
void init_shared(shared_data_t** shm, int* semid) {
    /* Создаём сегмент общей памяти */
    int shmid = shmget(SHM_ID, sizeof(shared_data_t), PERMS | IPC_CREAT);
    if (shmid < 0) {
        perror("shmget");
        exit(1);
    }
    *shm = (shared_data_t*)shmat(shmid, NULL, 0);
    if (*shm == (void*)-1) {
        perror("shmat");
        exit(1);
    }
    (*shm)->sum = 0.0;
    (*shm)->ready = 0;
    
    /* Создаём семафор для синхронизации доступа */
    *semid = semget(SEM_ID, 1, PERMS | IPC_CREAT);
    if (*semid < 0) {
        perror("semget");
        exit(1);
    }
    union semun { int val; struct semid_ds* buf; unsigned short* array; } arg;
    arg.val = 1; /* начальное значение = 1 (разрешён доступ) */
    semctl(*semid, 0, SETVAL, arg);
}

/* Очистка ресурсов (только мастер) */
void cleanup_shared(shared_data_t* shm, int semid) {
    shmdt(shm);
    shmctl(SHM_ID, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <num_processes> <num_threads>\n", argv[0]);
        return 1;
    }
    
    int np = atoi(argv[1]); /* число процессов */
    int nt = atoi(argv[2]); /* число потоков в процессе */
    if (np < 1) np = 1;
    if (nt < 1) nt = 1;
    
    double t_start = omp_get_wtime();
    
    /* Переменные для каждого процесса */
    int mp = 0;              /* логический номер процесса */
    pid_t spid = 0;          /* PID дочернего процесса */
    double proc_sum = 0.0;   /* сумма интеграла в этом процессе */
    pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
    
    /* === Создание тяжёлых процессов === */
    for (int i = 1; i < np; i++) {
        if (spid == 0) {
            mp = i;
            spid = fork();
            if (spid < 0) { perror("fork"); exit(1); }
            if (spid == 0) break; /* потомок выходит из цикла */
        }
    }
    
    /* === Инициализация общей памяти (только мастер) === */
    shared_data_t* shm = NULL;
    int semid = -1;
    if (mp == 0) {
        init_shared(&shm, &semid);
    } else {
        /* Потомки ждут, пока мастер создаст ресурсы */
        while ((shm = (shared_data_t*)shmat(shmget(SHM_ID, sizeof(shared_data_t), PERMS), NULL, 0)) == (void*)-1)
            usleep(100);
        while ((semid = semget(SEM_ID, 1, PERMS)) < 0)
            usleep(100);
    }
    
    /* === Распределение работы для процесса === */
    double h_proc = 1.0 / np;
    double a_proc = h_proc * mp;
    double b_proc = (mp == np - 1) ? 1.0 : a_proc + h_proc;
    int total_n = 1000000000; /* общее число узлов */
    
    /* === Создание и запуск лёгких процессов (потоков) === */
    pthread_t* threads = malloc(nt * sizeof(pthread_t));
    thread_arg_t* targs = malloc(nt * sizeof(thread_arg_t));
    
    for (int t = 0; t < nt; t++) {
        targs[t].mt = t;
        targs[t].nt = nt;
        targs[t].a = a_proc;
        targs[t].b = b_proc;
        targs[t].n = total_n;
        targs[t].mut = &mut;
        targs[t].proc_sum = &proc_sum;
        pthread_create(&threads[t], NULL, myjob, &targs[t]);
    }
    
    /* === Ожидание потоков === */
    for (int t = 0; t < nt; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(targs);
    
    /* === Агрегация результатов между процессами === */
    if (mp == 0) {
        /* Мастер ждёт, пока все процессы завершатся */
        struct sembuf wait_op = {0, -1, 0}; /* P-операция */
        struct sembuf post_op = {0, +1, 0}; /* V-операция */
        
        for (int i = 0; i < np; i++) {
            /* Ждём, пока процесс не запишет свою сумму */
            while (shm->ready < np) {
                semop(semid, &wait_op, 1);
                if (shm->ready == np) break;
                semop(semid, &post_op, 1);
                usleep(100);
            }
        }
        proc_sum = shm->sum;
        cleanup_shared(shm, semid);
    } else {
        /* Потомок записывает свою сумму в общую память */
        struct sembuf lock = {0, -1, 0};
        struct sembuf unlock = {0, +1, 0};
        
        semop(semid, &lock, 1);
        shm->sum += proc_sum;
        shm->ready++;
        semop(semid, &unlock, 1);
        
        shmdt(shm);
        exit(0);
    }
    
    double t_elapsed = omp_get_wtime() - t_start;
    printf("time=%lf sum= %le\n", t_elapsed, proc_sum);
    
    return 0;
}