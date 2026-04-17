#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <pthread.h>
#include <omp.h>

#define MSG_ID 7777
#define MSG_PERM 00666
#define MAX_NI 1000000000

/* Структура сообщения для IPC */
typedef struct {
    long mtype;
    double sum;
} msg_t;

/* Параметры для потока */
typedef struct {
    int tid;
    double a, b;
    int n;
    double result;
} thread_data_t;

/* Интегрируемая функция */
double f1(double x) { return 4.0 / (1.0 + x * x); }

/* Метод средних прямоугольников */
double integrate(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += f(a + h * (i + 0.5));
    return h * s;
}

/* Функция потока */
void* thread_func(void* arg) {
    thread_data_t* td = (thread_data_t*)arg;
    td->result = integrate(f1, td->a, td->b, td->n);
    return NULL;
}

int main(int argc, char* argv[]) {
    int np = 1, nt = 1;
    if (argc >= 3) { np = atoi(argv[1]); nt = atoi(argv[2]); }
    if (np < 1) np = 1;
    if (nt < 1) nt = 1;

    pid_t spid = 0;
    int mp = 0;
    int msgid = -1;

    double t_start = omp_get_wtime();

    /* Создание тяжелых процессов */
    for (int i = 1; i < np; i++) {
        if (spid == 0) {
            mp = i;
            spid = fork();
            if (spid < 0) { perror("fork"); exit(1); }
            if (spid == 0) break; /* Потомок выходит из цикла */
        }
    }

    /*Распределение работы для текущего процесса */
    double h_proc = 1.0 / np;
    double a_proc = h_proc * mp;
    double b_proc = (mp == np - 1) ? 1.0 : a_proc + h_proc;
    double h_th = (b_proc - a_proc) / nt;
    int n_th = MAX_NI / (np * nt);

    pthread_t threads[nt];
    thread_data_t tdata[nt];

    /*Создание и запуск потоков */
    for (int t = 0; t < nt; t++) {
        tdata[t].tid = t;
        tdata[t].a = a_proc + h_th * t;
        tdata[t].b = (t == nt - 1) ? b_proc : a_proc + h_th * (t + 1);
        tdata[t].n = n_th;
        pthread_create(&threads[t], NULL, thread_func, &tdata[t]);
    }

    /* суммирование */
    double proc_sum = 0.0;
    for (int t = 0; t < nt; t++) {
        pthread_join(threads[t], NULL);
        proc_sum += tdata[t].result;
        fprintf(stderr, "mp=%d mt=%d a1=%le b1=%le n1=%d s1=%le\n",
                mp, t, tdata[t].a, tdata[t].b, tdata[t].n, tdata[t].result);
    }

    /* Обмен через очередь сообщений System V */
    if (mp == 0) {
        msgid = msgget(MSG_ID, MSG_PERM | IPC_CREAT);
        if (msgid < 0) { perror("msgget master"); exit(1); }
        for (int i = 1; i < np; i++) {
            msg_t msg;
            msgrcv(msgid, &msg, sizeof(double), 0, 0);
            proc_sum += msg.sum;
        }
        msgctl(msgid, IPC_RMID, NULL);
    } 
    else {
        while ((msgid = msgget(MSG_ID, MSG_PERM)) < 0) usleep(100);
        msg_t msg;
        msg.mtype = mp;
        msg.sum = proc_sum;
        msgsnd(msgid, &msg, sizeof(double), 0);
    }

    double t_elapsed = omp_get_wtime() - t_start;

    if (mp == 0) {
        printf("time=%lf sum= %le\n", t_elapsed, proc_sum);
    }

    return 0;
}