#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define M 1024
#define K 32*1024
#define N 1024

uint64_t X[M][K];
uint64_t Y[K][N];
uint64_t Z[M][N];

int gen(const char* file, int m, int k, int n) {

    FILE* fp = fopen(file, "w");
    if (!fp) return 1;

    fprintf(fp, "%d %d %d\n", m, k, n);
    fprintf(fp, "\n");

    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != k; ++j) {
            X[i][j] = rand() / (RAND_MAX/UINT16_MAX);
            fprintf(fp, "%04X ", X[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    for (int i = 0; i != k; ++i) {
        for (int j = 0; j != n; ++j) {
            Y[i][j] = rand() / (RAND_MAX/UINT16_MAX);
            fprintf(fp, "%04X ", Y[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }

    for (int i = 0; i!= m; ++i) {
        for (int j = 0; j!= n; ++j) {
            fprintf(fp, "%016" PRIx64 " ", Z[i][j]);
        }
        fprintf(fp, "\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: ./gen file m k n");
        return 1;
    }
    srand(time(NULL));
    gen(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    return 0;
}
