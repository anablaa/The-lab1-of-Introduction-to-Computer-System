// DO NOT MODIFY THIS FILE
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "load.h"
#include "compute.h"
#include "compare.h"

// TA will change it to a larger value when grading
#define T 10

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: ./main filename load_select compute_select compare_select\n");
        return 1;
    } else {
        LOAD_SELECT = atoi(argv[2]);
        COMPUTE_SELECT = atoi(argv[3]);
        COMPARE_SELECT = atoi(argv[4]);
    }

    if(load(argv[1])) {
        fprintf(stderr, "Read file failed\n");
        return 1;
    }

    uint64_t total_time = 0;
    uint64_t n_round = 0;
    while (total_time < T * SEC) {
        total_time += compute();
        n_round += 1;
    }
    uint64_t avg = total_time / n_round;

    int s = avg / SEC;
    int ms = (avg / 1000000) % 1000;
    int us = (avg / 1000) % 1000;
    int ns = avg % 1000;
    printf("Elapsed time (Human readable): %ds %dms %dus %dns\n", s, ms, us, ns);
    printf("Elapsed time (Machine readable): %" PRId64 "\n", avg);

    if (compare()) {
        fprintf(stderr, "Comparison failed\n");
        return 1;
    };
    return 0;
}
