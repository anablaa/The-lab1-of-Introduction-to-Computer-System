#include <stdio.h>

#include "common.h"
#include "compare.h"

int compare_row_major() {
    for (int i = 0; i!= m; ++i) {
        for (int j = 0; j!= n; ++j) {
            //printf("%d, %d: %016" PRIx64 ", %016" PRIx64 "\n", i, j, Z[i][j], R[i][j]);
            if (Z[i][j] != R[i][j]) return 1;
        }
    }
    return 0;
}

int compare() {
    switch (COMPARE_SELECT) {
        case COMPARE_ROW_MAJOR:
            //printf("COMPARE_ROW_MAJOR\n");
            return compare_row_major();
        default:
            printf("Unreachable!");
            return 1;
    }
}
