#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <inttypes.h>

#define C 64
#define L1 32*1024
#define L2 1024*1024

#define M 1024
#define K 32*1024
#define N 1024

#define SEC 1000000000L

typedef enum LOAD {
    LOAD_ROW_MAJOR = 1,
    LOAD_ROW_MAJOR_32 = 2,
    LOAD_ROW_MAJOR_16 = 3,
    LOAD_Y_TRANSPOSE = 11,
    LOAD_Y_TRANSPOSE_32 = 12,
    LOAD_Y_TRANSPOSE_16 = 13,
    LOAD_P_ROW_MAJOR = 21,
    LOAD_P_ROW_MAJOR_32 = 22,
    LOAD_P_ROW_MAJOR_16 = 23,
    LOAD_P_Y_TRANSPOSE = 31,
    LOAD_P_Y_TRANSPOSE_32 = 32,
    LOAD_P_Y_TRANSPOSE_16 = 33,
} LOAD;

typedef enum COMPUTE {
    COMPUTE_ROW_MAJOR_MNK = 1,
    COMPUTE_ROW_MAJOR_MKN = 2,
    COMPUTE_ROW_MAJOR_KMN = 3,
    COMPUTE_ROW_MAJOR_NMK = 4,
    COMPUTE_ROW_MAJOR_NKM = 5,
    COMPUTE_ROW_MAJOR_KNM = 6,
    COMPUTE_Y_TRANSPOSE_MNK = 101,
    COMPUTE_ROW_MAJOR_MNKKMN_B32 = 111,
    COMPUTE_ROW_MAJOR_MNK_LU2 = 121,
    COMPUTE_SIMD = 201,
    COMPUTE_TASK2_FINAL = 301,
} COMPUTE;

typedef enum COMPARE {
    COMPARE_ROW_MAJOR = 1,
    COMPARE_COLUMN_MAJOR = 2,
    COMPARE_P_ROW_MAJOR = 3,
    COMPARE_P_COLUMN_MAJOR = 4,
} COMPARE;

extern LOAD LOAD_SELECT;            // 声明
extern COMPUTE COMPUTE_SELECT;      // 声明
extern COMPARE COMPARE_SELECT;      // 声明

extern uint32_t m;                  // 声明
extern uint32_t k;                  // 声明
extern uint32_t n;                  // 声明

extern uint64_t X[M][K];            // ...
extern uint64_t Y[K][N];            // ...
extern uint64_t YP[N][K];
extern uint64_t *Xp;
extern uint64_t *Yp;

extern uint32_t X32[M][K];
extern uint32_t Y32[K][N];
extern uint32_t YP32[N][K];
extern uint32_t *Xp32;
extern uint32_t *Yp32;

extern uint16_t X16[M][K];
extern uint16_t Y16[K][N];
extern uint16_t YP16[N][K];
extern uint16_t *Xp16;
extern uint16_t *Yp16;

extern uint64_t Z[M][N]; // target
extern uint64_t ZP[N][M]; // target
extern uint64_t *Zp; // target
extern uint64_t R[M][N]; // reference

#endif // COMMON_H
