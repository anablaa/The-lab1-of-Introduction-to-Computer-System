#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#ifdef SIMD
#include <arm_neon.h>
#include <omp.h>
// SIMD 版本需要 NEON 指令集，且使用 OpenMP 进行多线程优化
#endif

#include "common.h"
#include "compute.h"

void zero_z() {
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            Z[i][j] = 0;
        }
    }
}

void compute_row_major_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_mkn() {
    // TODO: task 1 原则上来说这个是task1里面最快的
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int l = 0; l != k; ++l) {
            for (int j = 0; j != n; ++j) {            
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_kmn() {
    // TODO: task 1
    zero_z();
    for (int l = 0; l != k; ++l) {
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != n; ++j) {            
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_nmk() {
    // TODO: task 1
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int i = 0; i != m; ++i) {        
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_nkm() {
    // TODO: task 1
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int l = 0; l != k; ++l) {
            for (int i = 0; i != m; ++i) {              
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_knm() {
    // TODO: task 1
    zero_z();
    for (int l = 0; l != k; ++l) {    
        for (int j = 0; j != n; ++j) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_y_transpose_mnk() {
    // TODO: task 2 使用转置和两个一起加，比单纯转置要快一些
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * YP[j][l] ;
            }
        }
    }
}

void compute_row_major_mnkkmn_b32() {
    int B = 30; // 分块大小
    zero_z();

    // 外层三层循环
    for (int i_o = 0; i_o < m; i_o += B) {
        for (int j_o = 0; j_o < n; j_o += B) {
            for (int l_o = 0; l_o < k; l_o += B) {
                
                // 内层三层循环
                
                for (int i = i_o; i < i_o + B && i < m; i++) {                        
                    for (int l = l_o; l < l_o + B && l < k; l++) {
                        for (int j = j_o; j < j_o + B && j < n; j++) {
                        Z[i][j]+= X[i][l] * Y[l][j];//反正都存的下，转置和不转置是一
                        }
                    }
                }

            }
        }
    }
}

void compute_row_major_mnk_lu2() {
    // TODO: task 2
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int l = 0; l != k; l += 2) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j] + X[i][l + 1] * Y[l + 1][j];
            }
        }
    }
}



//接下来是一个分块，转置和寄存器平铺的最终版本。块大小128(经过尝试，这是表现最好的)，寄存器平铺4x4
void compute_optimized_task2() {
    zero_z();
    
    const int MM = m;
    const int NN = n;
    const int KK = k;
    const int b = 128; 

    for (int i = 0; i < MM; i += b) {
        int i_end = (i + b > MM) ? MM : i + b;
        for (int j = 0; j < NN; j += b) {
            int j_end = (j + b > NN) ? NN : j + b;
            
            int ii, jj;
            // 1. 主循环：只处理能够凑满 4x4 的整块
            for (ii = i; ii <= i_end - 4; ii += 4) {
                for (jj = j; jj <= j_end - 4; jj += 4) {
                    
                    uint64_t c00=0, c01=0, c02=0, c03=0;
                    uint64_t c10=0, c11=0, c12=0, c13=0;
                    uint64_t c20=0, c21=0, c22=0, c23=0;
                    uint64_t c30=0, c31=0, c32=0, c33=0;

                    for (int ll = 0; ll < KK; ll++) {
                        uint64_t x0 = X[ii+0][ll]; uint64_t x1 = X[ii+1][ll];
                        uint64_t x2 = X[ii+2][ll]; uint64_t x3 = X[ii+3][ll];
                        uint64_t y0 = YP[jj+0][ll]; uint64_t y1 = YP[jj+1][ll];
                        uint64_t y2 = YP[jj+2][ll]; uint64_t y3 = YP[jj+3][ll];

                        c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;
                        c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;
                        c20 += x2 * y0; c21 += x2 * y1; c22 += x2 * y2; c23 += x2 * y3;
                        c30 += x3 * y0; c31 += x3 * y1; c32 += x3 * y2; c33 += x3 * y3;
                    }

                    Z[ii+0][jj+0] = c00; Z[ii+0][jj+1] = c01; Z[ii+0][jj+2] = c02; Z[ii+0][jj+3] = c03;
                    Z[ii+1][jj+0] = c10; Z[ii+1][jj+1] = c11; Z[ii+1][jj+2] = c12; Z[ii+1][jj+3] = c13;
                    Z[ii+2][jj+0] = c20; Z[ii+2][jj+1] = c21; Z[ii+2][jj+2] = c22; Z[ii+2][jj+3] = c23;
                    Z[ii+3][jj+0] = c30; Z[ii+3][jj+1] = c31; Z[ii+3][jj+2] = c32; Z[ii+3][jj+3] = c33;
                }
                
                // 2. 扫尾逻辑 A：处理当前行中，剩余不足 4 列的部分
                for (; jj < j_end; jj++) {
                    for (int r = ii; r < ii + 4; r++) {
                        uint64_t sum = 0;
                        for (int ll = 0; ll < KK; ll++) sum += X[r][ll] * YP[jj][ll];
                        Z[r][jj] = sum;
                    }
                }
            }

            // 3. 扫尾逻辑 B：处理最后剩余不足 4 行的部分
            for (; ii < i_end; ii++) {
                for (int c_jj = j; c_jj < j_end; c_jj++) {
                    uint64_t sum = 0;
                    for (int ll = 0; ll < KK; ll++) sum += X[ii][ll] * YP[c_jj][ll];
                    Z[ii][c_jj] = sum;
                }
            }
        }
    }
}






//ai教的SIMD版本，使用NEON指令集进行优化，然后转4线程并行计算

#ifdef SIMD
static inline uint64x2_t vmul_u64_manual(uint64x2_t a, uint64x2_t b) {
    return vcombine_u64(
        vcreate_u64(vgetq_lane_u64(a, 0) * vgetq_lane_u64(b, 0)),
        vcreate_u64(vgetq_lane_u64(a, 1) * vgetq_lane_u64(b, 1))
    );
// 这个函数手动实现了 uint64x2_t 的乘法，因为 NEON 没有直接的 uint64x2_t 乘法指令
}
#endif

void compute_simd() {
#ifdef SIMD
    zero_z();
    const int MM = m;
    const int NN = n;
    const int KK = k;

    // 以 4x4 为一个小块进行跳转
    #pragma omp parallel for collapse(2) num_threads(4)
    //对for多线程计算，打开循环嵌套，指定线程数为4
    for (int i = 0; i < MM; i += 4) {
        for (int j = 0; j < NN; j += 4) {
            
            // 1. 使用 8 个寄存器存储 4x4 的结果块 (每个寄存器存 2 个 uint64)
            // row0_01 代表 Z[i][j] 和 Z[i][j+1]，以此类推
            uint64x2_t r0_01 = vdupq_n_u64(0), r0_23 = vdupq_n_u64(0);
            uint64x2_t r1_01 = vdupq_n_u64(0), r1_23 = vdupq_n_u64(0);
            uint64x2_t r2_01 = vdupq_n_u64(0), r2_23 = vdupq_n_u64(0);
            uint64x2_t r3_01 = vdupq_n_u64(0), r3_23 = vdupq_n_u64(0);

            for (int l = 0; l < KK; l++) {
                // 2. 加载 YP 的两组向量 (共 4 个 uint64: YP[j...j+3][l])
                // 假设 LOAD_Y_TRANSPOSE 保证了 YP[j][l] 和 YP[j+1][l] 连续
                uint64x2_t y01 = {YP[j][l], YP[j+1][l]};
                uint64x2_t y23 = {YP[j+2][l], YP[j+3][l]};

                // 3. 针对 X 的每一行进行广播并乘加
                // 处理第 i 行
                uint64x2_t x0 = vdupq_n_u64(X[i][l]);
                r0_01 = vaddq_u64(r0_01, vmul_u64_manual(x0, y01));
                r0_23 = vaddq_u64(r0_23, vmul_u64_manual(x0, y23));

                // 处理第 i+1 行
                uint64x2_t x1 = vdupq_n_u64(X[i+1][l]);
                r1_01 = vaddq_u64(r1_01, vmul_u64_manual(x1, y01));
                r1_23 = vaddq_u64(r1_23, vmul_u64_manual(x1, y23));

                // 处理第 i+2 行
                uint64x2_t x2 = vdupq_n_u64(X[i+2][l]);
                r2_01 = vaddq_u64(r2_01, vmul_u64_manual(x2, y01));
                r2_23 = vaddq_u64(r2_23, vmul_u64_manual(x2, y23));

                // 处理第 i+3 行
                uint64x2_t x3 = vdupq_n_u64(X[i+3][l]);
                r3_01 = vaddq_u64(r3_01, vmul_u64_manual(x3, y01));
                r3_23 = vaddq_u64(r3_23, vmul_u64_manual(x3, y23));
            }

            // 4. 写回 16 个结果
            vst1q_u64(&Z[i][j], r0_01);   vst1q_u64(&Z[i][j+2], r0_23);
            vst1q_u64(&Z[i+1][j], r1_01); vst1q_u64(&Z[i+1][j+2], r1_23);
            vst1q_u64(&Z[i+2][j], r2_01); vst1q_u64(&Z[i+2][j+2], r2_23);
            vst1q_u64(&Z[i+3][j], r3_01); vst1q_u64(&Z[i+3][j+2], r3_23);
        }
    }
#endif
}









uint64_t elapsed(const struct timespec start, const struct timespec end) {
    struct timespec result;
    result.tv_sec = end.tv_sec - start.tv_sec;
    result.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += SEC;
    }
    uint64_t res = result.tv_sec * SEC + result.tv_nsec;
    return res;

}

uint64_t compute() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    switch (COMPUTE_SELECT) {
        case COMPUTE_ROW_MAJOR_MNK:
            //printf("COMPUTE_ROW_MAJOR_MNK\n");
            compute_row_major_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MKN:
            //printf("COMPUTE_ROW_MAJOR_MKN\n");
            compute_row_major_mkn();
            break;
        case COMPUTE_ROW_MAJOR_KMN:
            //printf("COMPUTE_ROW_MAJOR_KMN\n");
            compute_row_major_kmn();
            break;
        case COMPUTE_ROW_MAJOR_NMK:
            //printf("COMPUTE_ROW_MAJOR_NMK\n");
            compute_row_major_nmk();
            break;
        case COMPUTE_ROW_MAJOR_NKM:
            //printf("COMPUTE_ROW_MAJOR_NKM\n");
            compute_row_major_nkm();
            break;
        case COMPUTE_ROW_MAJOR_KNM:
            //printf("COMPUTE_ROW_MAJOR_KNM\n");
            compute_row_major_knm();
            break;
        case COMPUTE_Y_TRANSPOSE_MNK:
            //printf("COMPUTE_Y_TRANSPOSE_MNK\n");
            compute_y_transpose_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B32:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B32\n");
            compute_row_major_mnkkmn_b32();
            break;
        case COMPUTE_ROW_MAJOR_MNK_LU2:
            //printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
            compute_row_major_mnk_lu2();
            break;
        case COMPUTE_SIMD:
            //printf("COMPUTE_SIMD\n");
            compute_simd();
            break;
        case COMPUTE_TASK2_FINAL:
            compute_optimized_task2();
            break;
        default:
            printf("Unreachable!");
            return 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed(start, end);
}
