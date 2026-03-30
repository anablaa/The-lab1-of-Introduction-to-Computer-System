import matplotlib.pyplot as plt
from easypyplot.barchart import draw
from easypyplot.pdf import plot_open
import numpy as np

num_tasks = 3

for i in range(num_tasks):
    log_file = './tasks{}_run.log'.format(i+1)
    
    # 1. 提取所有包含 "Machine readable" 的数字
    perf_values = []
    with open(log_file, 'r') as f:
        for line in f:
            if "Machine readable" in line:
                # 提取行末的数字
                val = int(line.strip().split(' ')[-1])
                perf_values.append(val)
    
    if not perf_values:
        print(f"Warning: No data found in {log_file}")
        continue

    # 2. 定义每组实验应该有的算法数量 (即 group_names 的长度)
    # Task 1: 6个方向; Task 2: 4个优化; Task 3: 4个SIMD组合
    group_configs = {
        1: ["mnk", "mkn", "kmn", "nmk", "nkm", "knm"],
        2: ['mnk', 't_mnk', 'mnkkmn_b32', 'mnk_lu2','task2_final'],
        3: ['mnk', 'simd', 'o3', 'simd-o3']
    }
    
    current_groups = group_configs.get(i+1)
    num_groups = len(current_groups)
    
    # 3. 计算一共有多少个测试文件 (Entries)
    # perf_values 的总数应该是 num_groups * num_entries
    num_entries = len(perf_values) // num_groups
    
    if len(perf_values) % num_groups != 0:
        print(f"Error: Data in {log_file} does not match group size {num_groups}")
        continue

    # 4. 重组数据为矩阵 (Row: Algorithm, Column: Test Case)
    # 先 reshape 成 (num_entries, num_groups)，然后转置
    data_raw = np.array(perf_values).reshape(num_entries, num_groups)
    data_matrix = data_raw.T # 现在形状是 (num_groups, num_entries)
    
    # 5. 计算 Speedup (以每列的第一个元素为基准)
    # original 是每一列的第一个元素 (Baseline)
    original = data_matrix[0, :] 
    data_speedup = original / data_matrix

    # 6. 动态生成 Entry Names (1.txt, 2.txt...)
    current_entry_names = ["{}.txt".format(n+1) for n in range(num_entries)]

    # 7. 绘图
    output_pdf = "lab_{}".format(i+1)
    with plot_open(output_pdf) as fig:
        ax = fig.gca()
        draw(ax, data_speedup, breakdown=False,
             group_names = current_groups,
             entry_names = current_entry_names)
        
        ax.set_xlabel('Different implementations of compute kernel')
        ax.set_ylabel('Speedup')
        ax.set_title('Task {}'.format(i+1))
    
    print(f"Successfully generated {output_pdf}.pdf")