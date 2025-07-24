import re
import csv
import os

def parse_log_file(log_content):
    
    # 针对每个块内部的信息提取
    ut_name_pattern = re.compile(r'python .*\.py\s(Test.*?)\s---')
    actual_time_pattern = re.compile(r'Ran \d+ test in ([\d.]+)s')
    exit_status_pattern = re.compile(r'Exit Status: (\d+)')
    total_time_pattern = re.compile(r'\(Took ([\d.]+) seconds\)')

    parsed_data = {}
    skipped_count = 0

    # 使用 split 比 finditer 更简单，因为每个UT块都由'--- Running:'明确分隔
    test_blocks = log_content.split('--- Running:')[1:] # 第一个元素是文件头，忽略

    for i, block in enumerate(test_blocks):
        # 补全block，以便能匹配 --- Finished ---
        full_block_for_match = f"--- Running: {block}"
        
        if "OK\n" not in full_block_for_match:
            skipped_count += 1
            continue  # 跳过这个不成功的UT块

        # 1. 提取UT名字
        ut_name_match = ut_name_pattern.search(full_block_for_match)
        if not ut_name_match:
            print(f"Warning: Could not find UT name in block {i+1}")
            continue
        ut_name = ut_name_match.group(1).strip()
        
        # 2. 提取UT实际耗时
        actual_time_match = actual_time_pattern.search(block)
        actual_time = float(actual_time_match.group(1)) if actual_time_match else None

        # 3. 提取退出代码
        exit_status_match = exit_status_pattern.search(block)
        exit_status = int(exit_status_match.group(1)) if exit_status_match else None

        # 4. 提取总体耗时
        total_time_match = total_time_pattern.search(block)
        total_time = float(total_time_match.group(1)) if total_time_match else None

        # 创建一个标准化的键，用于跨文件匹配 (不区分大小写地移除 'cuda' 和 'xpu')
        normalized_key = re.sub(r'cuda|xpu', '', ut_name, flags=re.IGNORECASE)

        parsed_data[normalized_key] = {
            'name': ut_name,
            'actual_time_s': actual_time,
            'total_time_s': total_time,
            'exit_code': exit_status
        }

    return parsed_data, skipped_count

def main():
    """
    主函数，读取、解析、比对日志并生成CSV报告。
    """
    # --- 请在这里配置你的文件路径 ---
    cuda_log_file = '/workspace1/xingyuan/20250213-flexatt-enable/1133-ut-test/cuda-full-scale/20250708-222531-weekly/test_flex_attention.result.log'
    xpu_log_file = '/workspace1/xingyuan/20250213-flexatt-enable/1133-ut-test/20250704-080218-weekly/test_flex_attention.result.log'
    output_csv_file = '/workspace1/xingyuan/20250213-flexatt-enable/1133-ut-test/20250704-080218-weekly/flex_attention.csv'
    # ------------------------------------

    # 检查输入文件是否存在
    if not os.path.exists(cuda_log_file):
        print(f"Error: CUDA log file not found at '{cuda_log_file}'")
        return
    if not os.path.exists(xpu_log_file):
        print(f"Error: XPU log file not found at '{xpu_log_file}'")
        return

    print("Reading log files...")
    with open(cuda_log_file, 'r', encoding='utf-8') as f:
        cuda_content = f.read()
    with open(xpu_log_file, 'r', encoding='utf-8') as f:
        xpu_content = f.read()

    print("Parsing CUDA log...")
    cuda_results, cuda_skipped = parse_log_file(cuda_content)
    print(f"Found {len(cuda_results)} successful UTs in CUDA log (skipped {cuda_skipped} non-OK tests).")

    print("Parsing XPU log...")
    xpu_results, xpu_skipped = parse_log_file(xpu_content)
    print(f"Found {len(xpu_results)} successful UTs in XPU log (skipped {xpu_skipped} non-OK tests).")

    # 合并所有唯一的UT（基于标准化名称）
    all_normalized_keys = sorted(list(set(cuda_results.keys()) | set(xpu_results.keys())))
    print(f"Total unique UTs to compare: {len(all_normalized_keys)}")

    # 准备写入CSV文件
    header = [
        'CUDA UT Name', 'CUDA Actual Time (s)', 'CUDA Total Time (s)', 'CUDA Exit Code',
        'XPU UT Name', 'XPU Actual Time (s)', 'XPU Total Time (s)', 'XPU Exit Code',
        'Actual Time Ratio (CUDA/XPU)', 'Total Time Ratio (CUDA/XPU)'
    ]
    
    rows_to_write = []

    for key in all_normalized_keys:
        cuda_data = cuda_results.get(key, {}) # 如果key不存在，返回空字典
        xpu_data = xpu_results.get(key, {})

        c_actual_time = cuda_data.get('actual_time_s')
        x_actual_time = xpu_data.get('actual_time_s')
        c_total_time = cuda_data.get('total_time_s')
        x_total_time = xpu_data.get('total_time_s')
        
        # 计算比率
        actual_time_ratio = '' # 默认为空
        if c_actual_time is not None and x_actual_time is not None and x_actual_time > 0:
            try:
                actual_time_ratio = f"{c_actual_time / x_actual_time:.4f}"
            except ZeroDivisionError:
                actual_time_ratio = 'N/A (XPU time is 0)'
        total_time_ratio = '' # 默认为空
        if c_total_time is not None and x_total_time is not None and x_total_time > 0:
            try:
                total_time_ratio = f"{c_total_time / x_total_time:.4f}"
            except ZeroDivisionError:
                total_time_ratio = 'N/A (XPU time is 0)'
        
        row = [
            cuda_data.get('name', ''),
            c_actual_time if c_actual_time is not None else '',
            cuda_data.get('total_time_s', ''),
            cuda_data.get('exit_code', ''),
            xpu_data.get('name', ''),
            x_actual_time if x_actual_time is not None else '',
            xpu_data.get('total_time_s', ''),
            xpu_data.get('exit_code', ''),
            actual_time_ratio,
            total_time_ratio,
        ]
        rows_to_write.append(row)

    print(f"Writing report to '{output_csv_file}'...")
    with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows_to_write)

    print("Done.")

if __name__ == '__main__':
    main()