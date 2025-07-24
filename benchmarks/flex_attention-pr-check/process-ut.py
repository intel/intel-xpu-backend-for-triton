import csv
import os
import glob
import re
import sys
from collections import defaultdict, Counter

def remove_numbers_and_shapes(text):
    # 去除所有数字，包括科学计数法（如 1e5, 3.14e-10）
    text = re.sub(r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b', '', text)
    # 去除 tensor 形状
    # text = re.sub(r'\([^\)]+\)', '()', text)
    return text

def extract_log_snippets(log_dir):
    pattern = "python test/"
    error_exclusion_patterns = [
        r"^To execute this test, run the following",
        r"^This message can be suppressed"
    ]
    output_dir = os.path.join(log_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    error_to_command_global = defaultdict(set)
    
    for log_file in glob.glob(os.path.join(log_dir, "*.log")):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        extracted_snippets = []
        error_counter = Counter()
        error_to_command = defaultdict(set)
        
        for i, line in enumerate(lines):
            if pattern in line:
                command = line.strip()
                for j in range(i - 1, -1, -1):  # 从当前行向上查找
                    if any(re.match(pat, lines[j]) for pat in error_exclusion_patterns):
                        continue  # 跳过不相关的信息
                    match = re.search(r"\b(\w+Error)\b", lines[j])  # 查找错误信息
                    if match:
                        cleaned_error = remove_numbers_and_shapes(lines[j].strip())
                        error_counter[cleaned_error] += 1
                        extracted_snippets.extend(lines[max(0, j - 3): min(len(lines), i + 3)] + ["\n"])
                        # 记录错误和对应的 `python test/...` 命令

                        error_to_command[cleaned_error].add(command)
                        error_to_command_global[cleaned_error].add(command)
                        break  # 找到第一个错误行后就停止向上查找
        
        if extracted_snippets:
            output_file = os.path.join(output_dir, os.path.basename(log_file) + ".snippet")
            with open(output_file, "w", encoding="utf-8") as out_f:
                out_f.writelines(extracted_snippets)
        
        # 记录当前日志文件的错误统计
        if error_counter:
            stats_file = os.path.join(output_dir, os.path.basename(log_file) + ".error_stats.csv")
            with open(stats_file, "w", newline='', encoding="utf-8") as stats_f:
                writer = csv.writer(stats_f)
                writer.writerow(["Error Type", "Count"])
                for error, count in error_counter.most_common():
                    writer.writerow([error, count])
                total_errors = sum(error_counter.values())
                writer.writerow(["Total Errors", total_errors])

        if error_to_command:
            command_file_txt = os.path.join(output_dir, os.path.basename(log_file) + ".error_to_command.txt")
            with open(command_file_txt, "w", encoding="utf-8") as txt_f:
                for error, commands in sorted(error_to_command.items()):
                    txt_f.write(f"Error: {error}\n")
                    for cmd in sorted(commands):
                        txt_f.write(f"  - {cmd}\n")
                    txt_f.write("\n")
                
    # 输出错误与命令的映射（CSV 文件）
    if error_to_command_global:
        command_file_csv = os.path.join(output_dir, "error_to_command.csv")
        with open(command_file_csv, "w", newline='', encoding="utf-8") as cmd_f:
            writer = csv.writer(cmd_f)
            writer.writerow(["Error Type", "Python Test Command"])
            for error, commands in sorted(error_to_command_global.items()):
                for cmd in sorted(commands):  # 按命令排序
                    writer.writerow([error, cmd])

        # 输出错误与命令的映射（文本文件）
        command_file_txt = os.path.join(output_dir, "error_to_command.txt")
        with open(command_file_txt, "w", encoding="utf-8") as txt_f:
            for error, commands in sorted(error_to_command_global.items()):
                txt_f.write(f"Error: {error}\n")
                for cmd in sorted(commands):  # 按命令排序
                    txt_f.write(f"  - {cmd}\n")
                txt_f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_directory = sys.argv[1]
    else:
        log_directory = "20250331-053630"
    print(f"The log_dir argument is: {log_directory}")
    extract_log_snippets(log_directory)
