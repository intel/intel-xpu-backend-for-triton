import os
import subprocess
import pandas as pd


def test_kernel(kernel, kernel_dir):

    kernel_file = os.path.join(kernel_dir, kernel)
    command = f'python {kernel_file}'
    print(command)
    output = subprocess.run(['python', kernel_file], shell=True, capture_output=True, cwd=kernel_dir)
    time, throughput, bandwidth = output.stdout.decode().strip().split('    ')
    time = time[:-2]
    throughput = throughput[:-2]
    bandwidth = bandwidth[:-3]
    print([time, throughput, bandwidth])
    return [time, throughput, bandwidth]


def main():
    kernel_dir = '/home/guizili/liyang/triton/third_party/intel_xpu_backend/benchmark/inductor_kernels'

    data = []
    kernel_list = sorted(list(os.listdir(kernel_dir)))
    for kernel in kernel_list:
        if not kernel.endswith('.py'):
            continue
        perf_data = test_kernel(kernel, kernel_dir)
        data.append([kernel, *perf_data])
    header = ['kernel', 'time', 'throughput', 'bandwidth']
    perf_df = pd.DataFrame(data, columns=header)

    output_file = '/home/guizili/liyang/triton/third_party/intel_xpu_backend/benchmark/inductor_kernels/cuda_perf_data.csv'
    perf_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
