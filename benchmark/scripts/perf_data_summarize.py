import os
import pandas as pd


def collect_data(file):

    data = []
    with open(file) as fp:
        lines = fp.readlines()
        num_lines = len(lines)
        for i in range(0, num_lines, 2):
            kernel_file = lines[i]
            perf_data = lines[i + 1]
            kernel = kernel_file.strip().split(r'/')[-1]
            time, throughput, bandwidth = perf_data.strip().split('    ')
            data.append([kernel, time[:-2], throughput[:-2], bandwidth[:-4]])

    df = pd.DataFrame(data, columns=['kernel', 'time', 'throughput', 'bandwidth'])
    df = df.sort_values(by='kernel')
    return df


def main():
    work_dir = '/home/sdp/liyang/triton/third_party/intel_xpu_backend/benchmark/'
    xpu_df = collect_data(os.path.join(work_dir, 'perf_data_xpu.txt'))
    cuda_df = collect_data(os.path.join(work_dir, 'perf_data_cuda.txt'))
    res_df = xpu_df.merge(cuda_df, on='kernel')
    print(res_df)
    res_df.to_csv('inductor_triton_kernel_perf.csv', index=False)


if __name__ == "__main__":
    main()
