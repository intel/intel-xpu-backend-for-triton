#
# Auto kernel extraction procedure
# 1. Auto run 160+ models(HF, torchbench, TIMM) in debug mode
#    outputs: kernel performance log, output_code.py
# 2. Got the name of top 1 xpu time triton kernel from model execution log,
#    find out kernel corresponding `output_code.py` file path
# 3. Run kernel extract script to extract 160+ kernels
#
import os
import json
import subprocess


def run_model(args):

    suite, datatype, mode, scenario, device, card, shape, num_shards, shard_id, model = args
    work_dir = '/home/sdp/liyang/private-pytorch/'
    command = f'bash xpu_run_batch.sh {suite} {datatype} {mode} {scenario} {device} {card} {shape} {num_shards} {shard_id} {model}'
    subprocess.run(command, shell=True, cwd=work_dir)

    return


def parse_table(suite, model, dtype) -> (str, str):
    base_dir = '/home/sdp/liyang/private-pytorch/'
    model_dir = os.path.join(base_dir, 'inductor_log', suite, model, dtype)
    table_path = os.path.join(model_dir, f'graph_table_{model}.txt')

    if not os.path.exists(table_path):
        return '', ''

    # Find the top1 xpu time cost triton kernel name
    kernel_name = ''
    with open(table_path) as fp:
        for line in fp.readlines():
            kernel_name = line.strip().split('  ')[0]
            if not kernel_name.startswith('XPU Triton kernel'):
                continue
            # clean kernel name:
            # i.e. `XPU Triton kernel:triton_poi_fused_view_8_0d1d2de` -> `triton_poi_fused_view_8`
            kernel_name = kernel_name.strip().split(':')[-1]
            kernel_patterns = kernel_name.split('_')
            kernel_name = '_'.join(kernel_patterns[:-1])
            break   # find the first kernel and jump out

    # Find the testing file contains the target kernel
    print(f'Run command: grep -nr "def {kernel_name}" {model_dir}')
    command = f'grep -nr "def {kernel_name}" {model_dir}'
    grep_output = subprocess.run(command, shell=True, capture_output=True)
    file_candidates = grep_output.stdout.decode().split('\n')
    output_file = ''
    for candidate in file_candidates:
        if not candidate:
            continue
        output_file = candidate.strip().split(':')[0]
        if output_file.endswith('output_code.py'):
            # Jump out when find output_code.py
            break

    return kernel_name, output_file


def extract_kernels(model, kernel, src_file):

    extractor = '/home/sdp/liyang/private-pytorch/torch/_inductor/tools/micro_kernel_extractor.py'
    out_file = f'/home/sdp/liyang/private-pytorch/torch/_inductor/tools/triton_kernels/{model}_{kernel}.py'
    subprocess.run(['python', extractor, kernel, src_file, out_file])
    return


def main():
    model_src = ['huggingface', 'timm', 'torchbench']
    datatype = 'amp_bf16'   # float32 / float16 / amp_bf16 / amp_fp16
    mode = 'training'       # training / inference
    scenario = 'performance'  # accuracy / performance
    device = 'xpu'
    card = '3'
    shape = 'static'        # static / dynamic
    num_shards = 1
    shard_id = 0

    model_count = 0
    failed_models = {key: [] for key in model_src}

    with open('/home/sdp/liyang/private-pytorch/torch/_inductor/tools/extract_failed_list.json') as fp:
        model_dic = json.load(fp)

    # for suite in model_src:
    #     model_dir = '/home/sdp/liyang/private-pytorch/benchmarks/dynamo/'
    #     model_list_file = f'{suite}_models_list.txt'
    #     with open(os.path.join(model_dir, model_list_file)) as fp:
    #         if suite == 'timm':
    #             model_list = [line.split(' ')[0] for line in fp.readlines() if line]
    #             suite_name = 'timm_models'
    #         else:
    #             model_list = [line.split(',')[0] for line in fp.readlines() if line]

    for suite, model_list in model_dic.items():
        if suite != 'timm':
            continue
        suite_name = suite if suite != 'timm' else "timm_models"

        work_dir = '/home/sdp/liyang/private-pytorch/'
        os.chdir(work_dir)
        for model in model_list:
            model_count += 1
            model_args = (suite_name, datatype, mode, scenario, device, card, shape, num_shards, shard_id, model)
            run_model(model_args)
            kernel, output_file = parse_table(suite_name, model, datatype)
            if not kernel or not output_file:
                failed_models[suite].append(model)
                continue
            extract_kernels(model, kernel, output_file)

    for suite in model_src:
        print(f'Failed model counts: {suite} - {len(failed_models[suite])} / {model_count}')
    print('failed_model_list: ', )
    with open('/home/sdp/liyang/private-pytorch/torch/_inductor/tools/extract_failed_list.json', '+a') as fp:
        json.dump(failed_models, fp)

if __name__ == '__main__':
    main()

    # parse_table('huggingface', 'AlbertForMaskedLM', 'amp_fp16')

    # kernel_name = 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_sum_19'
    # output_file = '/home/sdp/liyang/private-pytorch/inductor_log/huggingface/AlbertForMaskedLM/amp_fp16/4p/c4pwo2w7f6arzq2nzfsov3u64yaqopscc7bo4payvcbseypfyo67.debug/output_code.py'
    # extract_kernels('AlbertForMaskedLM', kernel_name, output_file)
