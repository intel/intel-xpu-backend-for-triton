# Run this script with command
# $ python kernel_collection.py huggingface amp_bf16 inference performance /path/to/pytorch
#
# Auto kernel extraction procedure
# 1. Auto run pytorch inductor models(HF, torchbench, TIMM) in debug mode
#    outputs: kernel performance log, output_code.py
# 2. Got the name of top 1 xpu time triton kernel from model execution log,
#    find out kernel corresponding `output_code.py` file path
# 3. Run kernel extract script to extract kernels
#
import ast
import json
import logging
import os
import re
import subprocess
from typing import Tuple

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_model(pytorch_dir, args):

    suite, datatype, mode, scenario, device, card, shape, num_shards, shard_id, model = args
    work_dir = pytorch_dir
    command = f'bash xpu_run_batch.sh {suite} {datatype} {mode} {scenario} {device} {card} {shape} {num_shards} {shard_id} {model}'
    subprocess.run(command, shell=True, cwd=work_dir)

    return


def parse_table(pytorch_dir, suite, model, dtype) -> Tuple[str, str]:
    base_dir = pytorch_dir
    model_dir = os.path.join(base_dir, 'inductor_log', suite, model, dtype)
    table_path = os.path.join(model_dir, f'graph_table_{model}.txt')

    if not os.path.exists(table_path):
        return '', ''

    # Find the top1 xpu time cost triton kernel name
    kernel_name = ''
    with open(table_path) as fp:
        for line in fp.readlines():
            kernel_name = line.strip().split('  ')[0]
            if not kernel_name.startswith('triton_'):
                continue
            if re.search(r"(\dd)+$", kernel_name):
                # clean kernel name:
                # i.e. `triton_poi_fused_view_8_0d1d2d` -> `triton_poi_fused_view_8`
                tokens = kernel_name.split('_')
                kernel_name = '_'.join(tokens[:-1])
            break   # find the first kernel and jump out

    # Find the testing file contains the target kernel
    log.info(f'Run command: grep -nr "def {kernel_name}" {model_dir}')
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


def extract_kernel_code(source, kernel_name):

    src_ast = ast.parse(source)
    kernel_code = ''
    for node in src_ast.body:
        if type(node) is not ast.Assign:
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            is_fusion = node.value.func.value.id == 'async_compile'
            match_target = node.targets[0].id == kernel_name
            if is_fusion and match_target:
                kernel_code = node.value.args[1].value
                break
    return kernel_code


def extract_kernels(kernel, src_file, tgt_file):

    with open(src_file) as fp:
        source = fp.read()

    log.info(f'Extracting kernel {kernel} from {src_file}...')
    kernel_code = extract_kernel_code(source, kernel)
    with open(tgt_file, mode='w') as out_fp:
        out_fp.write(kernel_code)
    # os.system(f'python -m autopep8 -i {out_file}')
    log.info(f'Output kernel has been saved to {tgt_file}')

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=str, help="one of huggingface / timm / torchbench")
    parser.add_argument('datatype', type=str, help="one of float32 / float16 / amp_bf16 / amp_fp16")
    parser.add_argument('mode', type=str, help="one of training / inference")
    parser.add_argument('scenario', type=str, help="one of accuracy / performance")
    parser.add_argument('pytorch_dir', type=str)
    args = parser.parse_args()

    pytorch_dir = args.pytorch_dir
    suite = args.suite
    datatype = args.datatype
    mode = args.mode
    scenario = args.scenario
    device = 'xpu'
    card = '0'
    shape = 'static'        # static / dynamic
    num_shards = 1
    shard_id = 0

    model_count = 0
    failed_models = {suite: []}
    script_dir = os.path.dirname(__file__)
    kernel_dir = os.path.join(script_dir, "extracted_kernels")
    if not os.path.exists(kernel_dir):
        os.makedirs(kernel_dir)

    # Ger models name from models lists
    model_dir = f'{pytorch_dir}/benchmarks/dynamo/'
    model_list_file = f'{suite}_models_list.txt'
    with open(os.path.join(model_dir, model_list_file)) as fp:
        if suite == 'timm':
            suite_name = 'timm_models'
            model_list = [line.split(' ')[0] for line in fp.readlines() if line]
        else:
            model_list = [line.split(',')[0] for line in fp.readlines() if line]

    suite_name = suite if suite != 'timm' else "timm_models"
    # os.chdir(work_dir)
    for model in model_list:
        model_count += 1
        model_args = (suite_name, datatype, mode, scenario, device, card, shape, num_shards, shard_id, model)
        # Model runing, after that we can get
        #    1. triton kernels in cache folder;
        #    2. kernel performance summary table.
        run_model(pytorch_dir, model_args)

        # Analysis kernel performance summary table to get
        #    1. Top 1 XPU time cost kernel name
        #    2. The Source code file that contain the target kernel
        kernel, src_file = parse_table(pytorch_dir, suite_name, model, datatype)
        if not kernel or not src_file:
            failed_models[suite].append(model)
            continue

        # Extract target kernel from Source code file to a seperate file
        tgt_file = os.path.join(kernel_dir, f"{model}_{kernel}.py")
        extract_kernels(kernel, src_file, tgt_file)

    log.info(f"Failed model counts: {suite} - {len(failed_models[suite])} / {model_count}")
    failing_models = os.path.join(script_dir, "failing_models.json")
    log.info(f"Save failed_model_list to {failing_models}")
    with open(failing_models, '+a') as fp:
        json.dump(failed_models, fp)


if __name__ == '__main__':
    main()
