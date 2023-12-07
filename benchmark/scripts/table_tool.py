import os
import re
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_kernel_name(kernel_name: str):
    if kernel_name.lower() == 'name':
        return kernel_name

    if 'triton_' not in kernel_name:
        return ''

    # Remove XPU gen prefix XPUTriton:
    if kernel_name.startswith('XPU'):
        kernel_name = kernel_name.split(':')[-1]

    # Remove kernel name suffix 0d1d...nd, then kernel name match inductor-gen triton kernel
    kernel_name = '_'.join(kernel_name.split('_')[:-1])
    return kernel_name


def parse_data(value: str):
    # convert all float time to ms unit
    if value.endswith('ms'):
        return float(value[:-2])
    if value.endswith('us'):
        return float(value[:-2]) / 1000
    if value.endswith('s'):
        return float(value[:-1]) * 1000
    if value.endswith('%'):
        return float(value[:-1])
    if value.isdigit():
        return int(value)
    return value


def parse_table(table_file) -> dict:
    res = {}
    headers = []
    with open(table_file) as fp:
        lines = fp.readlines()
    for row in lines:
        items = [i.strip() for i in row.split('  ') if i.strip()]
        kernel_name = parse_kernel_name(items[0])
        if not kernel_name:
            continue
        if kernel_name.lower() == 'name':
            headers = [kernel_name] + items[7:]
        else:
            res[kernel_name] = [parse_data(i) for i in items[7:]]
    return res, headers


def _mask_kernel_name(kernel_name, table):
    # replace convert_element_type_{id} pattern with 'cet'
    new_name = re.sub(r'convert_element_type_\d+_', '', kernel_name)
    new_name = re.sub(r'convert_element_type_', '', new_name)
    new_name += f'_{table[kernel_name][-1]}'
    return new_name


def merge_table(xpu_table, cuda_table) -> dict:
    res = {}
    # merge that kernel_name fully match
    for key in list(xpu_table.keys()):
        cuda_value = cuda_table.pop(key, None)
        if not cuda_value:
            continue
        xpu_value = xpu_table.pop(key)
        res[key] = xpu_value + [''] + cuda_value

    # merge kernel_name that almost same
    # The node convert_element_type id is not always same on both xpu and cuda side, we need to mask this pattern out
    name_mask = {}
    for cuda_key in cuda_table.keys():
        name_mask[_mask_kernel_name(cuda_key, cuda_table)] = cuda_key
    for xpu_key, xpu_value in xpu_table.items():
        cuda_key = name_mask.get(_mask_kernel_name(xpu_key, xpu_table), None)
        if cuda_key:
            cuda_value = cuda_table.pop(cuda_key)
            res[xpu_key] = xpu_value + [cuda_key] + cuda_value
        else:
            res[xpu_key] = xpu_value + [0] * (len(xpu_value) + 1)

    # merge that cuda kernel_name cannot match
    for cuda_key, cuda_value in cuda_table.items():
        res[cuda_key] = [0] * len(cuda_value) + [cuda_key] + cuda_value

    return res


def analysis_and_save(headers: list, table: dict, file_path: str):
    data = []
    for key, value in table.items():
        data.append([key] + value)
    df = pd.DataFrame(data, columns=headers)
    df['XPU_name'] = np.where(df['XPU total'] < 0.00001, '', df['XPU_name'])

    columns_to_sum = ['Self XPU %', 'XPU total', 'Self CUDA %', 'CUDA total']
    sums = df[columns_to_sum].sum()
    sums['XPU_name'] = 'SUM'

    df = pd.concat([df, pd.DataFrame([sums], columns=headers)])
    df = df.fillna('')

    df = df.sort_values(by='XPU total', ascending=False)
    df['speed_up'] = (df['CUDA total'].astype(float) /
                      df['XPU total'].astype(float)).fillna(0)

    log.info('\n')
    log.info(df)
    df.to_csv(file_path, index=False)


def main():
    table_dir = '/data/liyang/private-pytorch/torch/_inductor/tools/kernel_tables'
    model_list = ['DebertaForMaskedLM', 'DebertaV2ForQuestionAnswering', 'DistilBertForMaskedLM', 'DistilBertForQuestionAnswering', 'MobileBertForMaskedLM',
                  'MobileBertForQuestionAnswering', 'PegasusForConditionalGeneration', 'Speech2Text2ForCausalLM', 'T5ForConditionalGeneration', 'T5Small']
    for model in model_list:
        log.info(f'Process Model {model}')
        xpu_table, xpu_headers = parse_table(
            os.path.join(table_dir, f'{model}_graph_table_xpu.txt'))
        cuda_table, cuda_headers = parse_table(
            os.path.join(table_dir, f'{model}_graph_table_cuda.txt'))
        full_table = merge_table(xpu_table, cuda_table)
        xpu_headers[0] = 'XPU_name'
        cuda_headers[0] = 'CUDA_name'
        headers = xpu_headers + cuda_headers
        output_file = os.path.join(table_dir, f'kernels_{model}.csv')
        log.info(f'Save comparison table to {output_file}')
        analysis_and_save(headers, full_table, output_file)


if __name__ == '__main__':
    main()
