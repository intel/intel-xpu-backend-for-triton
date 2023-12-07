import ast
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# trace back node args to their init node recursively
def trace_back_node(node: ast.Assign, candidates: dict, args: list, node_trace: dict):
    # TODO: record visited args to simplify args list
    lhs = node.targets[0].id
    rhs = node.value
    if lhs in node_trace or lhs in args:
        return
    if isinstance(rhs, ast.Name):
        node_trace[lhs] = node
        trace_back_node(candidates[rhs.id], candidates, args, node_trace)
    elif rhs.func.id in ['empty_strided', 'rand_strided']:
        node_trace[lhs] = node
        return
    else:
        node_trace[lhs] = node
        for buffer in rhs.args:
            if isinstance(buffer, ast.Name):
                trace_back_node(candidates[buffer.id], candidates, args, node_trace)
        return


def clean_caller_func(call_node, kernel_name):
    # Assume Call func has 3 nodes: Assign, Expr, With
    args_assign_node, arg_clean_node, with_node = call_node.body
    args = [item.id for item in args_assign_node.targets[0].elts]

    kernel_nodes = []
    new_body = []
    deletes = {}
    candidates = {}
    return_node = None

    # traverse the with body, categorizing node by type
    for node in with_node.body:
        if isinstance(node, ast.Expr):
            if node.value.func.attr == 'run':
                if node.value.func.value.id == kernel_name:
                    kernel_nodes.append(node)
            elif node.value.func.attr == 'set_device':
                new_body.append(node)
        elif isinstance(node, ast.Assign):
            target = node.targets[0].id
            # Variable assignment
            if isinstance(node.value, ast.Name):
                candidates[target] = node
            # Func call assignment
            elif node.value.func.id == 'get_xpu_stream':
                new_body.append(node)
            else:
                candidates[target] = node
        elif isinstance(node, ast.Delete):
            deletes[node.targets[0].id] = node
        elif isinstance(node, ast.Return):
            return_node = node
        else:
            new_body.append(node)

    # trace back kernel args, remove non-dependent nodes
    node_trace = {}
    for node in kernel_nodes:
        kernel_args = [buff for buff in node.value.args if isinstance(buff, ast.Name)]
        for kernel_arg in kernel_args:
            arg_name = kernel_arg.id
            trace_back_node(candidates[arg_name], candidates, args, node_trace)

    # # clean used args
    # arg_elts = []
    # for arg_name in args:
    #     if arg_name in node_trace:
    #         arg_elts.append(node_trace[arg_name])
    # args_assign_node.targets[0].elts = arg_elts

    # clean used return
    ret_elts = []
    for ret_val in return_node.value.elts:
        if not isinstance(ret_val, ast.Name):
            continue
        if ret_val.id in node_trace:
            ret_elts.append(ret_val)
    return_node.value.elts = ret_elts
    new_body.append(return_node)

    del_list = []
    for del_id, del_node in deletes.items():
        if del_id in node_trace:
            del_list.append(del_node)

    new_body.extend(node_trace.values())
    new_body.extend(del_list)
    new_body.extend(kernel_nodes)
    new_body.sort(key=lambda x: x.lineno)
    # TODO: check if this change can work
    with_node.body = new_body

    return call_node


def clean_main(node):
    # TODO: clean unused
    return node


def extract_kernel_case(source, kernel_name):

    kernel_root = ast.Module()
    setattr(kernel_root, 'body', [])
    setattr(kernel_root, 'type_ignores', [])
    src_ast = ast.parse(source)

    kernel_code = ''
    for node in src_ast.body:
        if type(node) is ast.Assign:
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                is_fusion = node.value.func.value.id == 'async_compile'
                match_target = node.targets[0].id == kernel_name
                if is_fusion and match_target:
                    kernel_code = node.value.args[1].value
                    break
        # elif type(node) is ast.FunctionDef:
        #     if node.name == 'call':
        #         node = clean_caller_func(node, kernel_name)
        # elif type(node) is ast.If:
        #     if '__main__' in ast.unparse(node.test):
        #         node = clean_main(node)
        # kernel_root.body.append(node)
    # kernel_code = ast.unparse(kernel_root)
    return kernel_code


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('kernel', type=str,
                        help='the kernel name to extract')
    parser.add_argument('in_file', type=str,
                        help='the source file to be extracted')
    parser.add_argument('out_file', type=str,
                        help='the output file path to be stored into')
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        raise FileNotFoundError()

    # work_dir = '/home/sdp/liyang/private-pytorch/inductor_log/huggingface/DistilBertForMaskedLM/amp/6k/c6kvxugfg7gnmum7tyexqldiavagcgmasij3yp3jgvqbr2wgcb7j.debug/'
    # src_file = 'output_code.py'
    # kernel = 'triton_fused_sum_12_view_123_7'
    # in_file = os.path.join(work_dir, src_file)
    # with open(in_file) as in_file:
    #     source = in_file.read()

    in_file = args.in_file
    kernel = args.kernel
    out_file = args.out_file
    # work_dir = os.path.dirname(os.path.abspath(in_file))
    # out_file = os.path.join(work_dir, f'test_{kernel}.py')

    with open(in_file) as in_fp:
        source = in_fp.read()

    log.info(f'Extracting kernel {kernel} from {in_file}...')
    output = extract_kernel_case(source, kernel)
    with open(out_file, mode='w') as out_fp:
        out_fp.write(output)
    # os.system(f'python -m autopep8 -i {out_file}')
    log.info(f'Output kernel has been saved to {out_file}')

    log.info('Done.')

if __name__ == '__main__':
    main()
