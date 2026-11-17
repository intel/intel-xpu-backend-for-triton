"""Print the plot_name of every @perf_report benchmark in the given file(s), one per line.

Parses the source with ast so it needs neither torch nor triton.
"""
import ast
import sys


def _decorator_plot_name(deco):
    # @perf_report(Benchmark(..., plot_name='x')) or a wrapper such as @_row_limit_perf_report('x').
    for node in ast.walk(deco):
        if isinstance(node, ast.keyword) and node.arg == 'plot_name' and isinstance(node.value, ast.Constant):
            return node.value.value
    for a in deco.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return None


def plot_names(path):
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('benchmark_'):
            names += [n for d in node.decorator_list if isinstance(d, ast.Call) and (n := _decorator_plot_name(d))]
    return names


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print(*plot_names(arg), sep='\n')
