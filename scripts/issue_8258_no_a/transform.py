"""Strict, CPU-only removal of A-owned LLVM 2D prefetch calls."""
import hashlib
import re

VALUE = r'%[-a-zA-Z$._0-9]+'
PREFETCHES = [
    re.compile(r'\bcall\s+(?:spir_func\s+)?void\s+@llvm\.genx\.GenISA\.LSC2DBlockPrefetch\.[\w.]+\(i64\s+('+VALUE+r')\s*,'),
    re.compile(r'\bcall\s+(?:spir_func\s+)?void\s+@_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i'
               r'\((?:i32\s+[^,\n]+,\s*){4}ptr\s+addrspace\(1\)\s+[^,%\n]*('+VALUE+r')\s*,'),
]


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def inspect(source):
    headers = re.findall(r'^define\b[^\n]*\bspir_kernel\s+void\s+@([^\s(]+)\((.*)\)[^{\n]*\{',source,re.M)
    if len(headers)!=1:
        raise ValueError('Expected exactly one LLVM spir_kernel definition')
    name,arguments = headers[0]
    # Kernel inputs in this study are three pointer arguments; saved CPU
    # fixtures can additionally contain runtime stride arguments.
    params = re.findall(r'(?:^|,)\s*ptr(?:\s+addrspace\(\d+\))?[^,]*?('+VALUE+r')(?=\s*(?:,|$))',arguments)
    if len(params)<3:
        raise ValueError(f'Expected A/B/C pointer arguments; got {params}')
    inputs = dict(zip(params[:3],['A','B','C']))
    inputs.update({p:'implicit_pointer' for p in params[3:]})
    definitions = dict(re.findall(r'^\s*('+VALUE+r')\s*=\s*(.*)$',source,re.M))

    def roots(value,seen=frozenset()):
        if value in inputs:
            return {inputs[value]}
        if value in seen:
            raise ValueError('Cyclic prefetch base provenance')
        body = definitions.get(value,'').split('!dbg')[0]
        operation = body.split()[0] if body else ''
        if operation not in {'ptrtoint','inttoptr','bitcast','addrspacecast','and','freeze'}:
            raise ValueError(f'Unsupported prefetch base provenance: {value} = {body}')
        operands = re.findall(VALUE,body)
        if not operands:
            raise ValueError(f'Unresolved prefetch base {value}')
        return set().union(*(roots(v,seen|{value}) for v in operands))

    calls = []
    for line_number,line in enumerate(source.splitlines(keepends=True),1):
        if not re.search(r'\bcall\b',line) or 'prefetch' not in line.lower():
            continue
        match = next((m for pattern in PREFETCHES if (m:=pattern.search(line))),None)
        if match is None:
            raise ValueError(f'Unsupported prefetch call at line {line_number}: {line.strip()}')
        owners = roots(match[1])
        if owners not in ({'A'},{'B'}):
            raise ValueError(f'Ambiguous or unexpected prefetch owner at line {line_number}: {owners}')
        calls.append({'line':line_number,'owner':next(iter(owners)),'base':match[1],'text':line})
    return {'name':name,'pointer_arguments':params,'counts':{p:sum(c['owner']==p for c in calls) for p in ['A','B']},'calls':calls}


def remove_a(source):
    before = inspect(source)
    removed = {c['line'] for c in before['calls'] if c['owner']=='A'}
    result = ''.join(line for i,line in enumerate(source.splitlines(keepends=True),1) if i not in removed)
    after = inspect(result)
    if after['counts']['A'] or before['counts']['B']!=after['counts']['B']:
        raise ValueError('Prefetch transformation changed the wrong operand')
    if [c['text'] for c in before['calls'] if c['owner']=='B']!=[c['text'] for c in after['calls']]:
        raise ValueError('B prefetch changed')
    return result,{'before':before,'after':after,'removed_lines':sorted(removed),
                   'no_op':not removed,'original_llir_sha256':sha(source),'transformed_llir_sha256':sha(result)}
