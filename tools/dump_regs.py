#!/usr/bin/env python3
"""Debug helper: dump the final register state of both the original inline-PTX
test.cu and the generated pure_cuda.cu for tid==0, then diff them.

Usage: python3 tools/dump_regs.py <bug_report_dir> [seed]
"""
import re
import sys
import os
import subprocess
import struct
import hashlib

RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
NVCC = "/usr/local/cuda-12.6/bin/nvcc"

def parse_regs(path):
    src = open(path).read()
    m = re.search(r'asm volatile\((.*?)\s*:\s*"=l"\(o0\)', src, re.S)
    body = m.group(1)
    instrs = re.findall(r'"(.*?)"', body)
    regs = {'xr': 10, 'xd': 6, 'xp': 4}
    for i in instrs:
        i = i.replace('\\n\\t', '\n').replace('\\n', '\n')
        for part in i.split(';'):
            part = part.strip().replace('%%', '%')
            mm = re.match(r'\.reg\s+\.\w+\s+%(\w+)<(\d+)>', part)
            if mm:
                regs[mm.group(1)] = max(regs.get(mm.group(1), 0), int(mm.group(2)))
    return regs

def build_dump_orig(test_cu, regs):
    src = open(test_cu).read()
    n_out_extra = regs['xr'] + (regs['xd'] - 2) + regs['xp']  # dump all xr + xd2..xdN-1 + xp
    # renumber inputs %2..%7 -> %16..%21 (since we add 14 extra outputs)
    # Actually outputs are 0..(2+n_out_extra-1); inputs shift to that offset.
    in_off = 2 + n_out_extra
    for k, new in [('%2', f'%{in_off+0}'), ('%3', f'%{in_off+1}'), ('%4', f'%{in_off+2}'),
                   ('%5', f'%{in_off+3}'), ('%6', f'%{in_off+4}'), ('%7', f'%{in_off+5}')]:
        src = src.replace(f'"mov.b32 %%xr0, {k};', f'"mov.b32 %%xr0, {new};')
        src = src.replace(f'"mov.b32 %%xr1, {k};', f'"mov.b32 %%xr1, {new};')
        src = src.replace(f'"mov.b32 %%xr2, {k};', f'"mov.b32 %%xr2, {new};')
        src = src.replace(f'"mov.b32 %%xr3, {k};', f'"mov.b32 %%xr3, {new};')
        src = src.replace(f'"mov.u64 %%xd0, {k};', f'"mov.u64 %%xd0, {new};')
        src = src.replace(f'"mov.u64 %%xd1, {k};', f'"mov.u64 %%xd1, {new};')
        src = src.replace(f'"mov.u64 %%xd2, {k};', f'"mov.u64 %%xd2, {new};')

    # append dump movs before final mov.u64 %0, %xd0
    needle = '"mov.u64 %0, %%xd0;\\n\\t"\n        "mov.u64 %1, %%xd1;"'
    dump = ['"mov.u64 %0, %%xd0;\\n\\t"', '"mov.u64 %1, %%xd1;\\n\\t"']
    idx = 2
    for i in range(regs['xr']):
        dump.append(f'"mov.u32 %{idx}, %%xr{i};\\n\\t"'); idx += 1
    for i in range(2, regs['xd']):
        dump.append(f'"mov.u64 %{idx}, %%xd{i};\\n\\t"'); idx += 1
    for i in range(regs['xp']):
        dump.append(f'"selp.u32 %{idx}, 1, 0, %%xp{i};\\n\\t"'); idx += 1
    src = src.replace(needle, '\n        '.join(dump))

    # add output constraints
    extra = ''.join(f', "=r"(xr{i})' for i in range(regs['xr'])) + ''.join(f', "=l"(xd{i})' for i in range(2, regs['xd'])) + ''.join(f', "=r"(xp{i})' for i in range(regs['xp']))
    src = src.replace('"=l"(o0), "=l"(o1)', '"=l"(o0), "=l"(o1)' + extra, 1)

    # declarations
    decl = '    unsigned long long o0 = 0ULL, o1 = 0ULL;\n'
    decl_new = decl + ''.join(f'    unsigned int xr{i} = 0;\n' for i in range(regs['xr'])) + ''.join(f'    unsigned long long xd{i} = 0ULL;\n' for i in range(2, regs['xd'])) + ''.join(f'    unsigned int xp{i} = 0;\n' for i in range(regs['xp']))
    src = src.replace(decl, decl_new, 1)

    # store dumps at output[128..]
    store = '\n    if (tid == 0u) {\n        output[128] = o0;\n        output[129] = o1;\n'
    base = 130
    for i in range(regs['xr']):
        store += f'        output[{base+i}] = xr{i};\n'
    base += regs['xr']
    for i in range(2, regs['xd']):
        store += f'        output[{base+(i-2)}] = xd{i};\n'
    base += (regs['xd'] - 2)
    for i in range(regs['xp']):
        store += f'        output[{base+i}] = xp{i};\n'
    store += '    }\n'
    src = src.replace('    output[tid * 2u + 1u] = o1;\n}', '    output[tid * 2u + 1u] = o1;\n' + store + '}')
    return src

def build_dump_pc(pure_cu, regs):
    src = open(pure_cu).read()
    store = '\n    if (tid == 0u) {\n        output[128] = o0;\n        output[129] = o1;\n'
    base = 130
    for i in range(regs['xr']):
        store += f'        output[{base+i}] = xr{i};\n'
    base += regs['xr']
    for i in range(2, regs['xd']):
        store += f'        output[{base+(i-2)}] = xd{i};\n'
    base += (regs['xd'] - 2)
    for i in range(regs['xp']):
        store += f'        output[{base+i}] = xp{i};\n'
    store += '    }\n'
    src = src.replace('    output[tid * 2u + 1u] = o1;\n}', '    output[tid * 2u + 1u] = o1;\n' + store + '}')
    return src

def run(cubin_path, d, seed):
    out = '/tmp/dump.out'
    subprocess.run([RUNNER, '--cubin', cubin_path, '--kernel', 'test_kernel',
                    '--input', os.path.join(d, 'input.bin'), '--output', out, '--result', '/tmp/dump.json',
                    '--input-words', '256', '--output-words', '160', '--grid', '1', '--block', '64',
                    '--seed', str(seed), '--guards', '0', '--test-id', 'x', '--device', '0', '--timeout-ms', '30000'],
                   check=True, capture_output=True)
    data = open(out, 'rb').read()
    return struct.unpack('<160Q', data[40:40+160*8])

def main():
    d = sys.argv[1]
    import json
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else json.load(open(os.path.join(d, 'manifest.json')))['kernel_seed']
    regs = parse_regs(os.path.join(d, 'test.cu'))
    print(f'regs: {regs}  seed={seed}')

    open('/tmp/dump_orig.cu', 'w').write(build_dump_orig(os.path.join(d, 'test.cu'), regs))
    open('/tmp/dump_pc.cu', 'w').write(build_dump_pc(os.path.join(d, 'pure_cuda.cu'), regs))

    subprocess.run([NVCC, '-arch=sm_86', '-O2', '-Xptxas=-O0', '-cubin', '-o', '/tmp/dump_orig.cubin', '/tmp/dump_orig.cu'], check=True, capture_output=True)
    subprocess.run([NVCC, '-arch=sm_86', '-O2', '-Xptxas=-O0', '-cubin', '-o', '/tmp/dump_pc.cubin', '/tmp/dump_pc.cu'], check=True, capture_output=True)

    vo = run('/tmp/dump_orig.cubin', d, seed)
    vp = run('/tmp/dump_pc.cubin', d, seed)

    names = ['o0', 'o1'] + [f'xr{i}' for i in range(regs['xr'])] + [f'xd{i}' for i in range(2, regs['xd'])] + [f'xp{i}' for i in range(regs['xp'])]
    ndiff = 0
    for i, n in enumerate(names):
        a, b = vo[128+i], vp[128+i]
        if a != b:
            print(f'  DIFF {n}: orig=0x{a:016X} pc=0x{b:016X}')
            ndiff += 1
    if ndiff == 0:
        print('  all registers match!')
    else:
        print(f'  {ndiff} registers differ')

if __name__ == '__main__':
    main()
