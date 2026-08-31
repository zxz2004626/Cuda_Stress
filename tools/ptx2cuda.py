#!/usr/bin/env python3
"""Translate an inline-PTX bug-report test.cu into a pure-CUDA .cu file.

Reads a directory containing test.cu (with a single `asm volatile` block and the
standard cuda_stress kernel signature), extracts the inline-PTX instruction
stream, and emits `pure_cuda.cu` with the same computation expressed in plain C.

Usage: python3 tools/ptx2cuda.py corpus/differentials/bug_reports/<name>
"""
import re
import sys
import os

# ---------------------------------------------------------------- parsing

def extract_asm(path):
    src = open(path).read()
    m = re.search(r'asm volatile\((.*?)\s*:\s*"=l"\(o0\)', src, re.S)
    if not m:
        raise SystemExit(f"no asm block found in {path}")
    body = m.group(1)
    instrs = re.findall(r'"(.*?)"', body)
    flat = []
    for i in instrs:
        i = i.replace('\\n\\t', '\n').replace('\\n', '\n')
        for part in i.split(';'):
            part = part.strip()
            if part:
                flat.append(part.replace('%%', '%'))
    # .reg declarations tell us how many registers of each kind
    regs = {'xr': 10, 'xd': 6, 'xp': 4}
    for f in flat:
        mm = re.match(r'\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>', f)
        if mm:
            name = mm.group(2)  # xr / xd / xp
            n = int(mm.group(3))
            regs[name] = max(regs.get(name, 0), n)
    flat = [f for f in flat if not f.startswith('.reg')]
    return flat, regs

# ---------------------------------------------------------------- helpers

HELPERS = r'''#include <cstdint>

#define MIN(a,b) ((a)<(b)?(a):(b))

__device__ __forceinline__ int sxt24(int a) {
    unsigned int u = (unsigned int)a & 0xFFFFFFu;
    return (int)((u ^ 0x800000u) - 0x800000u);
}
__device__ __forceinline__ unsigned int mul24lo_s32(int a, int b) {
    return (unsigned int)(sxt24(a) * sxt24(b));
}
__device__ __forceinline__ unsigned int mul24hi_s32(int a, int b) {
    long long p = (long long)sxt24(a) * (long long)sxt24(b);
    return (unsigned int)(int)(p >> 16);
}
__device__ __forceinline__ unsigned int mul24lo_u32(unsigned int a, unsigned int b) {
    return (a & 0xFFFFFFu) * (b & 0xFFFFFFu);
}
__device__ __forceinline__ unsigned int mul24hi_u32(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)(a & 0xFFFFFFu) * (b & 0xFFFFFFu)) >> 16);
}
__device__ __forceinline__ unsigned int bfind_u32(unsigned int x) { return 31u - __clz(x); }
__device__ __forceinline__ unsigned int bfind_u64(unsigned long long x) { return 63u - __clzll(x); }
__device__ __forceinline__ unsigned int bfind_s32(int x) {
    unsigned int u = (x < 0) ? (unsigned int)(~x) : (unsigned int)x;
    return 31u - __clz(u);
}
__device__ __forceinline__ unsigned int bfind_s64(long long x) {
    unsigned long long u = (x < 0) ? (unsigned long long)(~x) : (unsigned long long)x;
    return 63u - __clzll(u);
}
__device__ __forceinline__ unsigned int bfind_shamt_u32(unsigned int x) { return (x == 0u) ? 0xFFFFFFFFu : __clz(x); }
__device__ __forceinline__ unsigned int bfind_shamt_u64(unsigned long long x) { return (x == 0ULL) ? 0xFFFFFFFFu : (unsigned int)__clzll(x); }
__device__ __forceinline__ unsigned int bfind_shamt_s32(int x) {
    unsigned int u = (x < 0) ? (unsigned int)(~x) : (unsigned int)x;
    return (u == 0u) ? 0xFFFFFFFFu : __clz(u);
}
__device__ __forceinline__ unsigned int bfind_shamt_s64(long long x) {
    unsigned long long u = (x < 0) ? (unsigned long long)(~x) : (unsigned long long)x;
    return (u == 0ULL) ? 0xFFFFFFFFu : (unsigned int)__clzll(u);
}
__device__ __forceinline__ unsigned int prmt(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned int r = 0;
    for (int i = 0; i < 4; i++) {
        unsigned int nib = (sel >> (4 * i)) & 0xFu;
        unsigned int byte;
        if (nib < 4) byte = (a >> (8 * nib)) & 0xFFu;
        else if (nib < 8) byte = (b >> (8 * (nib - 4))) & 0xFFu;
        else byte = 0;
        r |= byte << (8 * i);
    }
    return r;
}
__device__ __forceinline__ unsigned long long bfi64(unsigned long long a, unsigned long long b, unsigned int pos, unsigned int len) {
    unsigned int e = MIN(len, 64);
    unsigned long long mask = (e >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << e) - 1ULL);
    unsigned int p = pos & 63;
    unsigned long long m = mask << p;
    return (b & ~m) | ((a << p) & m);
}
__device__ __forceinline__ unsigned int bfi32(unsigned int a, unsigned int b, unsigned int pos, unsigned int len) {
    unsigned int e = MIN(len, 32);
    unsigned int mask = (e >= 32) ? 0xFFFFFFFFu : ((1u << e) - 1u);
    unsigned int p = pos & 31;
    unsigned int m = mask << p;
    return (b & ~m) | ((a << p) & m);
}
__device__ __forceinline__ unsigned int bfe_u32(unsigned int a, unsigned int pos, unsigned int len) {
    if (pos >= 32) return 0u;
    unsigned int width = MIN(len, 32 - pos);
    unsigned int mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
    return (a >> pos) & mask;
}
__device__ __forceinline__ int bfe_s32(int a, unsigned int pos, unsigned int len) {
    if (pos >= 32) return 0;
    unsigned int width = MIN(len, 32 - pos);
    unsigned int v = (unsigned int)a >> pos;
    if (width == 0) return 0;
    if (width >= 32) return (int)v;
    unsigned int mask = (1u << width) - 1u;
    v &= mask;
    if (v & (1u << (width - 1))) v |= ~mask;
    return (int)v;
}
__device__ __forceinline__ unsigned long long bfe_u64(unsigned long long a, unsigned int pos, unsigned int len) {
    if (pos >= 64) return 0ULL;
    unsigned int width = MIN(len, 64 - pos);
    unsigned long long mask = (width >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << width) - 1ULL);
    return (a >> pos) & mask;
}
__device__ __forceinline__ long long bfe_s64(long long a, unsigned int pos, unsigned int len) {
    if (pos >= 64) return 0;
    unsigned int width = MIN(len, 64 - pos);
    unsigned long long v = (unsigned long long)a >> pos;
    if (width == 0) return 0;
    if (width >= 64) return (long long)v;
    unsigned long long mask = (1ULL << width) - 1ULL;
    v &= mask;
    if (v & (1ULL << (width - 1))) v |= ~mask;
    return (long long)v;
}
__device__ __forceinline__ unsigned int fns(unsigned int a, unsigned int b) {
    return ((a >> b) & 1u) ? b : 0xFFFFFFFFu;
}
__device__ __forceinline__ unsigned int sad(unsigned int a, unsigned int b, unsigned int c) {
    return (a > b ? a - b : b - a) + c;
}
__device__ __forceinline__ unsigned int lop3(unsigned int a, unsigned int b, unsigned int c, unsigned int lut) {
    unsigned int r = 0;
    for (int i = 0; i < 32; i++) {
        unsigned int idx = ((c >> i) & 1u) | (((b >> i) & 1u) << 1) | (((a >> i) & 1u) << 2);
        r |= ((lut >> idx) & 1u) << i;
    }
    return r;
}
__device__ __forceinline__ unsigned int shl_u32(unsigned int a, unsigned int n) { return (n >= 32) ? 0u : (a << n); }
__device__ __forceinline__ unsigned long long shl_u64(unsigned long long a, unsigned int n) { return (n >= 64) ? 0ULL : (a << n); }
__device__ __forceinline__ unsigned int shr_u32(unsigned int a, unsigned int n) { return (n >= 32) ? 0u : (a >> n); }
__device__ __forceinline__ unsigned long long shr_u64(unsigned long long a, unsigned int n) { return (n >= 64) ? 0ULL : (a >> n); }
__device__ __forceinline__ unsigned int shr_s32(int a, unsigned int n) { return (n >= 32) ? (unsigned int)(a >> 31) : (unsigned int)(a >> n); }
__device__ __forceinline__ unsigned long long shr_s64(long long a, unsigned int n) { return (n >= 64) ? (unsigned long long)(a >> 63) : (unsigned long long)(a >> n); }
__device__ __forceinline__ unsigned int shf_l(unsigned int a, unsigned int b, unsigned int c, int clamp) {
    unsigned int shift = clamp ? MIN(c, 32) : (c & 31);
    unsigned long long bc = ((unsigned long long)b << 32) | a;
    return (unsigned int)((bc << shift) >> 32);
}
__device__ __forceinline__ unsigned int shf_r(unsigned int a, unsigned int b, unsigned int c, int clamp) {
    unsigned int shift = clamp ? MIN(c, 32) : (c & 31);
    unsigned long long bc = ((unsigned long long)b << 32) | a;
    return (unsigned int)(bc >> shift);
}

'''

HEADER = '''// strategy=pure_cuda  — pure-CUDA translation of {name}/test.cu
// (auto-generated from the inline-PTX instruction stream).
//
// The original differential is a ptxas -O1+ miscompilation of an inline-PTX-only
// opcode. nvcc never emits that opcode from C source, so this translation is
// semantically equivalent but does NOT reproduce the differential.
// See PURE_CUDA_CONVERSION.md.

extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned long long w0 = input[(tid * 3u) % input_words] ^ seed;
    unsigned long long w1 = input[(tid * 5u + 1u) % input_words] ^ (seed >> 11u);
    unsigned long long w2 = input[(tid * 7u + 2u) % input_words] ^ (seed >> 29u);
    unsigned long long w3 = input[(tid * 11u + 3u) % input_words] ^ (seed >> 37u);
    unsigned int a0 = (unsigned int)(w0 ^ (w0 >> 32u));
    unsigned int a1 = (unsigned int)(w1 ^ (w1 >> 32u));
    unsigned int a2 = (unsigned int)(w2) ^ (unsigned int)(seed >> 47u);
    unsigned int a3 = (unsigned int)(w3);
    unsigned long long b0 = input[(tid * 13u + 4u) % input_words] ^ (seed * 3u);
    unsigned long long b1 = input[(tid * 17u + 5u) % input_words] ^ (seed >> 13u);

'''

def reg(name):
    if name.startswith('%xr'): return 'xr' + name[3:]
    if name.startswith('%xd'): return 'xd' + name[3:]
    if name.startswith('%xp'): return 'xp' + name[3:]
    return {'%0':'o0','%1':'o1','%2':'a0','%3':'a1','%4':'a2','%5':'a3','%6':'b0','%7':'b1'}[name]

def src_cast(regname, stype):
    if regname.startswith('%xd'):
        return '(long long)' if stype == 's64' else '(unsigned long long)'
    return '(int)' if stype == 's32' else '(unsigned int)'

def is_pred(r):
    return r.startswith('%xp')

def cst32(s):
    return s + 'u' if not s.startswith('0x') else s + 'u'

def cst64(s):
    return s + 'ULL'

CMP = {'lt':'<','gt':'>','le':'<=','ge':'>=','eq':'==','ne':'!='}

def translate(flat, regs):
    lines = []
    for f in flat:
        tok = f.split(); op = tok[0]
        ops = [o.strip() for o in f[len(op):].strip().split(',')]
        d = ops[0]
        def R(i):
            o = ops[i]
            return reg(o) if o.startswith('%') else o

        if op == 'mov.b32':
            s = ops[1]; lines.append(f'    {reg(d)} = {reg(s) if s.startswith("%") else cst32(s)};')
        elif op == 'mov.u64':
            s = ops[1]; lines.append(f'    {reg(d)} = {reg(s) if s.startswith("%") else cst64(s)};')
        elif op in ('xor.b32','xor.b64'): lines.append(f'    {reg(d)} = {R(1)} ^ {R(2)};')
        elif op in ('add.u32','add.u64'): lines.append(f'    {reg(d)} = {R(1)} + {R(2)};')
        elif op == 'sub.u32': lines.append(f'    {reg(d)} = {R(1)} - {R(2)};')
        elif op in ('and.b32','and.b64'): lines.append(f'    {reg(d)} = {R(1)} & {R(2)};')
        elif op in ('or.b32','or.b64'): lines.append(f'    {reg(d)} = {R(1)} | {R(2)};')
        elif op == 'and.pred': lines.append(f'    {reg(d)} = {R(1)} & {R(2)};')
        elif op == 'or.pred': lines.append(f'    {reg(d)} = {R(1)} | {R(2)};')
        elif op == 'not.pred': lines.append(f'    {reg(d)} = !{R(1)};')
        elif op == 'not.b32': lines.append(f'    {reg(d)} = ~{R(1)};')
        elif op == 'not.b64': lines.append(f'    {reg(d)} = ~{R(1)};')
        elif op == 'neg.s32': lines.append(f'    {reg(d)} = 0u - {R(1)};')
        elif op == 'neg.s64': lines.append(f'    {reg(d)} = 0ULL - {R(1)};')
        elif op == 'abs.s32': lines.append(f'    {reg(d)} = ((int){R(1)} < 0) ? (unsigned int)(-(int){R(1)}) : {R(1)};')
        elif op == 'abs.s64': lines.append(f'    {reg(d)} = ((long long){R(1)} < 0) ? (unsigned long long)(-(long long){R(1)}) : {R(1)};')
        elif op == 'clz.b32': lines.append(f'    {reg(d)} = __clz({R(1)});')
        elif op == 'clz.b64': lines.append(f'    {reg(d)} = (unsigned int)__clzll({R(1)});')
        elif op == 'cnot.b32' or op == 'cnot.b64': lines.append(f'    {reg(d)} = ({R(1)} == 0) ? 1u : 0u;')
        elif op == 'popc.b32': lines.append(f'    {reg(d)} = __popc({R(1)});')
        elif op == 'popc.b64': lines.append(f'    {reg(d)} = __popcll({R(1)});')
        elif op == 'brev.b32': lines.append(f'    {reg(d)} = __brev({R(1)});')
        elif op == 'brev.b64': lines.append(f'    {reg(d)} = __brevll({R(1)});')
        elif op == 'cvt.u64.u32': lines.append(f'    {reg(d)} = (unsigned long long){R(1)};')
        elif op == 'cvt.u32.u64': lines.append(f'    {reg(d)} = (unsigned int){R(1)};')
        elif op == 'div.u32': lines.append(f'    {reg(d)} = {R(1)} / {ops[2]};')
        elif op == 'div.u64': lines.append(f'    {reg(d)} = {R(1)} / {ops[2]};')
        elif op == 'rem.u32': lines.append(f'    {reg(d)} = {R(1)} % {ops[2]};')
        elif op == 'rem.u64': lines.append(f'    {reg(d)} = {R(1)} % {ops[2]};')
        elif op == 'shl.b32': lines.append(f'    {reg(d)} = shl_u32({R(1)}, {R(2)});')
        elif op == 'shl.b64': lines.append(f'    {reg(d)} = shl_u64({R(1)}, {R(2)});')
        elif op == 'shr.u32': lines.append(f'    {reg(d)} = shr_u32({R(1)}, {R(2)});')
        elif op == 'shr.u64': lines.append(f'    {reg(d)} = shr_u64({R(1)}, {R(2)});')
        elif op == 'shr.s32': lines.append(f'    {reg(d)} = shr_s32((int){R(1)}, {R(2)});')
        elif op == 'shr.s64': lines.append(f'    {reg(d)} = shr_s64((long long){R(1)}, {R(2)});')
        elif op in ('mul.lo.u32',): lines.append(f'    {reg(d)} = {R(1)} * {R(2)};')
        elif op == 'mul.lo.u64': lines.append(f'    {reg(d)} = {R(1)} * {R(2)};')
        elif op == 'mul.hi.u32': lines.append(f'    {reg(d)} = __umulhi({R(1)}, {R(2)});')
        elif op == 'mul.hi.u64': lines.append(f'    {reg(d)} = __umul64hi({R(1)}, {R(2)});')
        elif op == 'mul.wide.s32': lines.append(f'    {reg(d)} = (unsigned long long)((long long)(int){R(1)} * (long long)(int){R(2)});')
        elif op == 'mul.wide.u32': lines.append(f'    {reg(d)} = (unsigned long long){R(1)} * (unsigned long long){R(2)};')
        elif op == 'mul24.lo.s32': lines.append(f'    {reg(d)} = mul24lo_s32((int){R(1)}, (int){R(2)});')
        elif op == 'mul24.lo.u32': lines.append(f'    {reg(d)} = mul24lo_u32({R(1)}, {R(2)});')
        elif op == 'mul24.hi.s32': lines.append(f'    {reg(d)} = mul24hi_s32((int){R(1)}, (int){R(2)});')
        elif op == 'mul24.hi.u32': lines.append(f'    {reg(d)} = mul24hi_u32({R(1)}, {R(2)});')
        elif op == 'mad.lo.u32': lines.append(f'    {reg(d)} = {R(1)} * {R(2)} + {R(3)};')
        elif op == 'mad.lo.u64': lines.append(f'    {reg(d)} = {R(1)} * {R(2)} + {R(3)};')
        elif op == 'mad24.lo.s32': lines.append(f'    {reg(d)} = mul24lo_s32((int){R(1)}, (int){R(2)}) + {R(3)};')
        elif op == 'mad24.lo.u32': lines.append(f'    {reg(d)} = mul24lo_u32({R(1)}, {R(2)}) + {R(3)};')
        elif op == 'mad24.hi.s32': lines.append(f'    {reg(d)} = mul24hi_s32((int){R(1)}, (int){R(2)}) + {R(3)};')
        elif op == 'mad24.hi.u32': lines.append(f'    {reg(d)} = mul24hi_u32({R(1)}, {R(2)}) + {R(3)};')
        elif op == 'bfind.u32': lines.append(f'    {reg(d)} = bfind_u32({R(1)});')
        elif op == 'bfind.u64': lines.append(f'    {reg(d)} = bfind_u64({R(1)});')
        elif op == 'bfind.s32': lines.append(f'    {reg(d)} = bfind_s32((int){R(1)});')
        elif op == 'bfind.s64': lines.append(f'    {reg(d)} = bfind_s64((long long){R(1)});')
        elif op == 'bfind.shiftamt.u32': lines.append(f'    {reg(d)} = bfind_shamt_u32({R(1)});')
        elif op == 'bfind.shiftamt.u64': lines.append(f'    {reg(d)} = bfind_shamt_u64({R(1)});')
        elif op == 'bfind.shiftamt.s32': lines.append(f'    {reg(d)} = bfind_shamt_s32((int){R(1)});')
        elif op == 'bfind.shiftamt.s64': lines.append(f'    {reg(d)} = bfind_shamt_s64((long long){R(1)});')
        elif op == 'prmt.b32': lines.append(f'    {reg(d)} = prmt({R(1)}, {R(2)}, {ops[3]});')
        elif op == 'bfi.b64': lines.append(f'    {reg(d)} = bfi64({R(1)}, {R(2)}, {ops[3]}, {ops[4]});')
        elif op == 'bfi.b32': lines.append(f'    {reg(d)} = bfi32({R(1)}, {R(2)}, {ops[3]}, {ops[4]});')
        elif op == 'bfe.s32': lines.append(f'    {reg(d)} = (unsigned int)bfe_s32((int){R(1)}, {ops[2]}, {ops[3]});')
        elif op == 'bfe.u32': lines.append(f'    {reg(d)} = bfe_u32({R(1)}, {ops[2]}, {ops[3]});')
        elif op == 'bfe.s64': lines.append(f'    {reg(d)} = (unsigned long long)bfe_s64((long long){R(1)}, {ops[2]}, {ops[3]});')
        elif op == 'bfe.u64': lines.append(f'    {reg(d)} = bfe_u64({R(1)}, {ops[2]}, {ops[3]});')
        elif op == 'fns.b32': lines.append(f'    {reg(d)} = fns({R(1)}, {ops[2]});')
        elif op == 'sad.u32': lines.append(f'    {reg(d)} = sad({R(1)}, {R(2)}, {R(3)});')
        elif op == 'lop3.b32': lines.append(f'    {reg(d)} = lop3({R(1)}, {R(2)}, {R(3)}, {ops[4]});')
        elif op == 'shf.l.wrap.b32': lines.append(f'    {reg(d)} = shf_l({R(1)}, {R(2)}, {ops[3]}, 0);')
        elif op == 'shf.l.clamp.b32': lines.append(f'    {reg(d)} = shf_l({R(1)}, {R(2)}, {ops[3]}, 1);')
        elif op == 'shf.r.wrap.b32': lines.append(f'    {reg(d)} = shf_r({R(1)}, {R(2)}, {ops[3]}, 0);')
        elif op == 'shf.r.clamp.b32': lines.append(f'    {reg(d)} = shf_r({R(1)}, {R(2)}, {ops[3]}, 1);')
        elif op == 'min.s32': lines.append(f'    {reg(d)} = ((int){R(1)} < (int){R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'min.u32': lines.append(f'    {reg(d)} = ({R(1)} < {R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'min.s64': lines.append(f'    {reg(d)} = ((long long){R(1)} < (long long){R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'min.u64': lines.append(f'    {reg(d)} = ({R(1)} < {R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'max.s32': lines.append(f'    {reg(d)} = ((int){R(1)} > (int){R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'max.u32': lines.append(f'    {reg(d)} = ({R(1)} > {R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'max.s64': lines.append(f'    {reg(d)} = ((long long){R(1)} > (long long){R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'max.u64': lines.append(f'    {reg(d)} = ({R(1)} > {R(2)}) ? {R(1)} : {R(2)};')
        elif op == 'slct.b32.s32':
            a,b,c = R(1), R(2), R(3); lines.append(f'    {reg(d)} = ((int){c} >= 0) ? {a} : {b};')
        elif op == 'slct.b64.s32':
            a,b,c = R(1), R(2), R(3); lines.append(f'    {reg(d)} = ((int){c} >= 0) ? {a} : {b};')
        elif op.startswith('setp.'):
            parts = op.split('.'); cmp = parts[1]; st = parts[2] if len(parts) > 2 else 'u32'
            a, b = R(1), R(2); ca, cb = src_cast(ops[1], st), src_cast(ops[2], st)
            lines.append(f'    {reg(d)} = ({ca}{a} {CMP[cmp]} {cb}{b}) ? 1 : 0;')
        elif op.startswith('set.'):
            parts = op.split('.'); cmp = parts[1]; st = parts[-1]
            a, b = R(1), R(2); ca, cb = src_cast(ops[1], st), src_cast(ops[2], st)
            lines.append(f'    {reg(d)} = ({ca}{a} {CMP[cmp]} {cb}{b}) ? 0xFFFFFFFFu : 0u;')
        elif op in ('selp.b32','selp.b64'):
            lines.append(f'    {reg(d)} = {R(3)} ? {R(1)} : {R(2)};')
        else:
            raise SystemExit(f'UNHANDLED opcode: {op}  ({f})')
    return '\n'.join(lines)

def main():
    d = sys.argv[1]
    test_cu = os.path.join(d, 'test.cu')
    flat, regs = extract_asm(test_cu)
    body = translate(flat, regs)

    name = os.path.basename(os.path.normpath(d))
    decl = '    unsigned int ' + ', '.join(f'xr{i}' for i in range(regs['xr'])) + ';\n' \
           '    unsigned long long ' + ', '.join(f'xd{i}' for i in range(regs['xd'])) + ';\n' \
           '    int ' + ', '.join(f'xp{i}' for i in range(regs['xp'])) + ';\n' \
           '    unsigned long long o0, o1;\n'

    header = HEADER.replace('{name}', name)
    footer = '\n    output[tid * 2u]     = o0;\n    output[tid * 2u + 1u] = o1;\n}\n'

    full = HELPERS + header + decl + body + footer
    out = os.path.join(d, 'pure_cuda.cu')
    open(out, 'w').write(full)
    print(f'wrote {out}  ({len(body.splitlines())} statements)')

if __name__ == '__main__':
    main()
