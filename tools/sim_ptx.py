#!/usr/bin/env python3
"""Reference Python simulator for inline-PTX test.cu (tid==0 only).

Uses the empirically-verified semantics of every PTX opcode to recompute the
expected o0/o1 for a single thread, so we can cross-check the generated C.

Usage: python3 tools/sim_ptx.py <dir>
"""
import re, struct, json, sys, os

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

def s32(x): x &= M32; return x - (1 << 32) if x & (1 << 31) else x
def s64(x): x &= M64; return x - (1 << 64) if x & (1 << 63) else x
def u32(x): return x & M32
def u64(x): return x & M64

def clz(x, w): x &= (1 << w) - 1; return w if x == 0 else w - x.bit_length()

def sxt24(v): v &= 0xFFFFFF; return v - 0x1000000 if v & 0x800000 else v

def mul24lo_s32(a, b): return u32(sxt24(a) * sxt24(b))
def mul24hi_s32(a, b): return u32((sxt24(a) * sxt24(b)) >> 16)
def mul24lo_u32(a, b): return u32((a & 0xFFFFFF) * (b & 0xFFFFFF))
def mul24hi_u32(a, b): return u32(((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 16)

def bfind_u32(x): return u32(31 - clz(x, 32))
def bfind_u64(x): return u32(63 - clz(x, 64))
def bfind_s32(x):
    u = u32(~x) if s32(x) < 0 else u32(x)
    return u32(31 - clz(u, 32))
def bfind_s64(x):
    u = u64(~x) if s64(x) < 0 else u64(x)
    return u32(63 - clz(u, 64))
def bfind_shamt_u32(x): return 0xFFFFFFFF if u32(x) == 0 else clz(x, 32)
def bfind_shamt_u64(x): return 0xFFFFFFFF if u64(x) == 0 else clz(x, 64)
def bfind_shamt_s32(x):
    u = u32(~x) if s32(x) < 0 else u32(x)
    return 0xFFFFFFFF if u == 0 else clz(u, 32)
def bfind_shamt_s64(x):
    u = u64(~x) if s64(x) < 0 else u64(x)
    return 0xFFFFFFFF if u == 0 else clz(u, 64)

def prmt(a, b, sel):
    r = 0
    for i in range(4):
        nib = (sel >> (4 * i)) & 0xF
        if nib < 4: byte = (a >> (8 * nib)) & 0xFF
        elif nib < 8: byte = (b >> (8 * (nib - 4))) & 0xFF
        else: byte = 0
        r |= byte << (8 * i)
    return r

def bfi(a, b, pos, length, width):
    e = min(length, width)
    mask = ((1 << width) - 1) if e >= width else ((1 << e) - 1)
    p = pos % width
    m = (mask << p) & ((1 << width) - 1)
    return ((b & ~m) | ((a << p) & m)) & ((1 << width) - 1)

def bfe_u(a, pos, length, width):
    if pos >= width: return 0
    w = min(length, width - pos)
    mask = ((1 << width) - 1) if w >= width else ((1 << w) - 1)
    return (a >> pos) & mask

def bfe_s(a, pos, length, width):
    if pos >= width: return 0
    w = min(length, width - pos)
    v = a >> pos
    if w == 0: return 0
    if w >= width: return v
    mask = (1 << w) - 1
    v &= mask
    if v & (1 << (w - 1)): v |= ~mask & ((1 << width) - 1)
    return v

def shf_l(a, b, c, clamp):
    shift = min(c, 32) if clamp else (c & 31)
    bc = (u64(b) << 32) | u32(a)
    return u32((bc << shift) >> 32)

def shf_r(a, b, c, clamp):
    shift = min(c, 32) if clamp else (c & 31)
    bc = (u64(b) << 32) | u32(a)
    return u32(bc >> shift)

def fns(a, b): return b if (u32(a) >> b) & 1 else 0xFFFFFFFF
def sad(a, b, c): return u32((a - b if a > b else b - a) + c)

def lop3(a, b, c, lut):
    r = 0
    for i in range(32):
        idx = ((c >> i) & 1) | (((b >> i) & 1) << 1) | (((a >> i) & 1) << 2)
        r |= ((lut >> idx) & 1) << i
    return r

def shl_u(x, n, w): return 0 if n >= w else u32(x << n)
def shr_u(x, n, w): return 0 if n >= w else u32(x >> n)
def shr_s(x, n, w):
    if n >= w: n = w - 1
    return u32((s32(x) if w == 32 else s64(x)) >> n)

def extract_asm(path):
    src = open(path).read()
    m = re.search(r'asm volatile\((.*?)\s*:\s*"=l"\(o0\)', src, re.S)
    body = m.group(1)
    instrs = re.findall(r'"(.*?)"', body)
    flat = []
    for i in instrs:
        for part in i.replace('\\n\\t', '\n').replace('\\n', '\n').split(';'):
            part = part.strip().replace('%%', '%')
            if part and not part.startswith('.reg'):
                flat.append(part)
    return flat

def main():
    d = sys.argv[1]
    flat = extract_asm(os.path.join(d, 'test.cu'))
    seed = json.load(open(os.path.join(d, 'manifest.json')))['kernel_seed']
    data = struct.unpack('<256Q', open(os.path.join(d, 'input.bin'), 'rb').read())
    tid = 0; iw = 256
    w0 = data[(tid*3)%iw] ^ seed
    w1 = data[(tid*5+1)%iw] ^ (seed >> 11)
    w2 = data[(tid*7+2)%iw] ^ (seed >> 29)
    w3 = data[(tid*11+3)%iw] ^ (seed >> 37)
    a0 = u32(w0 ^ (w0 >> 32)); a1 = u32(w1 ^ (w1 >> 32))
    a2 = u32(u32(w2) ^ u32(seed >> 47)); a3 = u32(w3)
    b0 = u64(data[(tid*13+4)%iw] ^ (seed * 3))
    b1 = u64(data[(tid*17+5)%iw] ^ (seed >> 13))

    R = {f'xr{i}':0 for i in range(20)}
    D = {f'xd{i}':0 for i in range(10)}
    P = {f'xp{i}':0 for i in range(10)}

    def norm(r):
        r = r.lstrip('%')
        return {'0':'o0','1':'o1','2':'a0','3':'a1','4':'a2','5':'a3','6':'b0','7':'b1'}.get(r, r)
    def get(r):
        r = norm(r)
        if r.startswith('xr'): return R[r]
        if r.startswith('xd'): return D[r]
        if r.startswith('xp'): return P[r]
        return {'o0':0,'o1':1,'a0':a0,'a1':a1,'a2':a2,'a3':a3,'b0':b0,'b1':b1}[r]
    def setr(r, v):
        r = norm(r)
        if r.startswith('xr'): R[r] = u32(v)
        elif r.startswith('xd'): D[r] = u64(v)
        elif r.startswith('xp'): P[r] = 1 if v else 0
    def is_xd(r): return norm(r).startswith('xd')
    def sc(reg, st):
        if is_xd(reg): return s64(get(reg)) if st == 's64' else u64(get(reg))
        return s32(get(reg)) if st == 's32' else u32(get(reg))
    CMP = {'lt':'<','gt':'>','le':'<=','ge':'>=','eq':'==','ne':'!='}

    for f in flat:
        tok = f.split(); op = tok[0]
        ops = [o.strip().rstrip(',') for o in f[len(op):].strip().split(',')]
        d = ops[0]
        def O(i):
            o = ops[i]
            return get(o) if o.startswith('%') else int(o, 0)

        if op == 'mov.b32': setr(d, O(1))
        elif op == 'mov.u64': setr(d, O(1))
        elif op in ('xor.b32','xor.b64'): setr(d, O(1) ^ O(2))
        elif op in ('add.u32','add.u64'): setr(d, O(1) + O(2))
        elif op == 'sub.u32': setr(d, O(1) - O(2))
        elif op in ('and.b32','and.b64'): setr(d, O(1) & O(2))
        elif op in ('or.b32','or.b64'): setr(d, O(1) | O(2))
        elif op == 'and.pred': setr(d, O(1) & O(2))
        elif op == 'or.pred': setr(d, O(1) | O(2))
        elif op == 'not.pred': setr(d, 0 if O(1) else 1)
        elif op == 'not.b32': setr(d, ~O(1))
        elif op == 'not.b64': setr(d, ~O(1))
        elif op == 'neg.s32': setr(d, -O(1))
        elif op == 'neg.s64': setr(d, -O(1))
        elif op == 'abs.s32': setr(d, -s32(O(1)) if s32(O(1)) < 0 else O(1))
        elif op == 'abs.s64': setr(d, -s64(O(1)) if s64(O(1)) < 0 else O(1))
        elif op == 'clz.b32': setr(d, clz(O(1), 32))
        elif op == 'clz.b64': setr(d, clz(O(1), 64))
        elif op in ('cnot.b32','cnot.b64'): setr(d, 1 if O(1) == 0 else 0)
        elif op == 'popc.b32': setr(d, bin(O(1) & M32).count('1'))
        elif op == 'popc.b64': setr(d, bin(O(1) & M64).count('1'))
        elif op == 'brev.b32': setr(d, int(f'{O(1)&M32:032b}'[::-1], 2))
        elif op == 'brev.b64': setr(d, int(f'{O(1)&M64:064b}'[::-1], 2))
        elif op == 'cvt.u64.u32': setr(d, u32(O(1)))
        elif op == 'cvt.u32.u64': setr(d, O(1))
        elif op == 'div.u32': setr(d, u32(O(1)) // O(2))
        elif op == 'div.u64': setr(d, u64(O(1)) // O(2))
        elif op == 'rem.u32': setr(d, u32(O(1)) % O(2))
        elif op == 'rem.u64': setr(d, u64(O(1)) % O(2))
        elif op == 'shl.b32': setr(d, shl_u(O(1), O(2), 32))
        elif op == 'shl.b64': setr(d, 0 if O(2) >= 64 else O(1) << O(2))
        elif op == 'shr.u32': setr(d, shr_u(O(1), O(2), 32))
        elif op == 'shr.u64': setr(d, 0 if O(2) >= 64 else O(1) >> O(2))
        elif op == 'shr.s32': setr(d, shr_s(O(1), O(2), 32))
        elif op == 'shr.s64': setr(d, shr_s(O(1), O(2), 64))
        elif op == 'mul.lo.u32': setr(d, u32(O(1)) * u32(O(2)))
        elif op == 'mul.lo.u64': setr(d, u64(O(1)) * u64(O(2)))
        elif op == 'mul.hi.u32': setr(d, (u32(O(1)) * u32(O(2))) >> 32)
        elif op == 'mul.hi.u64': setr(d, (u64(O(1)) * u64(O(2))) >> 64)
        elif op == 'mul.wide.s32': setr(d, s32(O(1)) * s32(O(2)))
        elif op == 'mul.wide.u32': setr(d, u32(O(1)) * u32(O(2)))
        elif op == 'mul24.lo.s32': setr(d, mul24lo_s32(O(1), O(2)))
        elif op == 'mul24.lo.u32': setr(d, mul24lo_u32(O(1), O(2)))
        elif op == 'mul24.hi.s32': setr(d, mul24hi_s32(O(1), O(2)))
        elif op == 'mul24.hi.u32': setr(d, mul24hi_u32(O(1), O(2)))
        elif op == 'mad.lo.u32': setr(d, u32(O(1)) * u32(O(2)) + u32(O(3)))
        elif op == 'mad.lo.u64': setr(d, u64(O(1)) * u64(O(2)) + u64(O(3)))
        elif op == 'mad24.lo.s32': setr(d, mul24lo_s32(O(1), O(2)) + O(3))
        elif op == 'mad24.lo.u32': setr(d, mul24lo_u32(O(1), O(2)) + O(3))
        elif op == 'mad24.hi.s32': setr(d, mul24hi_s32(O(1), O(2)) + O(3))
        elif op == 'mad24.hi.u32': setr(d, mul24hi_u32(O(1), O(2)) + O(3))
        elif op == 'bfind.u32': setr(d, bfind_u32(O(1)))
        elif op == 'bfind.u64': setr(d, bfind_u64(O(1)))
        elif op == 'bfind.s32': setr(d, bfind_s32(O(1)))
        elif op == 'bfind.s64': setr(d, bfind_s64(O(1)))
        elif op == 'bfind.shiftamt.u32': setr(d, bfind_shamt_u32(O(1)))
        elif op == 'bfind.shiftamt.u64': setr(d, bfind_shamt_u64(O(1)))
        elif op == 'bfind.shiftamt.s32': setr(d, bfind_shamt_s32(O(1)))
        elif op == 'bfind.shiftamt.s64': setr(d, bfind_shamt_s64(O(1)))
        elif op == 'prmt.b32': setr(d, prmt(O(1), O(2), O(3)))
        elif op == 'bfi.b64': setr(d, bfi(O(1), O(2), O(3), O(4), 64))
        elif op == 'bfi.b32': setr(d, bfi(O(1), O(2), O(3), O(4), 32))
        elif op == 'bfe.s32': setr(d, bfe_s(O(1), O(2), O(3), 32))
        elif op == 'bfe.u32': setr(d, bfe_u(O(1), O(2), O(3), 32))
        elif op == 'bfe.s64': setr(d, bfe_s(O(1), O(2), O(3), 64))
        elif op == 'bfe.u64': setr(d, bfe_u(O(1), O(2), O(3), 64))
        elif op == 'fns.b32': setr(d, fns(O(1), O(2)))
        elif op == 'sad.u32': setr(d, sad(O(1), O(2), O(3)))
        elif op == 'lop3.b32': setr(d, lop3(O(1), O(2), O(3), O(4)))
        elif op == 'shf.l.wrap.b32': setr(d, shf_l(O(1), O(2), O(3), 0))
        elif op == 'shf.l.clamp.b32': setr(d, shf_l(O(1), O(2), O(3), 1))
        elif op == 'shf.r.wrap.b32': setr(d, shf_r(O(1), O(2), O(3), 0))
        elif op == 'shf.r.clamp.b32': setr(d, shf_r(O(1), O(2), O(3), 1))
        elif op == 'min.s32': setr(d, O(1) if s32(O(1)) < s32(O(2)) else O(2))
        elif op == 'min.u32': setr(d, O(1) if u32(O(1)) < u32(O(2)) else O(2))
        elif op == 'min.s64': setr(d, O(1) if s64(O(1)) < s64(O(2)) else O(2))
        elif op == 'min.u64': setr(d, O(1) if u64(O(1)) < u64(O(2)) else O(2))
        elif op == 'max.s32': setr(d, O(1) if s32(O(1)) > s32(O(2)) else O(2))
        elif op == 'max.u32': setr(d, O(1) if u32(O(1)) > u32(O(2)) else O(2))
        elif op == 'max.s64': setr(d, O(1) if s64(O(1)) > s64(O(2)) else O(2))
        elif op == 'max.u64': setr(d, O(1) if u64(O(1)) > u64(O(2)) else O(2))
        elif op == 'slct.b32.s32': setr(d, O(1) if s32(O(3)) >= 0 else O(2))
        elif op == 'slct.b64.s32': setr(d, O(1) if s32(O(3)) >= 0 else O(2))
        elif op.startswith('setp.'):
            parts = op.split('.'); cmp = parts[1]; st = parts[2]
            P[norm(d)] = 1 if eval(f'{sc(ops[1],st)} {CMP[cmp]} {sc(ops[2],st)}') else 0
        elif op.startswith('set.'):
            parts = op.split('.'); cmp = parts[1]; st = parts[-1]
            setr(d, 0xFFFFFFFF if eval(f'{sc(ops[1],st)} {CMP[cmp]} {sc(ops[2],st)}') else 0)
        elif op in ('selp.b32','selp.b64'):
            setr(d, O(1) if P[norm(ops[3])] else O(2))
        else:
            raise SystemExit('UNHANDLED: ' + f)
        dn = norm(d)
        if dn in ('xd0','xd1','xd3','xd5') and os.environ.get('TRACE'):
            print(f'{flat.index(f):3d} {f:40s} -> {dn}=0x{get(d):016X}', file=sys.stderr)

    print(f'o0 = 0x{D["xd0"]:016X}')
    print(f'o1 = 0x{D["xd1"]:016X}')
    for i in range(12): print(f'xr{i} = 0x{R[f"xr{i}"]:08X}')
    for i in range(6): print(f'xd{i} = 0x{D[f"xd{i}"]:016X}')
    for i in range(5): print(f'xp{i} = {P[f"xp{i}"]}')

if __name__ == '__main__':
    main()
