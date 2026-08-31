#!/usr/bin/env python3
"""Fast classification: program analysis on all cases + GPU verification on ~5 representatives.

Phase 1 (no GPU, ~5s): load every manifest, rebuild the program, extract op signatures
and trigger-family counts, cluster by op composition.

Phase 2 (GPU, ~15min): validate 1 representative per cluster (O0/O1 + family-disable).

Phase 3: write CLASSIFICATION_REPORT.md in corpus/differentials/.
"""

import copy
import json
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
NVCC = "/usr/local/cuda-12.6/bin/nvcc"
RUNNER = ROOT / "native" / "cuda_runner"
WORK = Path("/tmp/clsf"); WORK.mkdir(parents=True, exist_ok=True)
M32 = 0xFFFFFFFF; M64 = 0xFFFFFFFFFFFFFFFF

from pipeline.manifest import case_from_dir
from generator.strategies.inline_ptx import _gen_program, _emit_cuda, _simulate
from generator.strategies.helpers import unpack_words

# -- trigger families we care about --
TF = {
    "set.*":      ["set."],
    "bfind.*":    ["bfind."],
    "prmt.*":     ["prmt."],
    "bfe.*":      ["bfe."],
    "bfi.*":      ["bfi."],
    "shf.*":      ["shf."],
    "lop3.*":     ["lop3."],
    "sad.*":      ["sad."],
    "mul24.*":    ["mul24."],
    "mad.*":      ["mad."],
    "div.*":      ["div."],
    "rem.*":      ["rem."],
    "abs.*":      ["abs."],
    "neg.*":      ["neg."],
    "bmsk.*":     ["bmsk."],
    "szext.*":    ["szext."],
    "fns.*":      ["fns."],
    "slct.*":     ["slct."],
    "popc.*":     ["popc."],
    "clz.*":      ["clz."],
    "brev.*":     ["brev."],
}

def program_analysis(prog) -> dict:
    """Fast no-GPU analysis: which trigger families are present and how pervasively."""
    op_counts = Counter(st["op"] for st in prog.steps if st["op"] != "out")
    signature = dict(op_counts)
    families = {}
    prmt_self = []
    overmsb_bf = []
    for i, st in enumerate(prog.steps):
        if st["op"] == "out": continue
        for fam, prefixes in TF.items():
            if any(st["op"].startswith(p) for p in prefixes):
                families.setdefault(fam, []).append(st["op"])
        if st["op"] == "prmt.b32" and len(st["srcs"]) == 2 and st["srcs"][0] == st["srcs"][1]:
            prmt_self.append((i, st["imm"]))
        if st["op"].startswith("bf") and st["imm"]:
            w = 64 if st["op"].endswith("64") else 32
            if st["imm"][0] + st["imm"][1] > w:
                overmsb_bf.append((i, st["imm"]))
    fam_counts = {k: len(set(v)) for k, v in families.items()}
    return {"op_count": len(signature), "fam_counts": fam_counts,
            "prmt_self": len(prmt_self), "overmsb_bf": len(overmsb_bf),
            "signature": {k: v for k, v in op_counts.most_common(20)}}

def cluster_key(ana: dict) -> str:
    """Stable cluster label from the family presence vector."""
    parts = []
    for fam in sorted(TF):
        if ana["fam_counts"].get(fam, 0) > 0:
            parts.append(fam)
    # add the pressure knobs
    parts.append(f"prmtSelf={min(ana['prmt_self'], 2)}")
    parts.append(f"overMsbBF={min(ana['overmsb_bf'], 2)}")
    return "|".join(parts)

def oracle_for(prog, input_bytes, seed):
    ws = unpack_words(input_bytes); exp = []
    for tid in range(64):
        w0=ws[(tid*3)%256]^seed; w1=ws[(tid*5+1)%256]^(seed>>11)
        w2=ws[(tid*7+2)%256]^(seed>>29); w3=ws[(tid*11+3)%256]^(seed>>37)
        a0=(w0^(w0>>32))&M32; a1=(w1^(w1>>32))&M32
        a2=(w2&M32)^((seed>>47)&M32); a3=w3&M32
        b0=(ws[(tid*13+4)%256]^(seed*3))&M64; b1=(ws[(tid*17+5)%256]^(seed>>13))&M64
        e0,e1 = _simulate(prog, a0,a1,a2,a3,b0,b1)
        exp.append(e0); exp.append(e1)
    return b''.join(struct.pack('<Q', x) for x in exp)

def run_gpu(prog, ib, seed, opt, tag):
    (WORK/"t.cu").write_text(_emit_cuda(prog)); (WORK/"in.bin").write_bytes(ib)
    p = subprocess.run([NVCC,"-arch=sm_86","-cubin",f"-Xptxas=-O{opt}","-o",str(WORK/"k.cubin"),str(WORK/"t.cu")],
                       capture_output=True, text=True, timeout=240)
    if p.returncode: return None, f"CERR:{p.stderr[-150:]}"
    r = subprocess.run([str(RUNNER),"--cubin",str(WORK/"k.cubin"),"--kernel","test_kernel",
      "--input",str(WORK/"in.bin"),"--output",str(WORK/f"{tag}.bin"),"--result",str(WORK/"e.json"),
      "--input-words","256","--output-words","128","--grid","1","--block","64",
      "--seed",str(seed),"--guards","8","--test-id","x","--timeout-ms","60000"],
      capture_output=True, text=True, timeout=120)
    if r.returncode: return None, f"RUN:{r.stderr[-150:]}"
    data = (WORK/f"{tag}.bin").read_bytes()
    status = struct.unpack('<I', data[16:20])[0]
    if status != 0: return None, f"STATUS={status}"
    return data[40:-8], None

def disable_family(prog, prefixes):
    c = copy.deepcopy(prog)
    for st in c.steps:
        if any(st["op"].startswith(p) for p in prefixes):
            dk = st["dst"][0]
            if dk == "pred": st["op"]="setp.eq.u32"; st["srcs"]=st["srcs"][:2] if len(st["srcs"])>=2 else [("r32",0),("r32",0)]; st["imm"]=None
            else: st["op"]="mov.u64" if dk=="r64" else "mov.b32"; st["srcs"]=st["srcs"][:1] if st["srcs"] else [("r32",0)]; st["imm"]=None
    return c

# -- main --
corpus = ROOT / "corpus" / "differentials"
dirs = sorted(d for d in corpus.iterdir() if (d/"manifest.json").exists())
print(f"Phase 1: program analysis of {len(dirs)} cases ...")

analyses = {}
for d in dirs:
    try:
        m = json.loads((d/"manifest.json").read_text())
        prog = _gen_program(m["kernel_seed"], m["strategy_params"])
        a = program_analysis(prog)
        a["dir"] = d.name
        a["seed"] = m["kernel_seed"]
        a["params"] = {k: m["strategy_params"].get(k) for k in ("n_instr","n_regs32","coverage_mode")}
        analyses[d.name] = a
    except Exception as e:
        print(f"  {d.name}: ANALYSIS_ERROR {e}")

# cluster
clusters = {}
for name, a in analyses.items():
    key = cluster_key(a)
    clusters.setdefault(key, []).append(name)

print(f"Phase 1 done: {len(analyses)} analyzed, {len(clusters)} clusters")
for key, names in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    print(f"  cluster [{len(names)}]: {key[:120]}")

# Phase 2: verify 1 representative per cluster
print("\nPhase 2: GPU validation of 1 rep per cluster ...")
reps = {key: names[0] for key, names in clusters.items()}
verified = {}
for key, name in sorted(reps.items()):
    d = corpus / name
    case = case_from_dir(d); seed = case.kernel_seed; ib = case.input_bytes
    prog = _gen_program(seed, case.params); exp = oracle_for(prog, ib, seed)
    o0, e0 = run_gpu(prog, ib, seed, 0, "o0")
    if o0 is None: verified[key] = ("COMPILE_ERR", e0); continue
    o1, e1 = run_gpu(prog, ib, seed, 1, "o1")
    if o1 is None: verified[key] = ("RUN_ERR", e1); continue
    o0_ok = o0 == exp; o1_ok = o1 == exp; diverges = o0_ok and not o1_ok
    # family disable test (2 families most present)
    a = analyses[name]
    top_fams = sorted(a["fam_counts"], key=lambda k: a["fam_counts"][k])[-4:]
    essential = []
    for fam in top_fams:
        c = disable_family(prog, TF[fam])
        exp_c = oracle_for(c, ib, seed)
        o1f, _ = run_gpu(c, ib, seed, 1, "o1f")
        if o1f is not None and o1f == exp_c:
            essential.append(fam)
    verified[key] = (diverges, o0_ok, o1_ok, essential)
    print(f"  [{name}] diverges={diverges} O0={o0_ok} O1={o1_ok} essential={essential}")

# Phase 3: write report
report = corpus / "CLASSIFICATION_REPORT.md"
lines = [
    "# cuda_stress differential classification report",
    "",
    f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total cases: {len(dirs)} | Clusters: {len(clusters)} | Verified reps: {len(verified)}",
    "",
    "## Trigger family clusters",
    "",
]
for key, names in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    v = verified.get(key)
    status_str = ""
    if v:
        div, o0, o1, ess = v
        status_str = f"**{'CONFIRMED' if div else 'NEEDS REVIEW'}** (O0={o0}, O1={'diverges' if not o1 else 'matches'}) essential={ess}"
    lines.append(f"### Cluster `{key[:90]}...` ({len(names)} cases) {status_str}")
    params = Counter(str(analyses[n]["params"]) for n in names)
    lines.append(f"- params: {dict(params)}")
    lines.append(f"- examples: {', '.join(names[:5])}")
    if len(names) > 5: lines.append(f"- ... and {len(names)-5} more")
    lines.append("")

lines.append("---")
lines.append("## Per-case trigger-family matrix")
lines.append("")
lines.append("| case_id | cluster_families | prmt_self | overMsb_bf |")
lines.append("|---|---|---|---|")
for name in sorted(analyses):
    a = analyses[name]
    fams = "+".join(sorted(f for f in TF if a["fam_counts"].get(f,0)>0))
    lines.append(f"| {name} | {fams[:60]} | {a['prmt_self']} | {a['overmsb_bf']} |")

report.write_text("\n".join(lines) + "\n")
print(f"\nwritten {report}")
