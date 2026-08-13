#!/usr/bin/env python3
"""Comprehensive differential audit and classification.

For every case in corpus/differentials:
  1. Validate divergence (O0==oracle, O1!=oracle, deterministic) -- compile+run if no builds.
  2. Identify the ESSENTIAL op families via family-disabling.
  3. Classify into trigger clusters.

Output: corpus/differentials/CLASSIFICATION_REPORT.md plus per-case review.json update.
"""

from __future__ import annotations

import copy
import json
import os
import random
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "native" / "cuda_runner"
NVCC = "/usr/local/cuda-12.6/bin/nvcc"
WORK = Path("/tmp/classify_diffs"); WORK.mkdir(parents=True, exist_ok=True)

from pipeline.manifest import case_from_dir
from generator.strategies.inline_ptx import _gen_program, _emit_cuda, _simulate
from generator.strategies.helpers import unpack_words

M32 = 0xFFFFFFFF; M64 = 0xFFFFFFFFFFFFFFFF
FAMILIES = {
    "set.*":  ["set."],
    "bfind.*": ["bfind.", "bfind.shiftamt."],
    "bfe.*":  ["bfe."],
    "bfi.*":  ["bfi."],
    "prmt.*": ["prmt."],
    "shf.*":  ["shf."],
    "lop3.*": ["lop3."],
    "sad.*":  ["sad."],
    "mul24.*": ["mul24."],
    "mad.*":  ["mad."],
    "mul.hi.*": ["mul.hi."],
    "div.*":  ["div."],
    "rem.*":  ["rem."],
    "abs.*":  ["abs."],
    "neg.*":  ["neg."],
    "popc.*": ["popc."],
    "clz.*":  ["clz."],
    "brev.*": ["brev."],
    "bmsk.*": ["bmsk."],
    "szext.*": ["szext."],
    "fns.*":  ["fns."],
    "slct.*": ["slct."],
}

# ---------------------------------------------------------------------------
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

def run_gpu(prog, input_bytes, seed, opt, tag):
    (WORK/"t.cu").write_text(_emit_cuda(prog))
    (WORK/"in.bin").write_bytes(input_bytes)
    p = subprocess.run([NVCC,"-arch=sm_86","-cubin",f"-Xptxas=-O{opt}","-o",str(WORK/"k.cubin"),str(WORK/"t.cu")],
                       capture_output=True, text=True, timeout=240)
    if p.returncode: return None, f"COMPILE_FAIL: {p.stderr[-200:]}"
    r = subprocess.run([str(RUNNER),"--cubin",str(WORK/"k.cubin"),"--kernel","test_kernel",
      "--input",str(WORK/"in.bin"),"--output",str(WORK/f"{tag}.bin"),"--result",str(WORK/"e.json"),
      "--input-words","256","--output-words","128","--grid","1","--block","64",
      "--seed",str(seed),"--guards","8","--test-id","x","--timeout-ms","60000"],
      capture_output=True, text=True, timeout=120)
    if r.returncode: return None, f"RUN_FAIL: {r.stderr[-200:]}"
    data = (WORK/f"{tag}.bin").read_bytes()
    status = struct.unpack('<I', data[16:20])[0]
    if status != 0: return None, f"GPU_STATUS={status}"
    return data[40:-8], None

def disable_family(prog, prefix):
    c = copy.deepcopy(prog)
    for st in c.steps:
        if any(st["op"].startswith(p) for p in prefix):
            dk = st["dst"][0]
            if dk == "pred":
                st["op"] = "setp.eq.u32"
                st["srcs"] = st["srcs"][:2] if len(st["srcs"]) >= 2 else [("r32", 0), ("r32", 0)]
                st["imm"] = None
            else:
                st["op"] = "mov.u64" if dk == "r64" else "mov.b32"
                st["srcs"] = st["srcs"][:1] if st["srcs"] else [("r32", 0)]
                st["imm"] = None
    return c

def validate_case(case_dir: Path) -> dict:
    """Validate one case: O0/O1 match, determinism, family triggers."""
    case = case_from_dir(case_dir)
    seed = case.kernel_seed
    ib = case.input_bytes
    prog = _gen_program(seed, case.params)
    exp = oracle_for(prog, ib, seed)

    t0 = time.monotonic()
    o0, e0 = run_gpu(prog, ib, seed, 0, "o0")
    if o0 is None: return {"error": e0, "elapsed": time.monotonic()-t0}
    o1, e1 = run_gpu(prog, ib, seed, 1, "o1")
    if o1 is None: return {"error": e1, "elapsed": time.monotonic()-t0}
    # determinism: re-run O1
    o1b, e1b = run_gpu(prog, ib, seed, 1, "o1b")
    det = (o1b is not None) and (o1 == o1b)

    result = {
        "case_id": case.case_id(),
        "dir": case_dir.name,
        "seed": seed,
        "generator_version": case.generator_version,
        "n_instr": case.params.get("n_instr", "?"),
        "n_regs32": case.params.get("n_regs32", "?"),
        "coverage_mode": case.params.get("coverage_mode", "?"),
        "O0_match": o0 == exp,
        "O1_match": o1 == exp,
        "deterministic": det,
        "diverges": (o0 == exp) and (o1 != exp) and det,
        "elapsed": time.monotonic()-t0,
    }

    if not result["diverges"]:
        return result

    # family-disabling: which families are essential to the divergence?
    essential = []
    present_families = {k for k, v in FAMILIES.items()
                        if any(any(st["op"].startswith(p) for p in v) for st in prog.steps)}
    for fam_key in sorted(present_families):
        prefixes = FAMILIES[fam_key]
        c = disable_family(prog, prefixes)
        o1f, _ = run_gpu(c, ib, seed, 1, "o1f")
        if o1f is not None and o1f == exp:
            essential.append(fam_key)
    result["essential_families"] = essential

    # detailed op list for tagging the cluster
    sig = Counter(st["op"].split(".")[0] for st in prog.steps if st["op"] != "out")
    result["op_signature"] = dict(sig.most_common(20))
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    corpus = ROOT / "corpus" / "differentials"
    cases = sorted(d for d in corpus.iterdir() if (d/"manifest.json").exists())
    print(f"auditing {len(cases)} cases in {corpus}")
    results = []
    for i, d in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {d.name} ...", flush=True)
        try:
            r = validate_case(d)
        except Exception as e:
            r = {"case_id": d.name, "dir": d.name, "error": str(e)}
        results.append(r)
        # save per-case review.json update
        revf = d / "review.json"
        rev = {}
        if revf.exists():
            try: rev = json.loads(revf.read_text())
            except Exception: rev = {}
        rev.update({"classification_audit": r})
        revf.write_text(json.dumps(rev, indent=2, sort_keys=True))

        if "error" in r:
            print(f"  ERROR: {r['error']}", flush=True)
        else:
            print(f"  O0_match={r['O0_match']} O1_match={r['O1_match']} det={r['deterministic']} "
                  f"diverges={r.get('diverges')} essential={r.get('essential_families','?')}", flush=True)

    # ---- classification ----
    confirmed = [r for r in results if r.get("diverges")]
    pending  = [r for r in results if r.get("O0_match") is None]  # compile/run failure
    other    = [r for r in results if not r.get("diverges") and r.get("O0_match") is not None]
    print(f"\n=== summary: {len(results)} total, {len(confirmed)} confirmed, "
          f"{len(pending)} pending-recheck, {len(other)} other ===")

    # Cluster confirmed by essential_families
    clusters = {}
    for r in confirmed:
        key = "|".join(sorted(r.get("essential_families", [])))
        clusters.setdefault(key, []).append(r)

    if not confirmed:
        print("no confirmed cases to classify")
        return

    # write report
    report = ROOT / "corpus" / "differentials" / "CLASSIFICATION_REPORT.md"
    lines = [
        "# cuda_stress differential classification report",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total cases audited: {len(results)}",
        f"Confirmed ptxas bugs: {len(confirmed)}",
        f"Pending (compile/run failure): {len(pending)}",
        f"Other (no divergence or mismatch): {len(other)}",
        "",
        "---",
        "",
        "## Confirmed root-cause clusters",
        "",
    ]
    for i, (key, group) in enumerate(sorted(clusters.items(), key=lambda kv: -len(kv[1]))):
        lines.append(f"### Cluster {i+1}: `{key}`  ({len(group)} cases)")
        lines.append("")
        # detailed breakdown
        modes = Counter(r.get("coverage_mode", "?") for r in group)
        instructions = Counter(r.get("n_instr", "?") for r in group)
        lines.append(f"- coverage_mode: {dict(modes)}")
        lines.append(f"- n_instr distribution: {dict(instructions)}")
        lines.append(f"- example case_ids:")
        for r in group[:5]:
            lines.append(f"  - {r['dir']} (n_instr={r['n_instr']}, mode={r['coverage_mode']})")
        if len(group) > 5:
            lines.append(f"  - ... and {len(group)-5} more")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Individual case verdicts")
    lines.append("")
    lines.append("| case_id | O0_match | O1_match | det | diverges | essential |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (not x.get("diverges"), x["dir"])):
        lines.append(f"| {r['dir']} | {r.get('O0_match','?')} | {r.get('O1_match','?')} | "
                     f"{r.get('deterministic','?')} | {r.get('diverges','-')} | "
                     f"{r.get('essential_families','?')} |")

    report.write_text("\n".join(lines) + "\n")
    print(f"\nwritten {report}")

if __name__ == "__main__":
    main()
