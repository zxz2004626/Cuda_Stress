#!/usr/bin/env bash
# Drive bisect_minimal.py over the remaining bug-report cases, then verify each.
set -uo pipefail
ROOT=/home/linux/cuda_stress
cd "$ROOT"
CASES="inline_ptx__mad24 inline_ptx__mul inline_ptx__set inline_ptx__shf inline_ptx_prmt"

for c in $CASES; do
  d="corpus/differentials/bug_reports/$c"
  echo "================================================================"
  echo "CASE: $c"
  echo "================================================================"
  python3 tools/bisect_minimal.py "$d" 2>&1 | tail -20
  echo "--- verifying ---"
  python3 - "$d" <<'PYEOF'
import subprocess, tempfile, struct, hashlib, sys, json
from pathlib import Path
d = Path(sys.argv[1])
manifest = json.loads((d/"manifest.json").read_text())
seed = manifest["kernel_seed"]
RUNNER="/home/linux/cuda_stress/native/cuda_runner"
BLOB=struct.Struct("<QQIIQQ")
def run(nvcc, opt):
    with tempfile.TemporaryDirectory() as tmp:
        t=Path(tmp); c=t/"k.cubin"; o=t/"out.bin"
        r=subprocess.run([nvcc,"-arch=sm_86","-cubin",f"-Xptxas=-O{opt}","-o",str(c),str(d/"test.cu")],capture_output=True,text=True)
        if r.returncode!=0: return None
        r=subprocess.run([RUNNER,"--cubin",str(c),"--kernel","test_kernel","--input",str(d/"input.bin"),"--output",str(o),"--result",str(t/"r.json"),"--input-words","256","--output-words","128","--grid","1","--block","64","--seed",str(seed),"--guards","8","--test-id","x","--timeout-ms","60000"],capture_output=True,text=True,timeout=70)
        if r.returncode!=0 or not o.exists(): return None
        return hashlib.sha256(o.read_bytes()[BLOB.size:-8]).hexdigest()[:12]
ok=True
for lbl,nvcc in [("11.3","/usr/local/cuda-11.3/bin/nvcc"),("11.8","/usr/local/cuda-11.8/bin/nvcc"),("12.6","/usr/local/cuda-12.6/bin/nvcc")]:
    h0=run(nvcc,0); h1=run(nvcc,1)
    trig = h0 and h1 and h0!=h1
    if not trig: ok=False
    print(f"  {lbl}: O0={h0} O1={h1} {'OK' if trig else 'FAIL'}")
print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
PYEOF
done
echo "ALL DONE"
