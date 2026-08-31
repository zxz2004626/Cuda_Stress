"""Static (assemble-time) regression oracle.

Metrics per compiled kernel, compared lexicographically (higher = worse):

  CUDA: (spill_bytes, private_bytes, reg_count, isa_count)
    spill_bytes   = ptxas spill stores + loads         (nvcc -Xptxas -v)
    private_bytes = cuobjdump STACK + LOCAL            (-res-usage)
    reg_count     = registers used
    isa_count     = SASS instruction count             (cuobjdump -sass)

  ROCm: (private_bytes, reg_count, isa_count[, sgpr_count])
    private_bytes = private_segment_fixed_size         (llvm-readelf --notes)
    reg_count     = vgpr_count
    isa_count     = AMDGPU ISA instruction count       (llvm-objdump -d)

Oracle rules (per mutation edge parent->child):
  * opt polarity:   child worse than parent => violation
  * deopt polarity: child better than parent => anomaly
  * baseline:       new-compiler parent worse than old-compiler parent
  * edge delta:     (child-parent) larger on the new compiler => regression
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CudaMetrics:
    spill_bytes: int
    private_bytes: int
    reg_count: int
    isa_count: int

    def key(self) -> tuple:
        return (self.spill_bytes, self.private_bytes, self.reg_count,
                self.isa_count)

    def rate(self) -> int:
        # Lexicographic scalar: each field's max must stay below one unit of
        # the field above it.  spill gets 10**18 (not 10**12) so private_bytes
        # up to ~1GB cannot overflow into the spill slot -- kernels with local
        # arrays routinely exceed the old 1000-byte headroom, which silently
        # inverted spill-vs-private comparisons in edge_delta / verify_regression.
        return (10**18 * self.spill_bytes + 10**9 * self.private_bytes
                + 10**6 * self.reg_count + self.isa_count)


@dataclass(frozen=True)
class RocmMetrics:
    private_bytes: int
    reg_count: int
    isa_count: int
    sgpr_count: int = 0

    def key(self) -> tuple:
        return (self.private_bytes, self.reg_count, self.isa_count)

    def rate(self) -> int:
        return (10**9 * self.private_bytes + 10**6 * self.reg_count
                + self.isa_count)


Metrics = Union[CudaMetrics, RocmMetrics]


def kx_cmp_metrics(a: Metrics, b: Metrics) -> int:
    ka, kb = a.key(), b.key()
    return 1 if ka > kb else (-1 if ka < kb else 0)


# ---------------------------------------------------------------------------
# Text parsers
# ---------------------------------------------------------------------------

_RE_SPILL = re.compile(r"(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads")
_RE_USED_REGS = re.compile(r"Used\s+(\d+)\s+registers", re.IGNORECASE)
_RE_RES_REG = re.compile(r"\bREG:(\d+)")
_RE_RES_STACK = re.compile(r"\bSTACK:(\d+)")
_RE_RES_LOCAL = re.compile(r"\bLOCAL:(\d+)")
_RE_SASS_ADDR = re.compile(r"/\*[0-9a-fA-F]+\*/")

_RE_PRIVATE = re.compile(r"\.private_segment_fixed_size:\s*(\d+)")
_RE_VGPR = re.compile(r"\.vgpr_count:\s*(\d+)")
_RE_SGPR = re.compile(r"\.sgpr_count:\s*(\d+)")
_RE_ISA_LINE = re.compile(r"//\s*[0-9a-fA-F]+:")


def read_cuda_assemble_log(text: str) -> tuple:
    """(spill_stores, spill_loads, used_regs) from nvcc -Xptxas -v output."""
    spill_stores = spill_loads = used = 0
    m = _RE_SPILL.search(text)
    if m:
        spill_stores, spill_loads = int(m.group(1)), int(m.group(2))
    m = _RE_USED_REGS.search(text)
    if m:
        used = int(m.group(1))
    return spill_stores, spill_loads, used


def read_cuda_res_usage(text: str) -> tuple:
    """(reg, stack, local) from cuobjdump -res-usage output."""
    reg = stack = local = 0
    m = _RE_RES_REG.search(text)
    if m:
        reg = int(m.group(1))
    m = _RE_RES_STACK.search(text)
    if m:
        stack = int(m.group(1))
    m = _RE_RES_LOCAL.search(text)
    if m:
        local = int(m.group(1))
    return reg, stack, local


def parse_sass_count(text: str) -> int:
    """Count instruction lines (/*addr*/) in cuobjdump -sass output."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/*") and _RE_SASS_ADDR.match(stripped):
            if not stripped.endswith(".........."):
                count += 1
    return count


def gather_cuda_metrics(compile_log: str, res_usage: str,
                         sass_text: str) -> CudaMetrics:
    ss, sl, regs_log = read_cuda_assemble_log(compile_log)
    reg_res, stack, local = read_cuda_res_usage(res_usage)
    return CudaMetrics(
        spill_bytes=ss + sl,
        private_bytes=stack + local,
        reg_count=reg_res if reg_res else regs_log,
        isa_count=parse_sass_count(sass_text),
    )


def read_rocm_notes(text: str) -> dict:
    """private_segment_fixed_size / vgpr_count / sgpr_count (max over kernels)."""
    out = {"private_bytes": 0, "vgpr_count": 0, "sgpr_count": 0}
    for key, rx in (("private_bytes", _RE_PRIVATE),
                    ("vgpr_count", _RE_VGPR), ("sgpr_count", _RE_SGPR)):
        vals = rx.findall(text)
        if vals:
            out[key] = max(int(v) for v in vals)
    return out


def parse_isa_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if line and _RE_ISA_LINE.search(line) and "s_code_end" not in line:
            count += 1
    return count


def gather_rocm_metrics(notes_text: str, isa_text: str) -> RocmMetrics:
    notes = read_rocm_notes(notes_text)
    return RocmMetrics(
        private_bytes=notes["private_bytes"],
        reg_count=notes["vgpr_count"],
        isa_count=parse_isa_count(isa_text),
        sgpr_count=notes["sgpr_count"],
    )


# ---------------------------------------------------------------------------
# Compilation + extraction
# ---------------------------------------------------------------------------

def cuda_metrics_for_source(source: str, nvcc_path: str, arch: str,
                            work_dir: Optional[str] = None,
                            timeout: int = 120,
                            opt_level: str = "-O3") -> Optional[CudaMetrics]:
    """Compile to a cubin and extract metrics via nvcc + cuobjdump."""
    own_dir = work_dir is None
    if own_dir:
        work_dir = tempfile.mkdtemp(prefix="gpuperf_cuda_metrics_")
    try:
        os.makedirs(work_dir, exist_ok=True)
        src = os.path.join(work_dir, "kernel.cu")
        cubin = os.path.join(work_dir, "kernel.cubin")
        Path(src).write_text(source)

        cmd = [nvcc_path, f"-arch={arch}", opt_level, "-std=c++17", "--cubin",
               "--resource-usage", "-Xptxas=-v", src, "-o", cubin]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if proc.returncode != 0 or not os.path.isfile(cubin):
            return None
        compile_log = proc.stderr

        cuobjdump = os.path.join(os.path.dirname(nvcc_path), "cuobjdump")
        if not os.path.isfile(cuobjdump):
            cuobjdump = "cuobjdump"
        res_usage = ""
        sass = ""
        try:
            rp = subprocess.run([cuobjdump, "-res-usage", cubin],
                                capture_output=True, text=True, timeout=30)
            if rp.returncode == 0:
                res_usage = rp.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        try:
            sp = subprocess.run([cuobjdump, "-sass", cubin],
                                capture_output=True, text=True, timeout=60)
            if sp.returncode == 0:
                sass = sp.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return gather_cuda_metrics(compile_log, res_usage, sass)
    finally:
        if own_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _kx_find_llvm_tool(name: str, rocm_root: Optional[str]) -> Optional[str]:
    if rocm_root:
        p = os.path.join(rocm_root, "llvm", "bin", name)
        if os.path.isfile(p):
            return p
    for base in ("/opt/rocm/llvm/bin", "/opt/rocm-7.2.1/llvm/bin",
                 "/opt/rocm-6.4.0/llvm/bin", "/opt/rocm-6.3.0/llvm/bin"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None


def _pull_device_elf(fatbin_path: str, output_path: str) -> bool:
    """Extract the device ELF from a HIP fatbin section dump.

    The fatbin holds a CLANG_OFFLOAD_BUNDLE header followed by host and
    device code; the device ELF starts at a 4096-byte boundary (checked
    first), then any 4K boundary, then anywhere byte-wise.
    """
    try:
        with open(fatbin_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        log.debug("failed to read fatbin: %s", exc)
        return False
    elf_magic = b"\x7fELF"
    offsets = []
    if len(data) > 4100:
        offsets.append(4096)
    offsets.extend(range(0, len(data) - 4, 4096))
    idx = data.find(elf_magic)
    if idx >= 0:
        offsets.append(idx)
    for off in offsets:
        if data[off:off + 4] == elf_magic:
            try:
                with open(output_path, "wb") as f:
                    f.write(data[off:])
                return True
            except OSError as exc:
                log.debug("failed to write device ELF: %s", exc)
                return False
    log.debug("no ELF magic found in fatbin")
    return False


def _rocm_metrics_via_fatbin(hipcc_path: str, src: str, arch: str,
                             work_dir: str, timeout: int = 120,
                             opt_level: str = "-O3") -> Optional[RocmMetrics]:
    """Fallback extraction: assemble to an object file, dump the
    ``.hip_fatbin`` section with llvm-objcopy, extract the device ELF."""
    rocm_root = os.path.dirname(os.path.dirname(hipcc_path))
    readelf = _kx_find_llvm_tool("llvm-readelf", rocm_root)
    objdump = _kx_find_llvm_tool("llvm-objdump", rocm_root)
    objcopy = _kx_find_llvm_tool("llvm-objcopy", rocm_root)
    if not readelf or not objdump or not objcopy:
        return None
    obj = os.path.join(work_dir, "kernel.o")
    cmd = [hipcc_path, opt_level, "-std=c++17", f"--offload-arch={arch}",
           "-c", src, "-o", obj]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0 or not os.path.isfile(obj):
        return None
    fatbin = os.path.join(work_dir, "fatbin.bin")
    try:
        proc = subprocess.run(
            [objcopy, f"--dump-section=.hip_fatbin={fatbin}", obj],
            capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0 or not os.path.isfile(fatbin):
        return None
    device_elf = os.path.join(work_dir, "device.elf")
    if not _pull_device_elf(fatbin, device_elf):
        return None
    try:
        notes = subprocess.run([readelf, "--notes", device_elf],
                               capture_output=True, text=True, timeout=30).stdout
        isa = subprocess.run([objdump, "-d", device_elf],
                             capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return gather_rocm_metrics(notes, isa)


def rocm_metrics_for_source(source: str, hipcc_path: str, arch: str,
                            work_dir: Optional[str] = None,
                            timeout: int = 120,
                            opt_level: str = "-O3") -> Optional[RocmMetrics]:
    """Compile a HIP source and extract metrics from the embedded code object."""
    own_dir = work_dir is None
    if own_dir:
        work_dir = tempfile.mkdtemp(prefix="gpuperf_rocm_metrics_")
    try:
        os.makedirs(work_dir, exist_ok=True)
        src = os.path.join(work_dir, "kernel.hip")
        exe = os.path.join(work_dir, "kernel.out")
        Path(src).write_text(source)

        rocm_root = os.path.dirname(os.path.dirname(hipcc_path))
        readelf = _kx_find_llvm_tool("llvm-readelf", rocm_root)
        objdump = _kx_find_llvm_tool("llvm-objdump", rocm_root)
        if not readelf or not objdump:
            log.debug("LLVM tools not found under %s", rocm_root)
            return None

        cmd = [hipcc_path, opt_level, "-std=c++17", f"--offload-arch={arch}",
               src, "-o", exe]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if proc.returncode != 0:
            return None

        # Extract code objects next to the executable.
        try:
            subprocess.run([objdump, "--offloading", exe],
                           capture_output=True, text=True, timeout=30,
                           cwd=work_dir)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        gpu_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir)
                     if "gfx" in f and not f.endswith(".hip")]
        if not gpu_files:
            # Fatbin-extraction fallback (SPEC: llvm-objcopy + ELF search).
            return _rocm_metrics_via_fatbin(hipcc_path, src, arch, work_dir,
                                            opt_level=opt_level)
        gpu_file = gpu_files[0]

        try:
            notes = subprocess.run([readelf, "--notes", gpu_file],
                                   capture_output=True, text=True,
                                   timeout=30).stdout
            isa = subprocess.run([objdump, "-d", gpu_file],
                                 capture_output=True, text=True,
                                 timeout=60).stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return gather_rocm_metrics(notes, isa)
    finally:
        if own_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def metrics_for_source(source: str, compiler_path: str, backend: str,
                       arch: str, work_dir: Optional[str] = None,
                       timeout: int = 120,
                       opt_level: str = "-O3") -> Optional[Metrics]:
    if backend == "cuda":
        return cuda_metrics_for_source(source, compiler_path, arch,
                                       work_dir=work_dir, timeout=timeout,
                                       opt_level=opt_level)
    return rocm_metrics_for_source(source, compiler_path, arch,
                                   work_dir=work_dir, timeout=timeout,
                                   opt_level=opt_level)


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

@dataclass
class StaticVerdict:
    seed: int
    phase: str                    # baseline | opt | deopt | pass
    is_regression: bool
    reason: str
    violation_type: str = "none"  # direct_parent | direct_child | edge_delta | polarity
    score_delta: int = 0


class StaticOracle:
    """Lexicographic metric comparison per mutation edge."""

    def __init__(self, vendor: str = "cuda") -> None:
        self.vendor = vendor.lower()

    def classify_edge(self, parent: Metrics, child: Metrics,
                      polarity: str, seed: int = 0) -> StaticVerdict:
        """Single-compiler edge classification."""
        relation = kx_cmp_metrics(child, parent)
        delta = child.rate() - parent.rate()
        if polarity == "opt":
            if relation > 0:
                return StaticVerdict(seed, "opt", True,
                                     "opt mutation made code worse",
                                     "polarity", delta)
            return StaticVerdict(seed, "opt", False,
                                 "opt mutation improved/unchanged code", "none", delta)
        else:
            if relation < 0:
                return StaticVerdict(seed, "deopt", True,
                                     "deopt mutation unexpectedly improved code",
                                     "polarity", delta)
            return StaticVerdict(seed, "deopt", False,
                                 "deopt mutation worsened/unchanged code", "none", delta)

    def compare_edge(self, old_parent: Metrics, old_child: Metrics,
                     new_parent: Metrics, new_child: Metrics,
                     polarity: str, seed: int = 0) -> List[StaticVerdict]:
        """Cross-compiler comparison over one mutation edge."""
        verdicts: List[StaticVerdict] = []
        phase = "opt" if polarity == "opt" else "deopt"

        if kx_cmp_metrics(new_parent, old_parent) > 0:
            verdicts.append(StaticVerdict(
                seed, "baseline", True,
                f"direct regression on original: {old_parent} -> {new_parent}",
                "direct_parent",
                new_parent.rate() - old_parent.rate()))
        if kx_cmp_metrics(new_child, old_child) > 0:
            verdicts.append(StaticVerdict(
                seed, phase, True,
                f"direct regression on mutated: {old_child} -> {new_child}",
                "direct_child",
                new_child.rate() - old_child.rate()))

        d_old = old_child.rate() - old_parent.rate()
        d_new = new_child.rate() - new_parent.rate()
        if d_new > d_old:
            verdicts.append(StaticVerdict(
                seed, phase, True,
                f"edge delta regression: delta_old={d_old} delta_new={d_new}",
                "edge_delta", d_new - d_old))
        return verdicts

    def full_verify(self, src_old: Optional[Metrics], src_new: Optional[Metrics],
                   opt_old: Optional[Metrics] = None,
                   opt_new: Optional[Metrics] = None,
                   deopt_old: Optional[Metrics] = None,
                   deopt_new: Optional[Metrics] = None,
                   seed: int = 0) -> List[StaticVerdict]:
        out: List[StaticVerdict] = []
        if src_old is not None and src_new is not None:
            if kx_cmp_metrics(src_new, src_old) > 0:
                out.append(StaticVerdict(
                    seed, "baseline", True,
                    f"new compiler produces worse code: {src_old} -> {src_new}",
                    "direct_parent",
                    src_new.rate() - src_old.rate()))
            if opt_old is not None and opt_new is not None:
                out.extend(self.compare_edge(src_old, opt_old, src_new, opt_new,
                                             "opt", seed))
            if deopt_old is not None and deopt_new is not None:
                out.extend(self.compare_edge(src_old, deopt_old, src_new,
                                             deopt_new, "deopt", seed))
        return out

    def diff_compilers(self, old_parent: Metrics, old_child: Metrics,
                          new_parent: Metrics, new_child: Metrics) -> Dict[str, Any]:
        """Compare old vs new compiler on one mutation edge (dict form)."""
        ops, ocs = old_parent.rate(), old_child.rate()
        nps, ncs = new_parent.rate(), new_child.rate()
        edge_delta_old = ocs - ops
        edge_delta_new = ncs - nps
        return {
            "direct_regression_on_parent": nps > ops,
            "direct_regression_on_child": ncs > ocs,
            "edge_delta_old": edge_delta_old,
            "edge_delta_new": edge_delta_new,
            "new_worse_on_edge": edge_delta_new > edge_delta_old,
            "old_parent_score": ops, "old_child_score": ocs,
            "new_parent_score": nps, "new_child_score": ncs,
        }

    def verify_regression(self, polarity: str,
                         old_parent: Metrics, old_child: Metrics,
                         new_parent: Metrics, new_child: Metrics,
                         seed: int = 0) -> StaticVerdict:
        """Combined check: direct parent/child regressions, edge-delta
        regression, and a single-compiler polarity cross-check."""
        comp = self.diff_compilers(old_parent, old_child,
                                      new_parent, new_child)
        is_regression = False
        violation_type = "none"
        details: List[str] = []
        if comp["direct_regression_on_parent"]:
            is_regression = True
            violation_type = "direct_parent"
            details.append(
                f"direct regression on parent: "
                f"old={comp['old_parent_score']} -> new={comp['new_parent_score']}")
        if comp["direct_regression_on_child"]:
            is_regression = True
            if violation_type == "none":
                violation_type = "direct_child"
            details.append(
                f"direct regression on child: "
                f"old={comp['old_child_score']} -> new={comp['new_child_score']}")
        if comp["new_worse_on_edge"]:
            is_regression = True
            if violation_type == "none":
                violation_type = "edge_delta"
            details.append(
                f"edge delta regression: old_delta={comp['edge_delta_old']}, "
                f"new_delta={comp['edge_delta_new']}")
        classification = self.classify_edge(new_parent, new_child, polarity, seed)
        if classification.is_regression:
            is_regression = True
            details.append(f"polarity violation: {classification.reason}")
        severity = max(
            max(0, comp["new_parent_score"] - comp["old_parent_score"]),
            max(0, comp["new_child_score"] - comp["old_child_score"]))
        reason = details[0] if details else "no regression detected"
        return StaticVerdict(seed, polarity, is_regression, reason,
                             violation_type, severity)


class MutatorPolarity:
    """Mutation-direction vocabulary for the static oracle."""
    OPT = "opt"
    DEOPT = "deopt"


def kx_cuda_key(m: Metrics) -> tuple:
    return m.key()


def cuda_rate(m: Metrics) -> int:
    return m.rate()


def kx_rocm_key(m: Metrics) -> tuple:
    return m.key()


def rocm_rate(m: Metrics) -> int:
    return m.rate()


def batch_verify(oracle: StaticOracle,
                metrics_pairs: List[Dict[str, Any]]) -> List[StaticVerdict]:
    """Run oracle checks on a batch of metric-pair dicts."""
    results = []
    for pair in metrics_pairs:
        results.append(oracle.verify_regression(
            polarity=pair["polarity"],
            old_parent=pair["old_parent"], old_child=pair["old_child"],
            new_parent=pair["new_parent"], new_child=pair["new_child"],
            seed=pair.get("seed", 0)))
    return results


def kx_filter_regressions(verdicts: List[StaticVerdict]) -> List[StaticVerdict]:
    """Keep only regression verdicts."""
    return [v for v in verdicts if v.is_regression]


def grade_regressions(verdicts: List[StaticVerdict]) -> List[StaticVerdict]:
    """Rank regressions by severity (larger rate delta = worse)."""
    return sorted(kx_filter_regressions(verdicts),
                  key=lambda v: v.score_delta, reverse=True)


def execute_static_oracle(program, kx_translator, old_compiler, new_compiler,
                      mutation_pipeline, seed: int, work_dir: str,
                      polarity: str = "opt") -> StaticVerdict:
    """One-kernel static oracle: rendersrc, perturb, extract 4x metrics, check."""
    os.makedirs(work_dir, exist_ok=True)
    backend = "cuda" if old_compiler.extension == ".cu" else "hip"
    oracle = StaticOracle(backend)

    parent_source = kx_translator.rendersrc(program)
    if polarity == MutatorPolarity.OPT:
        child_program = mutation_pipeline.perturb_opt_random(program, num_mutations=1)
    else:
        child_program = mutation_pipeline.perturb_deopt_random(program, num_mutations=1)
    child_source = kx_translator.rendersrc(child_program)

    def _extract(compiler, source: str, label: str) -> Optional[Metrics]:
        return metrics_for_source(
            source, compiler.compiler_path, backend, compiler.isa,
            work_dir=os.path.join(work_dir, label))

    old_parent = _extract(old_compiler, parent_source, "old_parent")
    old_child = _extract(old_compiler, child_source, "old_child")
    new_parent = _extract(new_compiler, parent_source, "new_parent")
    new_child = _extract(new_compiler, child_source, "new_child")
    if None in (old_parent, old_child, new_parent, new_child):
        return StaticVerdict(seed, polarity, False,
                             "metrics extraction failed on one or more artifacts",
                             "none", 0)
    return oracle.verify_regression(
        polarity=polarity,
        old_parent=old_parent, old_child=old_child,
        new_parent=new_parent, new_child=new_child, seed=seed)


# ---------------------------------------------------------------------------
# Static pipeline (no execution)
# ---------------------------------------------------------------------------

@dataclass
class StaticCaseResult:
    seed: int
    template: str
    success: bool = True
    phase: str = "pass"
    src_old: Optional[Metrics] = None
    src_new: Optional[Metrics] = None
    opt_old: Optional[Metrics] = None
    opt_new: Optional[Metrics] = None
    deopt_old: Optional[Metrics] = None
    deopt_new: Optional[Metrics] = None
    verdicts: List[StaticVerdict] = field(default_factory=list)
    opt_mutator: str = ""
    deopt_mutator: str = ""
    errors: List[str] = field(default_factory=list)


def execute_static_pipeline(program, seed: int, template: str, kx_translator,
                        old_compiler, new_compiler, mutation_pipeline,
                        output_dir: Optional[str] = None,
                        opt_level: str = "-O3",
                        mutation_mode: str = "full") -> StaticCaseResult:
    """Compile-only cross-compiler comparison (6 metric extractions)."""
    result = StaticCaseResult(seed=seed, template=template)
    backend = "cuda" if old_compiler.extension == ".cu" else "hip"
    work_dir = output_dir or tempfile.mkdtemp(prefix="gpuperf_static_")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    try:
        src_code = kx_translator.rendersrc(program)
    except Exception as exc:
        result.errors.append(f"translation failed: {exc}")
        return result

    def _mutate(direction: str):
        try:
            if mutation_mode != "full" and hasattr(mutation_pipeline, "perturb_by_mode"):
                prog, applied = mutation_pipeline.perturb_by_mode(
                    program, mode=mutation_mode, num_mutations=1, paper_only=True)
                name = applied[0] if applied else "none"
            elif hasattr(mutation_pipeline, f"perturb_{direction}_by_mode"):
                prog, applied = getattr(mutation_pipeline,
                                        f"perturb_{direction}_by_mode")(
                    program, mode=mutation_mode, num_mutations=1)
                name = applied[0] if applied else "none"
            else:
                fn = getattr(mutation_pipeline, f"perturb_{direction}_random")
                prog = fn(program, paper_only=True)
                name = f"random_{direction}"
            return kx_translator.rendersrc(prog), name
        except Exception as exc:
            result.errors.append(f"{direction} mutation failed: {exc}")
            return None, ""

    opt_code, opt_name = _mutate("opt")
    deopt_code, deopt_name = _mutate("deopt")
    result.opt_mutator = opt_name
    result.deopt_mutator = deopt_name

    def collect(code: Optional[str], compiler, label: str) -> Optional[Metrics]:
        if code is None:
            return None
        return metrics_for_source(code, compiler.compiler_path, backend,
                                  compiler.isa,
                                  work_dir=os.path.join(work_dir, label),
                                  opt_level=opt_level)

    result.src_old = collect(src_code, old_compiler, "src_old")
    result.src_new = collect(src_code, new_compiler, "src_new")
    result.opt_old = collect(opt_code, old_compiler, "opt_old")
    result.opt_new = collect(opt_code, new_compiler, "opt_new")
    result.deopt_old = collect(deopt_code, old_compiler, "deopt_old")
    result.deopt_new = collect(deopt_code, new_compiler, "deopt_new")

    if result.src_old is None or result.src_new is None:
        result.errors.append("baseline compilation/metrics failed")
        return result

    oracle = StaticOracle(backend)
    verdicts = oracle.full_verify(
        result.src_old, result.src_new,
        result.opt_old, result.opt_new,
        result.deopt_old, result.deopt_new, seed=seed)
    result.verdicts = verdicts
    if verdicts:
        result.success = False
        for v in verdicts:
            if v.phase == "baseline":
                result.phase = "baseline"
                break
            elif v.phase == "opt":
                result.phase = "opt"
            elif v.phase == "deopt" and result.phase != "opt":
                result.phase = "deopt"

    if not result.success and output_dir is not None:
        ext = old_compiler.extension
        case_dir = Path(output_dir) / f"static_seed_{seed}_{result.phase}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"kernel_src{ext}").write_text(src_code)
        if opt_code:
            (case_dir / f"kernel_opt{ext}").write_text(opt_code)
        if deopt_code:
            (case_dir / f"kernel_deopt{ext}").write_text(deopt_code)
        report = {
            "seed": seed, "template": template, "phase": result.phase,
            "opt_mutator": opt_name, "deopt_mutator": deopt_name,
            "metrics": {k: repr(v) for k, v in (
                ("src_old", result.src_old), ("src_new", result.src_new),
                ("opt_old", result.opt_old), ("opt_new", result.opt_new),
                ("deopt_old", result.deopt_old), ("deopt_new", result.deopt_new))},
            "verdicts": [v.reason for v in verdicts],
            "errors": result.errors,
        }
        (case_dir / "result.json").write_text(json.dumps(report, indent=2))
        log.info("static regression case saved: %s", case_dir)
    return result
