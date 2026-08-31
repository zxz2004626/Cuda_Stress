"""Interestingness oracle for the IR reducer.

Interesting(P) means:
  * valid_ub(P) passes,
  * P emits a checksum harness,
  * cuda_driver_spec(P) succeeds,
  * both pinned builds compile and run through the Driver checksum executor,
  * their u64 checksums differ (same miscompile direction as the original),
  * optionally, the checksum is still input-sensitive.

Only the pinned failing/reference toolchain+profile pair is built -- never the
full matrix.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import shutil
import struct
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from generator.base import GeneratedCase
from generator_ir.ir.program import Program
from generator_ir.reduce.validity import validity_errors
from generator_ir.reduce.structure import structural_digest
from pipeline.config import Config, Toolchain, OptimizationProfile
from pipeline import compiler
from pipeline.executor import run_build


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""
    invalid: bool = False
    checksums: Dict[str, Optional[int]] = field(default_factory=dict)
    source_lines: int = 0
    digest: str = ""


class InterestingOracle:
    def __init__(self, config: Config,
                 failing_tc: Toolchain, failing_profile: OptimizationProfile,
                 reference_tc: Toolchain, reference_profile: OptimizationProfile,
                 original: Program,
                 timeout: int = 180,
                 compile_timeout: int = 120,
                 run_timeout: int = 15,
                 keep_work: bool = False,
                 work_root: Optional[Path] = None,
                 dry_run: bool = False,
                 check_input_salt: bool = True,
                 device: int = 0) -> None:
        self.config = config
        self.failing_tc = failing_tc
        self.failing_profile = failing_profile
        self.reference_tc = reference_tc
        self.reference_profile = reference_profile
        self.original = original
        self.timeout = timeout
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.keep_work = keep_work
        self.work_root = Path(work_root) if work_root is not None else None
        self.dry_run = dry_run
        self.check_input_salt = check_input_salt
        self.device = device
        self._cache: Dict[Tuple[str, int, bool], CheckResult] = {}
        self._ccache_dir = tempfile.TemporaryDirectory(prefix="ir_reduce_ccache_")
        self._ccache: "OrderedDict[Tuple[str, ...], Path]" = OrderedDict()
        self._ccache_max = 64

    # ------------------------------------------------------------------
    def check(self, program: Program, runs: int = 1,
              input_salt_check: Optional[bool] = None) -> CheckResult:
        """Run the pinned interestingness oracle for *program*.

        ``input_salt_check`` defaults to ``self.check_input_salt``.  Set it to
        False for the first cheap screening pass; the CLI then re-verifies
        accepted candidates with runs >= 2 and input-salt enabled.
        """
        if input_salt_check is None:
            input_salt_check = self.check_input_salt
        digest = structural_digest(program)
        cache_key = (digest, int(runs), bool(input_salt_check))
        if not self.dry_run and cache_key in self._cache:
            cached = self._cache[cache_key]
            return CheckResult(cached.ok, cached.reason, cached.invalid,
                               dict(cached.checksums), cached.source_lines, digest)

        from generator_ir.emit.nvidia import CudaEmitter
        from generator_ir.emit.host_harness import cuda_driver_spec

        errs = validity_errors(program, original=self.original, emit_check=True)
        if errs:
            return CheckResult(False, f"invalid: {'; '.join(errs[:5])}", True,
                               {}, 0, digest)

        try:
            src = CudaEmitter().rendersrc(program)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(False, f"emit failed: {exc}", True, {}, 0, digest)
        if "checksum=" not in src or "int main(" not in src:
            return CheckResult(False, "emitted source lacks checksum harness",
                               True, {}, src.count("\n") + 1, digest)

        kernel = program.fetch_kernel()
        if kernel is None:
            return CheckResult(False, "program has no kernel", True, {},
                               src.count("\n") + 1, digest)
        try:
            spec = cuda_driver_spec(program, kernel)
        except Exception as exc:  # noqa: BLE001
            return CheckResult(False, f"cuda_driver_spec failed: {exc}", True,
                               {}, src.count("\n") + 1, digest)

        if self.dry_run:
            return CheckResult(True, "dry-run: static UB gate passed", False,
                               {}, src.count("\n") + 1, digest)

        source_lines = src.count("\n") + 1

        # Build a throwaway GeneratedCase with the refreshed spec.
        case = GeneratedCase(
            test_id="ir_reduce",
            config_seed=0,
            test_seed=0,
            strategy="generator_ir",
            params={},
            generator_version="ir_reduce",
            kernel_seed=0,
            source=src,
            input_bytes=b"",
            expected_bytes=b"",
            grid=list(program.launch.grid),
            block=list(program.launch.block),
            input_words=0,
            output_words=1,
            output_words_per_thread=0,
            target_sm=program.launch.arch,
            cpu_oracle="cross_build_consensus",
            comparison_policy="cross_build_checksum_u64",
            cuda_features=[program.launch.arch, "generator_ir", "driver_checksum"],
            kernel_entry=kernel.name,
            sanitizer_policy="driver_checksum_harness",
            execution_mode="driver_checksum",
            oracle_mode="cross_build",
            execution_spec=spec,
        )

        with self._work_dir() as work:
            work = Path(work)
            src_path = work / "test.cu"
            src_path.write_text(src)

            failing_dir = work / "failing"
            reference_dir = work / "reference"
            failing_checksums, fail_err = self._build_and_run(
                case, src_path, self.failing_tc, self.failing_profile,
                failing_dir, digest, runs=runs, run_index_base=0)
            if fail_err:
                return CheckResult(False, f"failing build failed: {fail_err}",
                                   False, {}, source_lines, digest)

            reference_checksums, ref_err = self._build_and_run(
                case, src_path, self.reference_tc, self.reference_profile,
                reference_dir, digest, runs=runs, run_index_base=100)
            if ref_err:
                return CheckResult(False, f"reference build failed: {ref_err}",
                                   False, {}, source_lines, digest)

            if not failing_checksums or not reference_checksums:
                return CheckResult(False, "no checksums produced", False, {},
                                   source_lines, digest)

            f_vals = set(failing_checksums)
            r_vals = set(reference_checksums)
            if len(f_vals) != 1 or len(r_vals) != 1:
                return CheckResult(False,
                                   f"unstable checksums failing={failing_checksums} "
                                   f"reference={reference_checksums}",
                                   False, {}, source_lines, digest)
            f = failing_checksums[0]
            r = reference_checksums[0]
            checksums = {"failing": f, "reference": r}
            if f == r:
                return CheckResult(False, f"checksums converged: {f} == {r}",
                                   False, checksums, source_lines, digest)

            if input_salt_check:
                salted_dir = work / "reference_salted"
                salted_checksums, salt_err = self._build_and_run(
                    case, src_path, self.reference_tc, self.reference_profile,
                    salted_dir, digest, runs=1, run_index_base=200, input_salt=1)
                if salt_err:
                    return CheckResult(False, f"input-salt run failed: {salt_err}",
                                       False, checksums, source_lines, digest)
                if salted_checksums and salted_checksums[0] == r:
                    return CheckResult(
                        False, f"checksum is not input-sensitive (constant {r})",
                        False, checksums, source_lines, digest)

            result = CheckResult(True, "checksums diverge", False, checksums,
                                 source_lines, digest)
            self._cache[cache_key] = result
            return result

    # ------------------------------------------------------------------
    def _get_or_compile_ptx(self, src_path: Path, src_digest: str,
                            tc: Toolchain) -> Tuple[Optional[Path], str]:
        key = ("ptx", src_digest, tc.id)
        if key in self._ccache:
            path = self._ccache[key]
            self._ccache.move_to_end(key)
            if path.exists():
                return path, ""
            self._ccache.pop(key, None)
        ccache_dir = Path(self._ccache_dir.name)
        ptx_path = ccache_dir / f"ptx_{src_digest[:16]}_{tc.id}.ptx"
        rec = compiler.nvcc_ptx(tc, src_path, ptx_path, self.compile_timeout)
        if not rec.ok:
            return None, f"nvcc -ptx failed: {rec.stderr[-400:]}"
        self._ccache_put(key, ptx_path)
        return ptx_path, ""

    def _get_or_compile_cubin(self, tc: Toolchain, profile: OptimizationProfile,
                              ptx_path: Path, src_digest: str) -> Tuple[Optional[Path], str]:
        key = ("cubin", src_digest, tc.id, profile.name)
        if key in self._ccache:
            path = self._ccache[key]
            self._ccache.move_to_end(key)
            if path.exists():
                return path, ""
            self._ccache.pop(key, None)
        ccache_dir = Path(self._ccache_dir.name)
        cubin_path = ccache_dir / (
            f"cubin_{src_digest[:16]}_{tc.id}_O{profile.ptxas_opt}.cubin")
        rec = compiler.ptxas_cubin(tc, profile, ptx_path, cubin_path,
                                   timeout=self.compile_timeout)
        if not rec.ok:
            return None, f"ptxas -O{profile.ptxas_opt} failed: {rec.stderr[-400:]}"
        self._ccache_put(key, cubin_path)
        return cubin_path, ""

    def _ccache_put(self, key, path: Path) -> None:
        self._ccache[key] = path
        self._ccache.move_to_end(key)
        while len(self._ccache) > self._ccache_max:
            old_key, old_path = self._ccache.popitem(last=False)
            try:
                if old_path.exists():
                    old_path.unlink()
            except OSError:
                pass

    def _build_and_run(self, case: GeneratedCase, src_path: Path,
                       tc: Toolchain, profile: OptimizationProfile,
                       build_dir: Path, src_digest: str,
                       runs: int, run_index_base: int,
                       input_salt: int = 0) -> tuple:
        build_dir.mkdir(parents=True, exist_ok=True)
        ptx_path, ptx_err = self._get_or_compile_ptx(src_path, src_digest, tc)
        if ptx_path is None:
            return [], ptx_err
        cubin_path, cubin_err = self._get_or_compile_cubin(
            tc, profile, ptx_path, src_digest)
        if cubin_path is None:
            return [], cubin_err
        local_cubin = build_dir / "kernel.cubin"
        if cubin_path != local_cubin:
            shutil.copyfile(cubin_path, local_cubin)
            cubin_path = local_cubin

        checksums: list = []
        for i in range(runs):
            run_index = run_index_base + i
            try:
                result = run_build(self.config, tc, case, build_dir, run_index,
                                   device=self.device, timeout=self.run_timeout,
                                   input_salt=input_salt)
            except OSError as exc:
                return [], f"driver_checksum runner unavailable: {exc}"
            if result.status != "OK":
                return [], (f"driver_checksum run failed ({result.status}): "
                            f"{result.error[-400:]}")
            if not result.output_words or result.meaningful_words != 1:
                return [], "driver_checksum did not produce exactly one u64"
            checksums.append(struct.unpack("<Q", result.output_words)[0])
        return checksums, ""

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _work_dir(self):
        if self.keep_work and self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)
            d = Path(tempfile.mkdtemp(prefix="cand_", dir=str(self.work_root)))
            try:
                yield d
            finally:
                # keep_work deliberately leaves intermediate artifacts on disk.
                pass
        else:
            with tempfile.TemporaryDirectory(prefix="ir_reduce_") as d:
                yield Path(d)
