# cuda_stress — deterministic CUDA compiler differential-testing pipeline

A lightweight, high-throughput, reproducible pipeline that hammers `nvcc` / `cicc` /
`ptxas` with thousands of generated kernels and compares their compiled behavior across
toolchain versions and ptxas optimization levels.  It hunts **wrong-code regressions**,
not performance: every testcase is validated against an independent CPU oracle, every
build is executed multiple times, and a result is only ever labelled a potential compiler
bug when multiple *stable* builds disagree.

No database, no web service, no LLM at runtime.  Python orchestrates, one tiny C++
Driver-API binary executes kernels, YAML configures, JSON/JSONL records, directories
archive, pytest tests.

```
Generator → stress-gate → N×M compile → PTX/SASS metrics → GPU run ×3
         → compare (stability / oracle / cross-matrix) → classify
         → archive differentials / retain interesting / GC redundant
         → mutate & generate next batch   (continuous, resumable loop)
```

---

## Requirements

* Linux with an NVIDIA GPU + driver (WSL2 works)
* One or more CUDA toolkits (11.3 / 11.8 / 12.1 / 12.6 / 13.x tested)
* Python ≥ 3.10, `PyYAML`, `pytest`; `gcc`/`g++`

## From an empty directory

```bash
git clone <this repo> && cd cuda_stress

# 1. Build the tiny Driver-API runner (needs no CUDA runtime, only libcuda)
make -C native

# 2. Sanity-check toolchains, GPU and the real ptxas -O0..-O4 landscape
python3 pipeline.py doctor --force

# 3. Generate a batch of deterministic testcases into corpus/accepted
python3 -m cuda_stress generate --strategy integer_chain \
        --strategy predicate_pressure --count 4 --test-seed 100

# 4. Compile + run one case's full matrix (all toolchains × all ptxas profiles × 3 runs)
python3 -m cuda_stress run --test corpus/accepted/integer_chain__<id> --runs 3
python3 -m cuda_stress compare --test corpus/accepted/integer_chain__<id>

# 5. Continuous, resumable differential-testing loop
python3 pipeline.py loop --max-cases 100          # finite run for regression
python3 pipeline.py loop --duration 24h           # long unattended run
python3 pipeline.py status                        # queue / run state
python3 pipeline.py resume                        # continue an interrupted loop
python3 pipeline.py report --run <run_id>         # summarize a run
python3 pipeline.py gc                            # enforce disk budget now

# 6. Regenerate any case byte-for-byte from its manifest
python3 -m cuda_stress reproduce --id integer_chain__<id> --out /tmp/repro

# 7. Tests (unit + GPU smoke)
python3 -m pytest tests/ -v
```

---

## What the probe (`doctor`) found on this machine

`doctor` compiles a stress kernel with every toolkit at `-Xptxas=-O0..-O4`, compares the
normalized SASS, and tries to load+run each CUBIN on the host driver:

| toolkit   | nvcc           | ptxas -O accepted | behaviorally distinct | driver runs? |
|-----------|----------------|-------------------|-----------------------|--------------|
| cuda-11.3 | V11.3.58       | O0 O1 O2 O3 O4    | O0, O1, O2            | yes          |
| cuda-11.8 | V11.8.89       | O0 O1 O2 O3 O4    | O0, O1, O2            | yes          |
| cuda-12.1 | V12.1.105      | O0 O1 O2 O3 O4    | O0, O1, O2            | yes          |
| cuda-12.6 | V12.6.85       | O0 O1 O2 O3 O4    | O0, O1, O2            | yes          |
| cuda-13.0 | V13.0.88       | O0 O1 O2 O3 O4    | O0, O1, O2            | **no**       |
| cuda-13.3 | V13.3.73       | O0 O1 O2 O3 O4    | O0, O1, O2            | **no**       |

**`-O4` is not a real ptxas level.**  Every toolkit *accepts* `-Xptxas=-O4` but emits SASS
byte-identical to `-O3` (registers, spills, stack frame identical).  `-O4` is therefore a
silent alias and is **not** advertised as a fifth level; the framework's profile matrix is
fully configurable (`optimization_profiles` in `config/toolchains.yaml`) and defaults to the
real O0–O3.

**`-O3` collapses into `-O2`** on modern ptxas (11.8+): for every stress kernel tried,
`ptxas -O2` and `-O3` produce identical SASS and register counts.  The pipeline detects
this per testcase and reports it rather than pretending the levels are distinct.

**CUDA 13 cannot execute on a CUDA 12.1 driver** (error 200 at `cuModuleLoad`,
`CUDA_ERROR_INVALID_IMAGE`); its PTX (ISA 9.x) is also rejected by the 12.1 JIT
(error 222).  CUDA 13 toolchains still participate in the compile + SASS-metrics matrix;
their run results are recorded as `UNSUPPORTED`, never as a compiler bug.

## Testcase ABI and layout

Every testcase compiles to the same entry point:

```cuda
extern "C" __global__ void test_kernel(
    const unsigned long long* input,     // input_words × u64
    unsigned long long* output,          // grid·block · output_words_per_thread × u64
    unsigned int input_words,
    unsigned long long seed);
```

A case directory contains `test.cu`, `manifest.json`, `input.bin`, `expected.bin`,
`generation.json`.  A case is a **pure function of `(config_seed, test_seed, strategy,
params)`** — regenerate it byte-for-byte with `reproduce`.

## Generators (5 strategies × parameterized variants)

| strategy            | what it stresses |
|---------------------|------------------|
| `predicate_pressure`| 32–96 runtime predicates, distant duplicated comparisons (CSE vs recompute), diamonds, predicate→GPR folding, branch/select order, `__launch_bounds__` variants |
| `integer_chain`     | long loop-carried u32/u64 chain: `mul.hi`, funnel shift, `bfe`/`bfi`, `prmt`, `lop3`, `popc`/`clz`/`brev`, rotates; boundary inputs; everything unsigned-modular (no UB) |
| `memory_address`    | u8/u16/u32/u64 + vector loads, `ld.global` cache operators (default/ca/cg/cs), multi-level runtime addressing, struct indexing, local-array spill pressure |
| `warp_mixed`        | `__ballot`/`__any`/`__all`/`__activemask`/`__shfl*`/`__match_any`/`__match_all`, full and legal partial member masks (width-16 segments), then a long chain + store |
| `register_pressure` | 32–128 simultaneously-live u64 values across `__noinline__`/`__forceinline__` chains, merge + reuse, `launch_bounds`/`maxrregcount` variants |
| `inline_ptx`        | a **randomly generated inline-PTX block** (`asm volatile`) from an empirically-verified PTX instruction set: rare ops (prmt/shf.r/bfe/bfi/lop3/sad/mul.hi/popc/clz/brev), predicates, 32/64-bit mixing, long chains.  Every instruction is folded into a running accumulator so ptxas cannot dead-code it at any `-O`; register/instruction counts are parameterized (`n_instr`/`n_regs32`/`n_regs64`/`n_preds`) |

Every strategy has an **independent CPU oracle** (`cpu_oracle`) that mirrors the CUDA
semantics exactly (mod-2^64 arithmetic, PTX bit ops, warp lane/member-mask rules, byte
endianness) — verified on-GPU by the smoke tests.  All shift amounts, divisors, indices,
warp member masks and byte offsets are constrained at generation time; unsafe candidates
are rejected with a reason.

### inline_ptx implementation notes

nvcc inline-asm substitutes `%N` operands and owns `%r/%rd/%p` register names, so the
generator emits internal PTX registers through the `%%xr`/`%%xd`/`%%xp` escape (literal
`%`), and every shift count is pre-masked into a legal range (PTX requires `.b32` count
registers for `.b64` shifts).  All op semantics in the table were empirically verified
against their Python mirrors on a real GPU.  Two real bugs were found and fixed during
validation: PTX `bfe/bfi` require `pos+wid <= 32` (the generator once allowed 33, which is
undefined behaviour that ptxas `-O1+` exploits while `-O0` does not), and `prmt`'s selector
must be emitted with a `0x` prefix (a bare hex literal parses as decimal).

During the final sweep one generated program (120 instructions, a specific seed) was
correct on `-O0` on all runnable toolchains but differed on `-O1/-O2/-O3` on cuda-11.8 /
12.1 / 12.6 (cuda-11.3 stayed correct).  The pipeline flags exactly this class of case as
**POTENTIAL_COMPILER_BUG** and archives it under `runs/<id>/differentials/<case_id>/` with a
`reproduce.sh`.  Whether it is a ptxas scheduling bug or a subtle latent UB is left to
investigation; the oracle was validated against the unoptimized build, and each op
individually at `-O3`.

## Pipeline modules

```
native/cuda_runner.cpp     Driver-API only: load CUBIN, launch, copy, guard check,
                           writes a self-describing blob (magic/id/status/words/result/canary)
pipeline/config.py         YAML config loading/merging, isolated env per nvcc invocation
pipeline/probe.py          doctor: ptxas -O0..-O4 probe, driver-compat matrix (cached)
pipeline/compiler.py       nvcc→PTX, ptxas -O{n}→CUBIN, cuobjdump→SASS; full command records
pipeline/disasm.py         SASS/PTX instruction counts, opcode histograms, normalized hashes
pipeline/stress_metrics.py configurable stress gate + weighted novelty score
pipeline/comparator.py     per-build stability, CPU-oracle equality, cross-matrix differential
pipeline/executor.py       runner invocation, per-device serialization, output-blob parsing
pipeline/manifest.py       standard case-dir read/write, hash verification
pipeline/oracle.py         CPU-oracle recompute, input-sensitivity check
pipeline/queue.py          directory queue with atomic-rename state transitions
pipeline/archive.py        runs/<id> artifacts, results.jsonl ledger, differentials/<id> +
                           reproduce.sh, disk-budget garbage collector
pipeline/scheduler.py      producer–consumer loop (parallel compile, serial GPU, resume)
pipeline/cli.py            generate / reproduce / doctor / run / compare / report / status /
                           loop / resume / gc
```

## Results and classifications

Per-case status: `PASS` · `POTENTIAL_COMPILER_BUG` · `ORACLE_MISMATCH` ·
`NONDETERMINISTIC` · `COMPILE_FAILED` · `RUN_FAILED` · `RUN_TIMEOUT` ·
`INCOMPLETE_MATRIX` · `UNSUPPORTED` · `INVALID_TEST`.

`POTENTIAL_COMPILER_BUG` is only set when **multiple stable builds produce different
outputs**.  A single unstable run, a compile failure, a driver that can't load a CUBIN or a
timeout never masquerades as a wrong-code regression.

Per-case category (retention): `interesting` (clears the stress gate) · `potential_bug`
(differential / compiler crash / GPU fault / oracle mismatch) · `flaky` · `redundant`
(deletable) · `invalid` · `unsupported`.  `potential_bug` and `flaky` cases are protected
from garbage collection and are snapshotted into `runs/<id>/differentials/<case_id>/`
together with a `reproduce.sh`.

## Continuous loop

```
generate → (stress gate) → N×M compile → PTX/SASS metrics → GPU run ×3
        → compare → classify → archive / retain / GC → mutate & generate next
```

* generator workers, bounded compile pool (`jobs_compile`), per-GPU **serial** execution
  (one worker per device by default — no resource contention, deterministic),
* directory queue + atomic `rename` per state transition, so a crash leaves every task in
  a recoverable stage (`resume` rehydrates),
* every random decision is seed-driven; a case's id is content-derived (automatic dedup),
* `SIGINT`/`SIGTERM`: stop generating, drain in-flight tasks, atomically persist queue
  state + seeds + statistics,
* storage budget enforced by `gc` (never deletes unique repro material or protected cases).

## Known limitations

* `-O3` is not distinct from `-O2` on the ptxas versions tested here; the 4-profile matrix
  therefore yields ~3 distinct SASS families per toolkit.
* CUDA 13.x runs are unsupported on a CUDA 12.1 driver (documented above).
* GPU workers are serial per device; multi-GPU needs `gpu_devices` set in config.
* The novelty score does not yet persist corpus SASS hashes across runs (per-run only).
* No float strategies yet (float semantics, ULP and contraction policies are a future
  extension).
