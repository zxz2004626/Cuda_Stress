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

Only one generation backend feeds the queue/comparison/archive pipeline:
`generator_ir`, which emits complete CUDA Runtime programs and compares a
deterministic u64 checksum across builds.  The original fixed-ABI generator
strategies (predicate_pressure, integer_chain, memory_address, warp_mixed,
register_pressure, inline_ptx) have been removed from the generation path.

---

## HIP / ROCm port (AMD differential testing)

The original builtin strategies emit CUDA/PTX-specific code (inline PTX asm,
`__byte_perm`, `ptxas` flags, ...), so the AMD port uses the repository's
`generator_ir` HIP backend instead: `hip_stress.py` generates self-contained
`.hip` test programs and compiles each one with every configured `hipcc`
version at every optimization profile, then compares the deterministic
`checksum=` lines emitted by the built-in harness.

On the AMD host (`shuaipc1`) three isolated ROCm installs are used:

```bash
cd Cuda_Stress_hip
python3 hip_stress.py doctor
python3 hip_stress.py generate --seed 1 --count 8 --out corpus_hip
python3 hip_stress.py run --corpus corpus_hip --runs 3
python3 hip_stress.py loop --seed 1 --seeds 50 --complexity medium --runs 3
```

Configuration lives in `config/hip_toolchains.yaml`.  The host's distro only
provides `libxml2.so.16`, while ROCm's `lld` links against `libxml2.so.2`; the
script creates a small user-owned shim symlink automatically (configured by
`libxml2_dir` / `libxml2_source`).


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

# 3. Generate a batch of deterministic generator_ir testcases into corpus/accepted
python3 -m cuda_stress generate --count 4 --test-seed 100
python3 -m cuda_stress generate --strategy generator_ir --count 1 --test-seed 100

# 4. Compile + run one case's full matrix (all toolchains × all ptxas profiles × 3 runs)
python3 -m cuda_stress run --test corpus/accepted/generator_ir__<id> --runs 3
python3 -m cuda_stress compare --test corpus/accepted/generator_ir__<id>

# 5. Continuous, resumable differential-testing loop
python3 pipeline.py loop --max-cases 100          # finite run for regression
python3 pipeline.py loop --duration 24h           # long unattended run
python3 pipeline.py status                        # queue / run state
python3 pipeline.py resume                        # continue an interrupted loop
python3 pipeline.py report --run <run_id>         # summarize a run
python3 pipeline.py gc                            # enforce disk budget now

# 6. Regenerate any case byte-for-byte from its manifest
python3 -m cuda_stress reproduce --id generator_ir__<id> --out /tmp/repro

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

## Testcase execution contracts and layout

Original strategies compile to the fixed entry point:

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

`generator_ir` cases declare `execution_mode: driver_checksum` and
`oracle_mode: cross_build` in the manifest.  Each CUDA toolkit independently compiles
the source to PTX once; the toolkit's own `ptxas` then builds each optimization-profile
cubin.  The manifest records the variable kernel ABI, launch dimensions, deterministic
input construction, and global/constant-memory initialization.  A Driver API worker
replays that contract, launches exactly once, and returns the same u64 checksum as the
generated Runtime harness.  Workers reuse one `CUcontext` for the configured number of
successful jobs and are killed immediately after any CUDA error.

`input.bin` and `expected.bin` remain empty compatibility artifacts; they are never
misrepresented as a CPU oracle. Each build is still run repeatedly for stability, and
stable checksum disagreement across the toolchain/profile matrix is classified as
`POTENTIAL_COMPILER_BUG`. Legacy archived `standalone_checksum` manifests remain
runnable for reproduction.

## Generators (original strategies + optional IR backend)

| strategy            | what it stresses |
|---------------------|------------------|
| `predicate_pressure`| 32–96 runtime predicates, distant duplicated comparisons (CSE vs recompute), diamonds, predicate→GPR folding, branch/select order, `__launch_bounds__` variants |
| `integer_chain`     | long loop-carried u32/u64 chain: `mul.hi`, funnel shift, `bfe`/`bfi`, `prmt`, `lop3`, `popc`/`clz`/`brev`, rotates; boundary inputs; everything unsigned-modular (no UB) |
| `memory_address`    | u8/u16/u32/u64 + vector loads, `ld.global` cache operators (default/ca/cg/cs), multi-level runtime addressing, struct indexing, local-array spill pressure |
| `warp_mixed`        | `__ballot`/`__any`/`__all`/`__activemask`/`__shfl*`/`__match_any`/`__match_all`, full and legal partial member masks (width-16 segments), then a long chain + store |
| `register_pressure` | 32–128 simultaneously-live u64 values across `__noinline__`/`__forceinline__` chains, merge + reuse, `launch_bounds`/`maxrregcount` variants |
| `inline_ptx`        | an ISA-constrained inline-PTX block (`asm volatile`) with 140+ CPU-mirrored type/modifier variants across scalar integer, bit manipulation, logic, shifts, comparisons and selection. `balanced` mode guarantees each eligible variant before repetition; `random` mode explores stochastic cross-family chains. PTX-version/SM gates, boundary immediates, 32/64-bit mixing and accumulator folds keep cases valid and optimization-relevant |
| `generator_ir`      | optional typed-IR/template generator (`map_transform`, reductions, scans, 2D tiles, synchronization, register/ISA stress, and more); records a dynamic Driver ABI, compiles one PTX per toolkit plus profile-specific cubins, and uses deterministic cross-build checksum consensus rather than replacing any original strategy |

Every original strategy has an **independent CPU oracle** (`cpu_oracle`) that mirrors the CUDA
semantics exactly (mod-2^64 arithmetic, PTX bit ops, warp lane/member-mask rules, byte
endianness) — verified on-GPU by the smoke tests.  All shift amounts, divisors, indices,
warp member masks and byte offsets are constrained at generation time; unsafe candidates
are rejected with a reason.

The IR backend does not have a complete CPU interpreter (its IR includes host-initialized
global/constant state, atomics, synchronization and optional 2D kernels).  Its explicit
`cross_build` oracle therefore detects compiler/configuration divergence without inventing
an `expected.bin`.  `observable_fold` is forced on, performance warmups are disabled, and
an input-salt rerun checks that retained checksums really depend on runtime input.

### inline_ptx implementation notes

nvcc inline-asm substitutes `%N` operands and owns `%r/%rd/%p` register names, so the
generator emits internal PTX registers through the `%%xr`/`%%xd`/`%%xp` escape (literal
`%`). PTX requires a 32-bit count operand even for 64-bit `shl`/`shr`; counts greater
than the register width are **clamped to the width**, not wrapped. The generator now
samples `0`, `N-1`, `N`, `N+1`, and `255` without pre-masking. Funnel-shift `.wrap`
continues to wrap as specified, while `.clamp` samples its distinct boundary behavior.
All op semantics in the table are checked against Python mirrors and exercised by GPU
O0/O3 smoke cases. One real generation bug was found and fixed:
`prmt`'s selector must be emitted with a `0x` prefix (a bare hex literal parses as
decimal). Generic `prmt` selectors now use the full nibble range `0..15`; selector bit 3
requests byte-sign replication and was previously over-constrained away. PTX `bfe/bfi`
take position/length in `[0,255]` and a field extending past the
register MSB is **well-defined** (zero-extended for `bfe`, the excess bits simply dropped
for `bfi`) — there is no `pos+wid <= width` constraint; the generator samples that full
over-MSB region.

Each operation records its minimum PTX ISA and SM requirement. The default comparison
profile is PTX 7.3 because CUDA 11.3 is the oldest configured toolkit; PTX 7.6+ operations
such as `bmsk`/`szext` are available only when `max_ptx_isa` is raised explicitly. Run
`python3 tools/audit_ptx_coverage.py` to compare deliberate generator coverage with the
saved PTX ISA 9.3 instruction index. Family coverage is intentionally reported separately
from type/modifier variants: memory, texture, tensor-core, async-copy and synchronization
families require dedicated stateful oracles and are not falsely counted as scalar coverage.

With the bfe/bfi generator sampling the full legal over-MSB range, a sweep over 40
configurations found **4 programs** that are correct on `-O0` on all runnable toolchains
but produce deterministic, different output on `-O1/-O2/-O3` on cuda-11.8 / 12.1 / 12.6
(cuda-11.3 stays correct).  Two distinct trigger classes were isolated:
  * `prmt.b32(d, a, a, sel)` (test_seed 9999, case `inline_ptx__ccf599c62b41dba5`),
  * **over-MSB `bfe/bfi` fields** (pos+wid > 32; test_seed 777, case
    `inline_ptx__ebf45243624dc5d6`, plus the 180-instruction variants of test_seeds
    77 / 99).

Both are real ptxas O1+ miscompilations on fully valid PTX: two independent CPU oracles
agree; the exact `nvcc -ptx` output fed directly to `ptxas -O0` vs `-O1` reproduces; a
runtime legality audit finds no UB; the culprit instruction is correct in isolation but
mis-optimized inside the large inline-asm block.  The pipeline flags this class as
**POTENTIAL_COMPILER_BUG** and archives it under `runs/<id>/differentials/<case_id>/`
with a `reproduce.sh`.  Note that the first of these was initially (wrongly) attributed
to 'bfe/bfi pos+wid>32 being UB'; PTX defines those cases, and widening the generator to
the legal range is what surfaced the additional bugs.

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
                           + isolated standalone checksum execution for generator_ir
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
* every random decision in either backend is seed-driven; a case's id is content-derived
  for automatic deduplication,
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
* `generator_ir` uses cross-build consensus rather than an independent CPU oracle, and its
  standalone CUDA Runtime harness does not currently provide the native runner's device
  output guard.  Differentials should therefore be confirmed with sanitizers before filing.
