# hang-O0-threshold

> **RECOVERY NOTE (2026-08-19).** This directory's original `test.cu` (the reduced,
> ~219-line self-contained repro) and `run.sh` were **untracked** files that were
> deleted by a concurrent `git commit 4388809 "add poc and clean builds"` (which ran
> a `git clean`-style sweep of untracked paths). They were never committed and were
> not captured in any session transcript, so their exact reduced contents could not be
> recovered. What remains authoritative:
>
> - the **metadata.yaml** (trigger ranges from the sm_86 11.1..13.3 sweep — intact), and
> - the **README below** (recovered verbatim from the session transcript).
>
> The original, unreduced generator source survives at:
> `runs/run-20260819-000123-2433/differentials/generator_ir__418864b9bd901c35/test.cu`
> (574 lines). Note: the *unreduced* source times out at O0 **and** O1 on this host at
> 13s, so it does not by itself reproduce the clean O0-only split — that split was a
> property of the reduced kernel (smaller launch + retained `helper_0`/`helper_1`).
> Re-reducing from the original source is the recommended path if the reduced `test.cu`
> is needed again.

---

# CUDA O0 timeout reduced testcase

Original case:

`/export/ssd1/xzhanglb/Cuda_Stress/runs/run-20260819-000123-2433/differentials/generator_ir__418864b9bd901c35/test.cu`

This is a reduced, self-contained timeout testcase. It does not need `input.bin` or `expected.bin`; the input buffer is initialized in the program.

## Reproduce

```sh
cd /export/ssd1/xzhanglb/Cuda_Stress/poc/hang-O0-threshold
./run.sh
```

`run.sh` uses `RUN_TIMEOUT=10s` by default. With that threshold, the reduced case reproduces the O0 timeout group:

```text
cuda-13.3 O0  timeout
cuda-13.0 O0  timeout
cuda-12.5 O0  timeout
cuda-13.3 O1/O2  complete
cuda-13.0 O1/O2  complete
cuda-12.5 O1/O2  complete
```

The case is a threshold-style runtime hang, not an infinite loop. With a 13 second timeout, the measured runtime during reduction was approximately:

```text
cuda-13.3 O0  10.6s
cuda-13.0 O0  10.7s
cuda-12.5 O0  12.5s
cuda-13.3 O1   7.9s
cuda-13.0 O1   8.5s
cuda-12.5 O1   8.3s
cuda-13.3 O2   8.4s
cuda-13.0 O2   8.1s
cuda-12.5 O2   7.7s
```

## Reduction Notes

The original 574-line generator case was reduced to the finite hot loop that preserves the timeout split:

- `helper_0` and `helper_1` are retained from the original generator output.
- The outer random-expression noise, `helper_2`, host input generation, host checksum fold, and final bank/switch stress blocks were removed.
- The remaining kernel keeps the original stress shape: repeated strided global loads, an atomic update, shared-memory neighbor access with `__syncthreads`, and nested calls to `helper_1(helper_0(...))`.

The testcase has no inline PTX. The loop bounds are finite; the observed hang classification comes from the runtime threshold used by the differential harness.
