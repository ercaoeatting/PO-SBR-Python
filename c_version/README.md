# C Version (CPU Reference)

This directory currently provides a **CPU reference implementation** of the PO-SBR flow.

## Important
- This implementation **does not use CUDA or OptiX**.
- It is intended for portability, debugging and algorithm validation.
- If you need production GPU performance, use the original Python + modified `rtxpy` OptiX pipeline.

## Build
```bash
cd c_version
make
```

## Run
```bash
./po_rcs ../geometries/trihedral.obj
```

## Why keep a CPU version?
- Easier to run in CI without NVIDIA GPU/driver stack.
- Easier to debug numerical logic step-by-step.
- Useful as a baseline to compare against later CUDA/OptiX implementation.

## Suggested migration path to CUDA/OptiX
1. Keep current `po_solver.h` data contract as host-side API.
2. Replace `trace_mesh` loop with OptiX acceleration structure + raygen/hit/miss programs.
3. Move bounce/polarization update into device programs or staged CUDA kernels.
4. Keep PO accumulation logic as either device reduction or host-side postprocess.
5. Cross-check GPU outputs against this CPU reference on the same mesh and angle sweep.
