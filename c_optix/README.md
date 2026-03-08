# CUDA/OptiX version (OptiX 7 skeleton)

This directory provides a direct **CUDA/OptiX** implementation scaffold for PO-SBR.

## What is included now
- `optix_solver.h`: C API similar to CPU version.
- `optix_solver.cu`: OptiX context/pipeline/SBT/GAS setup and launch path.
- `main_rcs_optix.cpp`: CLI entry to load OBJ and sweep `phi`.
- `CMakeLists.txt`: CUDA + OptiX build entry.

## Build
```bash
cmake -S c_optix -B c_optix/build -DOPTIX_ROOT=/path/to/NVIDIA-OptiX-SDK
cmake --build c_optix/build -j
```

## Run
```bash
./c_optix/build/po_rcs_optix geometries/trihedral.obj
```

## Notes
- This is an OptiX-backed baseline (pipeline + AS + launch).
- The PO accumulation kernel is currently a placeholder return path in `optix_po_simulate`; next step is to map the full bounce/polarization/integration logic into device programs.
