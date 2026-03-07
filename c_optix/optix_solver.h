#ifndef OPTIX_SOLVER_H
#define OPTIX_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float re;
    float im;
} OptixComplex;

typedef struct {
    OptixComplex e_theta;
    OptixComplex e_phi;
    float r0;
} OptixSimResult;

typedef struct OptixPoContext OptixPoContext;

int optix_po_create(OptixPoContext **ctx);
void optix_po_destroy(OptixPoContext *ctx);

int optix_po_load_obj(OptixPoContext *ctx, const char *filename);

OptixSimResult optix_po_simulate(
    OptixPoContext *ctx,
    float alpha_deg,
    float phi_deg,
    float theta_deg,
    float freq_hz,
    int rays_per_lambda,
    int max_bounces
);

#ifdef __cplusplus
}
#endif

#endif
