#include "optix_solver.h"

#include <cmath>
#include <cstdio>

static float rcs_db(OptixComplex e_theta, OptixComplex e_phi, float r0) {
    const float pi = 3.14159265358979323846f;
    const float et = e_theta.re * e_theta.re + e_theta.im * e_theta.im;
    const float ep = e_phi.re * e_phi.re + e_phi.im * e_phi.im;
    const float sigma = 4.0f * pi * r0 * r0 * (et + ep);
    return 10.0f * std::log10(sigma + 1e-30f);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.obj>\n", argv[0]);
        return 1;
    }

    OptixPoContext *ctx = nullptr;
    if (optix_po_create(&ctx) != 0) {
        std::fprintf(stderr, "failed to create OptiX context\n");
        return 2;
    }
    if (optix_po_load_obj(ctx, argv[1]) != 0) {
        std::fprintf(stderr, "failed to load OBJ: %s\n", argv[1]);
        optix_po_destroy(ctx);
        return 3;
    }

    const float alpha = 180.0f;
    const float theta = 90.0f;
    const float freq = 3e9f;
    const int rays_per_lambda = 3;
    const int bounces = 3;

    for (float phi = 45.0f; phi < 135.0f; phi += 1.0f) {
        OptixSimResult r = optix_po_simulate(ctx, alpha, phi, theta, freq, rays_per_lambda, bounces);
        std::printf("phi=%6.2f deg, RCS=%8.3f dBsm\n", phi, rcs_db(r.e_theta, r.e_phi, r.r0));
    }

    optix_po_destroy(ctx);
    return 0;
}
