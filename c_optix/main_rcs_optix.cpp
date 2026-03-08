#include "optix_solver.h"

#include <cmath>
#include <cstdio>
#include <string>

static float field_db(OptixComplex e_theta, OptixComplex e_phi) {
    const float et = e_theta.re * e_theta.re + e_theta.im * e_theta.im;
    const float ep = e_phi.re * e_phi.re + e_phi.im * e_phi.im;
    const float e2 = et + ep;
    return 10.0f * std::log10(e2 + 1e-30f);
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::string(argv[1]) == "--optix-version") {
        std::printf("OptiX version: %s\n", optix_po_version_string());
        return 0;
    }

    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.obj>\n", argv[0]);
        std::fprintf(stderr, "       %s --optix-version\n", argv[0]);
        return 1;
    }

    std::printf("OptiX version: %s\n", optix_po_version_string());

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
        (void)r.r0;
        std::printf("phi=%6.2f deg, Level=%8.3f dB\n", phi, field_db(r.e_theta, r.e_phi));
    }

    optix_po_destroy(ctx);
    return 0;
}
