#include "po_solver.h"

#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

static double field_db(Complex e_theta, Complex e_phi) {
    double et = e_theta.re * e_theta.re + e_theta.im * e_theta.im;
    double ep = e_phi.re * e_phi.re + e_phi.im * e_phi.im;
    double e2 = et + ep;
    return 10.0 * log10(e2 + 1e-30);
}

int main(int argc, char **argv) {
    const char *filename = NULL;
    const double alpha = 180.0;
    const double theta = 90.0;
    const double freq = 3e9;
    const int rays_per_lambda = 3;
    const int bounces = 3;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.obj>\n", argv[0]);
        fprintf(stderr, "example: %s ../geometries/trihedral.obj\n", argv[0]);
        return 1;
    }
    filename = argv[1];

    printf("[info] running CPU reference solver (no CUDA/OptiX)\n");

    Mesh mesh;
    if (po_load_obj(filename, &mesh) != 0) {
        fprintf(stderr, "failed to load mesh: %s\n", filename);
        return 1;
    }

    for (double phi = 45.0; phi < 135.0; phi += 1.0) {
        SimResult result = po_simulate(alpha, phi, theta, freq, rays_per_lambda, bounces, &mesh);
        (void)result.r0;
        double level_db = field_db(result.e_theta, result.e_phi);
        printf("phi=%6.2f deg, Level=%8.3f dB\n", phi, level_db);
    }

    po_free_mesh(&mesh);
    return 0;
}
