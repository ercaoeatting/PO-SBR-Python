#ifndef PO_SOLVER_H
#define PO_SOLVER_H

#include <stddef.h>

typedef struct {
    double x;
    double y;
    double z;
} Vec3;

typedef struct {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
} Triangle;

typedef struct {
    Triangle *triangles;
    size_t triangle_count;
    Vec3 bb_min;
    Vec3 bb_max;
} Mesh;

typedef struct {
    double re;
    double im;
} Complex;

typedef struct {
    Complex e_theta;
    Complex e_phi;
    double r0;
} SimResult;

int po_load_obj(const char *filename, Mesh *mesh);
void po_free_mesh(Mesh *mesh);

SimResult po_simulate(
    double alpha_deg,
    double phi_deg,
    double theta_deg,
    double freq_hz,
    int rays_per_lambda,
    int max_bounces,
    const Mesh *mesh
);

#endif
