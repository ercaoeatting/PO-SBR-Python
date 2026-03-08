#include "po_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-9
#define RAY_EPS 1e-6
#define C0 299792458.0
#define PI 3.14159265358979323846

typedef struct {
    Vec3 origin;
    Vec3 dir;
    Vec3 pol;
    double distance;
    int active;
    int has_hit;
} Ray;

static inline double deg2rad(double d) { return d * PI / 180.0; }
static inline double cosd(double d) { return cos(deg2rad(d)); }
static inline double sind(double d) { return sin(deg2rad(d)); }

static inline Vec3 v_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 v_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 v_scale(Vec3 a, double s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }
static inline double v_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 v_cross(Vec3 a, Vec3 b) {
    return (Vec3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline double v_norm(Vec3 a) { return sqrt(v_dot(a, a)); }
static inline Vec3 v_normed(Vec3 a) {
    double n = v_norm(a);
    return n < EPS ? (Vec3){0.0, 0.0, 0.0} : v_scale(a, 1.0 / n);
}

static inline Complex c_make(double re, double im) { return (Complex){re, im}; }
static inline Complex c_add(Complex a, Complex b) { return c_make(a.re + b.re, a.im + b.im); }
static inline Complex c_mul(Complex a, Complex b) {
    return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static inline Complex c_scale(Complex a, double s) { return c_make(a.re * s, a.im * s); }
static inline Complex c_expj(double phase) { return c_make(cos(phase), sin(phase)); }

static void ortho_set(double phi, double theta, Vec3 *normal, Vec3 *up, Vec3 *right) {
    *normal = (Vec3){sind(theta) * cosd(phi), sind(theta) * sind(phi), cosd(theta)};
    *right = (Vec3){sind(phi), -cosd(phi), 0.0};
    *up = v_cross(*right, *normal);
}

static int parse_face_index(const char *token) {
    return atoi(token) - 1;
}

int po_load_obj(const char *filename, Mesh *mesh) {
    memset(mesh, 0, sizeof(*mesh));

    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    size_t v_cap = 1024;
    size_t f_cap = 1024;
    size_t v_count = 0;
    size_t f_count = 0;
    Vec3 *verts = (Vec3 *)malloc(v_cap * sizeof(Vec3));
    Triangle *tris = (Triangle *)malloc(f_cap * sizeof(Triangle));
    if (!verts || !tris) {
        fclose(fp);
        free(verts);
        free(tris);
        return -2;
    }

    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            if (v_count == v_cap) {
                v_cap *= 2;
                verts = (Vec3 *)realloc(verts, v_cap * sizeof(Vec3));
                if (!verts) break;
            }
            Vec3 v = {0.0, 0.0, 0.0};
            sscanf(line + 2, "%lf %lf %lf", &v.x, &v.y, &v.z);
            verts[v_count++] = v;
        } else if (line[0] == 'f' && line[1] == ' ') {
            if (f_count == f_cap) {
                f_cap *= 2;
                tris = (Triangle *)realloc(tris, f_cap * sizeof(Triangle));
                if (!tris) break;
            }
            char *tok = strtok(line + 2, " \t\r\n");
            int idx[3];
            int n = 0;
            while (tok && n < 3) {
                idx[n++] = parse_face_index(tok);
                tok = strtok(NULL, " \t\r\n");
            }
            if (n == 3 && idx[0] >= 0 && idx[1] >= 0 && idx[2] >= 0 &&
                (size_t)idx[0] < v_count && (size_t)idx[1] < v_count && (size_t)idx[2] < v_count) {
                Triangle t;
                t.v0 = verts[idx[0]];
                t.v1 = verts[idx[1]];
                t.v2 = verts[idx[2]];
                t.normal = v_normed(v_cross(v_sub(t.v1, t.v0), v_sub(t.v2, t.v0)));
                tris[f_count++] = t;
            }
        }
    }
    fclose(fp);

    if (!verts || !tris || f_count == 0) {
        free(verts);
        free(tris);
        return -3;
    }

    Vec3 bb_min = verts[0];
    Vec3 bb_max = verts[0];
    for (size_t i = 1; i < v_count; ++i) {
        if (verts[i].x < bb_min.x) bb_min.x = verts[i].x;
        if (verts[i].y < bb_min.y) bb_min.y = verts[i].y;
        if (verts[i].z < bb_min.z) bb_min.z = verts[i].z;
        if (verts[i].x > bb_max.x) bb_max.x = verts[i].x;
        if (verts[i].y > bb_max.y) bb_max.y = verts[i].y;
        if (verts[i].z > bb_max.z) bb_max.z = verts[i].z;
    }

    mesh->triangles = tris;
    mesh->triangle_count = f_count;
    mesh->bb_min = bb_min;
    mesh->bb_max = bb_max;

    free(verts);
    return 0;
}

void po_free_mesh(Mesh *mesh) {
    free(mesh->triangles);
    memset(mesh, 0, sizeof(*mesh));
}

static int intersect_triangle(const Ray *ray, const Triangle *tri, double *t_out, Vec3 *n_out) {
    Vec3 e1 = v_sub(tri->v1, tri->v0);
    Vec3 e2 = v_sub(tri->v2, tri->v0);
    Vec3 pvec = v_cross(ray->dir, e2);
    double det = v_dot(e1, pvec);
    if (fabs(det) < EPS) return 0;

    double inv_det = 1.0 / det;
    Vec3 tvec = v_sub(ray->origin, tri->v0);
    double u = v_dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) return 0;

    Vec3 qvec = v_cross(tvec, e1);
    double v = v_dot(ray->dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) return 0;

    double t = v_dot(e2, qvec) * inv_det;
    if (t <= RAY_EPS) return 0;

    *t_out = t;
    *n_out = tri->normal;
    return 1;
}

static int trace_mesh(const Mesh *mesh, const Ray *ray, double *t_hit, Vec3 *n_hit) {
    double best_t = 1e100;
    int hit = 0;
    Vec3 best_n = {0, 0, 0};
    for (size_t i = 0; i < mesh->triangle_count; ++i) {
        double t = 0.0;
        Vec3 n = {0, 0, 0};
        if (intersect_triangle(ray, &mesh->triangles[i], &t, &n)) {
            if (t < best_t) {
                best_t = t;
                best_n = n;
                hit = 1;
            }
        }
    }
    if (hit) {
        *t_hit = best_t;
        *n_hit = best_n;
    }
    return hit;
}

static Vec3 polarise(Vec3 pol, Vec3 k_inc, Vec3 k_ref, Vec3 normal) {
    Vec3 inc_x_n = v_cross(k_inc, normal);
    Vec3 e_perp = v_normed(inc_x_n);
    Vec3 e_par = v_normed(v_cross(k_inc, e_perp));

    Vec3 e_ref_perp = e_perp;
    Vec3 e_ref_par = v_normed(v_cross(k_ref, e_ref_perp));

    double E_par = v_dot(pol, e_par);
    double E_perp = v_dot(pol, e_perp);

    return v_sub(v_scale(e_ref_par, E_par), v_scale(e_ref_perp, E_perp));
}

SimResult po_simulate(
    double alpha_deg,
    double phi_deg,
    double theta_deg,
    double freq_hz,
    int rays_per_lambda,
    int max_bounces,
    const Mesh *mesh
) {
    SimResult out = {0};
    if (!mesh || !mesh->triangles || mesh->triangle_count == 0 || rays_per_lambda <= 0 || max_bounces < 0) {
        return out;
    }

    double lam = C0 / freq_hz;
    double k = 2.0 * PI / lam;
    double tube = lam / (double)rays_per_lambda;

    Vec3 pol = {
        cosd(phi_deg) * cosd(theta_deg) * cosd(alpha_deg) - sind(phi_deg) * sind(alpha_deg),
        sind(phi_deg) * cosd(theta_deg) * cosd(alpha_deg) + cosd(phi_deg) * sind(alpha_deg),
        -sind(theta_deg) * cosd(alpha_deg)
    };

    Vec3 bb_center = v_scale(v_add(mesh->bb_min, mesh->bb_max), 0.5);
    double bb_radius = 0.5 * v_norm(v_sub(mesh->bb_max, mesh->bb_min));

    Vec3 obs = {sind(theta_deg) * cosd(phi_deg), sind(theta_deg) * sind(phi_deg), cosd(theta_deg)};
    Vec3 ray_dir = v_scale(obs, -1.0);

    Vec3 ant_center = v_sub((Vec3){0, 0, 0}, v_scale(ray_dir, bb_radius + 1.0));

    int n = (int)((bb_radius * 2.0) / tube);
    if (n < 1) n = 1;

    Vec3 normal, up, right;
    ortho_set(phi_deg, theta_deg, &normal, &up, &right);

    Vec3 pool_min = v_sub(ant_center, v_scale(v_add(right, up), bb_radius));
    Vec3 up_step = v_scale(up, tube);
    Vec3 right_step = v_scale(right, tube);
    Vec3 pool_begin = v_add(pool_min, v_scale(v_add(up_step, right_step), 0.5));

    int ray_count = n * n;
    Ray *rays = (Ray *)calloc((size_t)ray_count, sizeof(Ray));
    if (!rays) return out;

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Vec3 p = v_add(pool_begin, v_add(v_scale(up_step, (double)i), v_scale(right_step, (double)j)));
            rays[idx].origin = p;
            rays[idx].dir = ray_dir;
            rays[idx].pol = pol;
            rays[idx].distance = 0.0;
            rays[idx].active = 1;
            rays[idx].has_hit = 0;
            idx++;
        }
    }

    for (int b = 0; b < max_bounces; ++b) {
        for (int r = 0; r < ray_count; ++r) {
            if (!rays[r].active) continue;
            double t_hit = 0.0;
            Vec3 n_hit = {0, 0, 0};
            if (!trace_mesh(mesh, &rays[r], &t_hit, &n_hit)) {
                rays[r].active = 0;
                continue;
            }

            if (v_dot(rays[r].dir, n_hit) > 0.0) n_hit = v_scale(n_hit, -1.0);

            rays[r].distance += t_hit;
            rays[r].has_hit = 1;
            Vec3 hit_p = v_add(rays[r].origin, v_scale(rays[r].dir, t_hit));
            Vec3 ref_dir = v_sub(rays[r].dir, v_scale(n_hit, 2.0 * v_dot(rays[r].dir, n_hit)));
            rays[r].pol = polarise(rays[r].pol, rays[r].dir, ref_dir, n_hit);
            rays[r].origin = v_add(hit_p, v_scale(n_hit, RAY_EPS));
            rays[r].dir = ref_dir;
        }
    }

    Vec3 dir_phi = {-sind(phi_deg), cosd(phi_deg), 0.0};
    Vec3 dir_theta = {cosd(theta_deg) * cosd(phi_deg), cosd(theta_deg) * sind(phi_deg), -sind(theta_deg)};
    Vec3 dir_r = {sind(theta_deg) * cosd(phi_deg), sind(theta_deg) * sind(phi_deg), cosd(theta_deg)};
    double r0 = v_norm(ant_center);

    Complex sum_theta = c_make(0.0, 0.0);
    Complex sum_phi = c_make(0.0, 0.0);

    for (int r = 0; r < ray_count; ++r) {
        if (!rays[r].has_hit || rays[r].distance <= 0.0) continue;

        Vec3 r_prime = rays[r].origin;
        Vec3 direction = rays[r].dir;
        Vec3 ray_pol = rays[r].pol;
        double dist = rays[r].distance - r0;

        double phase_ap = -k * dist;
        Complex ej = c_expj(phase_ap);
        Vec3 E_ap_re = v_scale(ray_pol, ej.re);
        Vec3 E_ap_im = v_scale(ray_pol, ej.im);

        Vec3 H_re = v_cross(direction, E_ap_re);
        Vec3 H_im = v_cross(direction, E_ap_im);

        Vec3 t1_re = v_add(v_cross(v_scale(dir_phi, -1.0), E_ap_re), v_cross(dir_theta, H_re));
        Vec3 t1_im = v_add(v_cross(v_scale(dir_phi, -1.0), E_ap_im), v_cross(dir_theta, H_im));
        Vec3 t2_re = v_add(v_cross(dir_theta, E_ap_re), v_cross(dir_phi, H_re));
        Vec3 t2_im = v_add(v_cross(dir_theta, E_ap_im), v_cross(dir_phi, H_im));

        double B_theta_re = v_dot(t1_re, direction);
        double B_theta_im = v_dot(t1_im, direction);
        double B_phi_re = v_dot(t2_re, direction);
        double B_phi_im = v_dot(t2_im, direction);

        double dot_kr = k * v_dot(dir_r, r_prime);
        Complex spatial = c_expj(dot_kr);
        Complex prefac = c_scale(c_make(0.0, k / (4.0 * PI * r0)), tube * tube);

        Complex factor = c_mul(prefac, spatial);
        Complex B_theta = c_make(B_theta_re, B_theta_im);
        Complex B_phi = c_make(B_phi_re, B_phi_im);

        sum_theta = c_add(sum_theta, c_mul(factor, B_theta));
        sum_phi = c_add(sum_phi, c_mul(factor, B_phi));
    }

    free(rays);

    out.e_theta = sum_theta;
    out.e_phi = sum_phi;
    out.r0 = r0;
    (void)bb_center;
    return out;
}
