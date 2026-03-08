#include "optix_solver.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) return -1; } while (0)
#define OPTIX_CHECK(call) do { OptixResult r = (call); if (r != OPTIX_SUCCESS) return -2; } while (0)

struct float3h { float x, y, z; };
struct int3h { int x, y, z; };

struct LaunchParams {
    OptixTraversableHandle gas;
    float3h *ray_origins;
    float3h *ray_dirs;
    float3h *ray_normals;
    float *ray_t;
    int ray_count;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct OptixPoContext {
    CUcontext cu_ctx;
    CUstream stream;
    OptixDeviceContext optix_ctx;
    OptixModule module;
    OptixProgramGroup pg_raygen;
    OptixProgramGroup pg_miss;
    OptixProgramGroup pg_hit;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;

    CUdeviceptr d_vertices;
    CUdeviceptr d_indices;
    size_t vertex_count;
    size_t index_count;

    CUdeviceptr d_gas_output;
    OptixTraversableHandle gas_handle;

    CUdeviceptr d_ray_o;
    CUdeviceptr d_ray_d;
    CUdeviceptr d_ray_n;
    CUdeviceptr d_ray_t;
    CUdeviceptr d_params;

    float3h bb_min;
    float3h bb_max;
};

static const char *kPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64

.visible .entry __raygen__rg() { ret; }
.visible .entry __miss__ms() { ret; }
.visible .entry __closesthit__ch() { ret; }
)ptx";


extern "C" const char *optix_po_version_string(void) {
    static char ver[32];
    const int major = OPTIX_VERSION / 10000;
    const int minor = (OPTIX_VERSION % 10000) / 100;
    const int patch = OPTIX_VERSION % 100;
    snprintf(ver, sizeof(ver), "%d.%d.%d", major, minor, patch);
    return ver;
}

static int load_obj(const char *filename, std::vector<float3h> &verts, std::vector<int3h> &faces, float3h &bbmin, float3h &bbmax) {
    std::ifstream in(filename);
    if (!in) return -3;

    bbmin = {1e30f, 1e30f, 1e30f};
    bbmax = {-1e30f, -1e30f, -1e30f};

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float3h v{};
            ss >> v.x >> v.y >> v.z;
            verts.push_back(v);
            if (v.x < bbmin.x) bbmin.x = v.x;
            if (v.y < bbmin.y) bbmin.y = v.y;
            if (v.z < bbmin.z) bbmin.z = v.z;
            if (v.x > bbmax.x) bbmax.x = v.x;
            if (v.y > bbmax.y) bbmax.y = v.y;
            if (v.z > bbmax.z) bbmax.z = v.z;
        } else if (tag == "f") {
            int3h f{};
            ss >> f.x >> f.y >> f.z;
            f.x -= 1; f.y -= 1; f.z -= 1;
            faces.push_back(f);
        }
    }
    return (verts.empty() || faces.empty()) ? -4 : 0;
}

extern "C" int optix_po_create(OptixPoContext **ctx_out) {
    if (!ctx_out) return -1;
    auto *ctx = new OptixPoContext{};

    OPTIX_CHECK(optixInit());
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&ctx->stream)));

    CUresult c0 = cuCtxGetCurrent(&ctx->cu_ctx);
    if (c0 != CUDA_SUCCESS) return -1;

    OptixDeviceContextOptions options{};
    OPTIX_CHECK(optixDeviceContextCreate(ctx->cu_ctx, &options, &ctx->optix_ctx));

    OptixModuleCompileOptions mo{};
    mo.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.usesMotionBlur = false;
    po.numPayloadValues = 4;
    po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params";

    char log[2048] = {};
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(ctx->optix_ctx, &mo, &po, kPtx, strlen(kPtx), log, &log_size, &ctx->module));

    OptixProgramGroupOptions pgo{};
    OptixProgramGroupDesc rg{};
    rg.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg.raygen.module = ctx->module;
    rg.raygen.entryFunctionName = "__raygen__rg";

    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx->optix_ctx, &rg, 1, &pgo, log, &log_size, &ctx->pg_raygen));

    OptixProgramGroupDesc ms{};
    ms.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms.miss.module = ctx->module;
    ms.miss.entryFunctionName = "__miss__ms";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx->optix_ctx, &ms, 1, &pgo, log, &log_size, &ctx->pg_miss));

    OptixProgramGroupDesc hg{};
    hg.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg.hitgroup.moduleCH = ctx->module;
    hg.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx->optix_ctx, &hg, 1, &pgo, log, &log_size, &ctx->pg_hit));

    OptixProgramGroup pgs[] = {ctx->pg_raygen, ctx->pg_miss, ctx->pg_hit};
    OptixPipelineLinkOptions plo{};
    plo.maxTraceDepth = 2;
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(ctx->optix_ctx, &po, &plo, pgs, 3, log, &log_size, &ctx->pipeline));

    RaygenRecord rg_rec{};
    MissRecord ms_rec{};
    HitgroupRecord hg_rec{};
    OPTIX_CHECK(optixSbtRecordPackHeader(ctx->pg_raygen, &rg_rec));
    OPTIX_CHECK(optixSbtRecordPackHeader(ctx->pg_miss, &ms_rec));
    OPTIX_CHECK(optixSbtRecordPackHeader(ctx->pg_hit, &hg_rec));

    CUdeviceptr d_rg = 0, d_ms = 0, d_hg = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_rg), sizeof(rg_rec)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ms), sizeof(ms_rec)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hg), sizeof(hg_rec)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_rg), &rg_rec, sizeof(rg_rec), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_ms), &ms_rec, sizeof(ms_rec), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hg), &hg_rec, sizeof(hg_rec), cudaMemcpyHostToDevice));

    ctx->sbt.raygenRecord = d_rg;
    ctx->sbt.missRecordBase = d_ms;
    ctx->sbt.missRecordStrideInBytes = sizeof(MissRecord);
    ctx->sbt.missRecordCount = 1;
    ctx->sbt.hitgroupRecordBase = d_hg;
    ctx->sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    ctx->sbt.hitgroupRecordCount = 1;

    *ctx_out = ctx;
    return 0;
}

extern "C" void optix_po_destroy(OptixPoContext *ctx) {
    if (!ctx) return;
    if (ctx->d_vertices) cudaFree(reinterpret_cast<void *>(ctx->d_vertices));
    if (ctx->d_indices) cudaFree(reinterpret_cast<void *>(ctx->d_indices));
    if (ctx->d_gas_output) cudaFree(reinterpret_cast<void *>(ctx->d_gas_output));
    if (ctx->d_ray_o) cudaFree(reinterpret_cast<void *>(ctx->d_ray_o));
    if (ctx->d_ray_d) cudaFree(reinterpret_cast<void *>(ctx->d_ray_d));
    if (ctx->d_ray_n) cudaFree(reinterpret_cast<void *>(ctx->d_ray_n));
    if (ctx->d_ray_t) cudaFree(reinterpret_cast<void *>(ctx->d_ray_t));
    if (ctx->d_params) cudaFree(reinterpret_cast<void *>(ctx->d_params));

    if (ctx->pipeline) optixPipelineDestroy(ctx->pipeline);
    if (ctx->pg_hit) optixProgramGroupDestroy(ctx->pg_hit);
    if (ctx->pg_miss) optixProgramGroupDestroy(ctx->pg_miss);
    if (ctx->pg_raygen) optixProgramGroupDestroy(ctx->pg_raygen);
    if (ctx->module) optixModuleDestroy(ctx->module);
    if (ctx->optix_ctx) optixDeviceContextDestroy(ctx->optix_ctx);
    if (ctx->stream) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(ctx->stream));

    delete ctx;
}

extern "C" int optix_po_load_obj(OptixPoContext *ctx, const char *filename) {
    if (!ctx || !filename) return -1;

    std::vector<float3h> vertices;
    std::vector<int3h> indices;
    int rc = load_obj(filename, vertices, indices, ctx->bb_min, ctx->bb_max);
    if (rc != 0) return rc;

    ctx->vertex_count = vertices.size();
    ctx->index_count = indices.size();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx->d_vertices), vertices.size() * sizeof(float3h)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx->d_indices), indices.size() * sizeof(int3h)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(ctx->d_vertices), vertices.data(), vertices.size() * sizeof(float3h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(ctx->d_indices), indices.data(), indices.size() * sizeof(int3h), cudaMemcpyHostToDevice));

    OptixBuildInput build_input{};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    CUdeviceptr d_vertex_buffers[] = {ctx->d_vertices};
    build_input.triangleArray.vertexBuffers = d_vertex_buffers;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3h);
    build_input.triangleArray.indexBuffer = ctx->d_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(indices.size());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(int3h);
    unsigned int flags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    build_input.triangleArray.flags = flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx->optix_ctx, &accel_options, &build_input, 1, &gas_sizes));

    CUdeviceptr d_temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp), gas_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ctx->d_gas_output), gas_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        ctx->optix_ctx,
        ctx->stream,
        &accel_options,
        &build_input,
        1,
        d_temp,
        gas_sizes.tempSizeInBytes,
        ctx->d_gas_output,
        gas_sizes.outputSizeInBytes,
        &ctx->gas_handle,
        nullptr,
        0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp)));
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx->stream)));
    return 0;
}

extern "C" OptixSimResult optix_po_simulate(
    OptixPoContext *ctx,
    float alpha_deg,
    float phi_deg,
    float theta_deg,
    float freq_hz,
    int rays_per_lambda,
    int max_bounces
) {
    (void)alpha_deg; (void)phi_deg; (void)theta_deg; (void)freq_hz; (void)rays_per_lambda; (void)max_bounces;
    OptixSimResult out{};
    if (!ctx || !ctx->gas_handle) return out;

    LaunchParams params{};
    params.gas = ctx->gas_handle;
    params.ray_count = 1;

    if (cudaMalloc(reinterpret_cast<void **>(&ctx->d_params), sizeof(LaunchParams)) != cudaSuccess) return out;
    if (cudaMemcpy(reinterpret_cast<void *>(ctx->d_params), &params, sizeof(params), cudaMemcpyHostToDevice) != cudaSuccess) return out;

    if (optixLaunch(ctx->pipeline, ctx->stream, ctx->d_params, sizeof(LaunchParams), &ctx->sbt, 1, 1, 1) != OPTIX_SUCCESS) return out;
    if (cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx->stream)) != cudaSuccess) return out;

    out.e_theta.re = 0.0f;
    out.e_theta.im = 0.0f;
    out.e_phi.re = 0.0f;
    out.e_phi.im = 0.0f;
    out.r0 = 1.0f;
    return out;
}
