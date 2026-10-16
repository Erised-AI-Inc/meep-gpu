
#define BC_PERIODIC 0
#define BC_METALLIC 1
// A MIRROR-FOLDED axis whose termination is PERIODIC -- stepping._boundary_kinds
// resolves it to "mirror" and stepping._stored_past_owned says the stored array
// carries one slot past MEEP's owned window. A folded METALLIC axis is handed
// BC_METALLIC instead and has no code of its own: its ghost rule, its cell-0 mask
// and its (absent) top-plane mask are METALLIC's, arithmetic for arithmetic, and
// that coincidence was measured 48/48 rather than assumed.
#define BC_MIRROR_PERIODIC 2

// stepping._shift_up: the neighbour one cell up, with that axis's far-face rule.
__device__ __forceinline__ float shift_up(
    const float* __restrict__ g, int idx, int ia, int na, int stride, int bc
) {
    if (ia + 1 < na) return g[idx + stride];
    return (bc == BC_PERIODIC) ? g[idx - ia * stride] : 0.0f;
}

// stepping._shift_down: the neighbour one cell below, with that axis's near-face rule.
__device__ __forceinline__ float shift_dn(
    const float* __restrict__ g, int idx, int ia, int na, int stride, int bc
) {
    if (ia > 0) return g[idx - stride];
    return (bc == BC_PERIODIC) ? g[idx + (na - 1) * stride] : 0.0f;
}

// stepping._apply_pml_update, in its in-place order:
//   fu *= kms; fu -= curl; fu *= sinv;
//   f *= kms_u; f += fu; f -= fu_previous; f *= sinv_u
__device__ __forceinline__ void pml_apply(
    float* __restrict__ f, float* __restrict__ fu, int idx, float curl,
    float kms, float sinv, float kms_u, float sinv_u
) {
    float fprev = fu[idx];
    float fu_new = ((fprev * kms) - curl) * sinv;
    fu[idx] = fu_new;
    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;
}

extern "C" __global__ void step_B_pml_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> cell 0 on x,
    // and on a folded PERIODIC axis the top plane on y and z.
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> cell 0 on y,
    // top plane on z and x.
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: curl_z = dEy/dx - dEx/dy; dsig=x, dsigu=y; iyee=(1,1,0) -> cell 0 on z,
    // top plane on x and y.
    {
        float f1 = Ey[idx];
        float sf = shift_up(Ey, idx, i, nx, sx, bc_x);
        float f2 = Ex[idx];
        float ss = shift_up(Ex, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
