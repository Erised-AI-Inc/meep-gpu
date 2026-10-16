
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

kernel void pml_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* kmx     [[buffer(9)]],
    device const float* sinvx   [[buffer(10)]],
    device const float* kmy     [[buffer(11)]],
    device const float* sinvy   [[buffer(12)]],
    device const float* kmz     [[buffer(13)]],
    device const float* sinvz   [[buffer(14)]],
    constant uint&      nx      [[buffer(15)]],
    constant uint&      ny      [[buffer(16)]],
    constant uint&      nz      [[buffer(17)]],
    constant uint&      n_elem  [[buffer(18)]],
    constant float&     dtdx    [[buffer(19)]],
    uint idx [[thread_position_in_grid]])
{
    // The dispatch is sized from the first tensor argument's element count, so a
    // volume wider than n_elem would step cells the array path does not own.
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    int si = i + 1, sj = j + 1, sk = k + 1;
    bool vx = true, vy = true, vz = true;
    vx = (si >= 0) && (si < nxi);
    vy = (sj >= 0) && (sj < nyi);
    vz = (sk >= 0) && (sk < nzi);

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact 0.0 past the wall -- Triton's `other=0.0`.
    float a   = g0[ii];
    float b   = g1[ii];
    float c   = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f;
    float a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f;
    float b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f;
    float c_y = vy ? g2[oy] : 0.0f;

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
    float curl2 = dtdx * ((b_x - b) + (a - a_y));

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
    curl0 = at_x ? 0.0f : curl0;
    curl1 = at_y ? 0.0f : curl1;
    curl2 = at_z ? 0.0f : curl2;

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float p0 = u0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;

    float p1 = u1[ii];
    float n1 = ((p1 * km_z) - curl1) * si_z;
    float v1 = (((f1[ii] * km_x) + n1) - p1) * si_x;

    float p2 = u2[ii];
    float n2 = ((p2 * km_x) - curl2) * si_x;
    float v2 = (((f2[ii] * km_y) + n2) - p2) * si_y;

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
