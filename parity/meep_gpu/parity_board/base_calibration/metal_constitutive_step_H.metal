
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

kernel void constitutive_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       w0      [[buffer(3)]],
    device float*       w1      [[buffer(4)]],
    device float*       w2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* e0      [[buffer(9)]],
    device const float* e1      [[buffer(10)]],
    device const float* e2      [[buffer(11)]],
    device const float* kp0     [[buffer(12)]],
    device const float* km0     [[buffer(13)]],
    device const float* kp1     [[buffer(14)]],
    device const float* km1     [[buffer(15)]],
    device const float* kp2     [[buffer(16)]],
    device const float* km2     [[buffer(17)]],
    constant uint&      nx      [[buffer(18)]],
    constant uint&      ny      [[buffer(19)]],
    constant uint&      nz      [[buffer(20)]],
    constant uint&      n_elem  [[buffer(21)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }

    int nyi = int(ny), nzi = int(nz);
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // The coefficient index is on the component's OWN axis: kps_x/kms_x for
    // target 0, _y for 1, _z for 2 (H_CONSTITUTIVE_TERMS / E_CONSTITUTIVE_TERMS,
    // stepping.py:226-227). That is MEEP's dsigw, NOT the dsig/dsigu cycle the
    // curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];

    // --- component 0 -----------------------------------------------------------
    // `prev` is read BEFORE `fw` is written: the array path copies fw first
    // (stepping.py:2090). Reversed, this is wrong only where kms != 0 -- INSIDE
    // THE PML ONLY -- and looks like a slightly worse absorber, not like a bug.
    float prev0 = w0[ii];
    float src0 = g0[ii];
    w0[ii] = src0;
    float a0 = f0[ii];
    // Two SEPARATE accumulations, left to right: ((f + kps*src) - kms*prev).
    // Flattening to f + (kps*src - kms*prev) is a different float32 number, and
    // neither multiply may contract into an fma -- the contraction pragma is what
    // stops the second.
    a0 = a0 + kp_0 * src0;
    a0 = a0 - km_0 * prev0;
    f0[ii] = a0;

    // --- component 1 -----------------------------------------------------------
    float prev1 = w1[ii];
    float src1 = g1[ii];
    w1[ii] = src1;
    float a1 = f1[ii];
    a1 = a1 + kp_1 * src1;
    a1 = a1 - km_1 * prev1;
    f1[ii] = a1;

    // --- component 2 -----------------------------------------------------------
    float prev2 = w2[ii];
    float src2 = g2[ii];
    w2[ii] = src2;
    float a2 = f2[ii];
    a2 = a2 + kp_2 * src2;
    a2 = a2 - km_2 * prev2;
    f2[ii] = a2;
}
