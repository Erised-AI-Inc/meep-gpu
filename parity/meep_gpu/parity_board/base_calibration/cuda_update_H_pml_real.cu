
// stepping._apply_constitutive_pml (stepping.py:2065), MEEP's step_update_EDHB
// with dsigw active:
//
//     realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
//     fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
//     f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
//
// The two accumulations are kept SEPARATE and LEFT-TO-RIGHT (note 1 above);
// `prev` is loaded before the store (note 3). The intermediate `a` is a
// register rather than a second read-modify-write of f[idx]: a float32 stored
// to global memory and loaded back is the identity on the bits, which the
// sibling track measured as its own null mutation (reload_fu_from_memory,
// 60/60 identical) on the curl's equivalent expression.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms
) {
    float prev = fw[idx];
    fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}

extern "C" __global__ void update_H_pml_real(
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ f_w_Hx, float* __restrict__ f_w_Hy,
    float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // THE OWN-CELL HOIST (register view): every word this thread reads at idx,
    // loaded before the first store. The certified constitutive_apply steps the
    // registers (f = &field, fw = &aux, idx = 0); the write-back stores them.
    float w_x = f_w_Hx[idx];
    float h_x = Hx[idx];
    float w_y = f_w_Hy[idx];
    float h_y = Hy[idx];
    float w_z = f_w_Hz[idx];
    float h_z = Hz[idx];
    float src_x = Bx[idx];
    float src_y = By[idx];
    float src_z = Bz[idx];

    // Hx <- Bx, dsigw = x.  Hy <- By, dsigw = y.  Hz <- Bz, dsigw = z.
    constitutive_apply(&h_x, &w_x, 0, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(&h_y, &w_y, 0, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(&h_z, &w_z, 0, src_z, kps_z[k], kms_z[k]);

    // The write-back, in the certified store order (auxiliary, then field).
    f_w_Hx[idx] = w_x;
    Hx[idx] = h_x;
    f_w_Hy[idx] = w_y;
    Hy[idx] = h_y;
    f_w_Hz[idx] = w_z;
    Hz[idx] = h_z;
}
