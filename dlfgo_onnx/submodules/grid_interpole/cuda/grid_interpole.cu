#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ inline scalar_t get_val_padded(const scalar_t* grid, int c, int u, int v, int s, int t,
                                            int c_stride, int u_stride, int v_stride, int s_stride,
                                            int U, int V, int S, int T) {
    if(u < 0 || u >= U || v < 0 || v >= V || s < 0 || s >= S || t < 0 || t >= T)
        return 0;
    return grid[c * c_stride + u * u_stride + v * v_stride + s * s_stride + t];
}

template <typename scalar_t>
__global__ void grid_interpolate_1d_forward_kernel(
    const scalar_t* __restrict__ grid,   // [C * U]
    const scalar_t* __restrict__ rays,   // [N * 1]
    scalar_t* __restrict__ out,          // [N * C]
    int C, int U,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;  // ray index
    int c = idx % C;  // channel index

    scalar_t u_f = ((rays[n * 4 + 0] + 1) / 2) * (U - 1);
    int u0 = floorf(u_f);
    int u1 = u0 + 1;
    scalar_t du = u_f - u0;

    scalar_t c0 = get_val_padded<scalar_t>(grid, c, u0, 0, 0, 0, C, U, 1, 1, 1, 1, 1, 1);
    scalar_t c1 = get_val_padded<scalar_t>(grid, c, u1, 0, 0, 0, C, U, 1, 1, 1, 1, 1, 1);
    scalar_t val = c0 * (1 - du) + c1 * du;
    out[idx] = val;
}

template <typename scalar_t>
__global__ void grid_interpolate_1d_backward_kernel(
    const scalar_t* __restrict__ grad_out,  // [N * C]
    const scalar_t* __restrict__ rays,      // [N * 1]
    scalar_t* __restrict__ grad_grid,       // [C * U]
    int C, int U,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;  // ray index
    int c = idx % C;  // channel index

    scalar_t gVal = grad_out[idx]; // dL/d(Output[n,c])
    
    scalar_t u_f = ((rays[n * 4 + 0] + 1) / 2) * (U - 1);
    int u0 = floorf(u_f);
    int u1 = u0 + 1;
    scalar_t du = u_f - u0;

    auto idx_1d = [&](int uu) {
        return c * U + uu;
    };
    auto add_grad = [&](int uu, scalar_t w) {
        if (uu < 0 || uu >= U) return;
        atomicAdd(&grad_grid[idx_1d(uu)], w * gVal);
    };

    add_grad(u0, (1 - du));
    add_grad(u1, du);
}

void grid_interpole_1d_forward_cuda(
    torch::Tensor grid,  // [C, U]
    torch::Tensor rays,  // [N, 1]
    torch::Tensor output // [N, C]
) {
    int C = grid.size(0);
    int U = grid.size(1);
    int N = rays.size(0);

    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "grid_interpole_1d_forward_cuda", ([&] {
        grid_interpolate_1d_forward_kernel<scalar_t><<<blocks, threads>>>(
            grid.data_ptr<scalar_t>(),
            rays.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, U,
            N
        );
    }));
}

void grid_interpole_1d_backward_cuda(
    torch::Tensor grad_out,  // [N, C]
    torch::Tensor rays,      // [N, 1]
    torch::Tensor grad_grid  // [C, U]
) {
    int C = grad_grid.size(0);
    int U = grad_grid.size(1);

    int N = grad_out.size(0);

    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "grid_interpole_1d_backward_cuda", ([&] {
        grid_interpolate_1d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            rays.data_ptr<scalar_t>(),
            grad_grid.data_ptr<scalar_t>(),
            C, U,
            N
        );
    }));
}

template <typename scalar_t>
__global__ void grid_interpolate_4d_forward_kernel(
    const scalar_t* __restrict__ grid,   // [C * U * V * S * T]
    const scalar_t* __restrict__ rays,   // [N * 4]
    scalar_t* __restrict__ out,          // [N * C]
    int C, int U, int V, int S, int T,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;  // ray index
    int c = idx % C;  // channel index

    scalar_t u_f = ((rays[n * 4 + 0] + 1) / 2) * (U - 1);
    scalar_t v_f = ((rays[n * 4 + 1] + 1) / 2) * (V - 1);
    scalar_t s_f = ((rays[n * 4 + 2] + 1) / 2) * (S - 1);
    scalar_t t_f = ((rays[n * 4 + 3] + 1) / 2) * (T - 1);
    
    int u0 = floorf(u_f);
    int v0 = floorf(v_f);
    int s0 = floorf(s_f);
    int t0 = floorf(t_f);

    int u1 = u0 + 1;
    int v1 = v0 + 1;
    int s1 = s0 + 1;
    int t1 = t0 + 1;

    scalar_t du = u_f - u0;
    scalar_t dv = v_f - v0;
    scalar_t ds = s_f - s0;
    scalar_t dt = t_f - t0;

    int c_stride = U * V * S * T;
    int u_stride = V * S * T;
    int v_stride = S * T;
    int s_stride = T;

    scalar_t c0000 = get_val_padded<scalar_t>(grid, c, u0, v0, s0, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1000 = get_val_padded<scalar_t>(grid, c, u1, v0, s0, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0100 = get_val_padded<scalar_t>(grid, c, u0, v1, s0, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1100 = get_val_padded<scalar_t>(grid, c, u1, v1, s0, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0010 = get_val_padded<scalar_t>(grid, c, u0, v0, s1, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1010 = get_val_padded<scalar_t>(grid, c, u1, v0, s1, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0110 = get_val_padded<scalar_t>(grid, c, u0, v1, s1, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1110 = get_val_padded<scalar_t>(grid, c, u1, v1, s1, t0, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0001 = get_val_padded<scalar_t>(grid, c, u0, v0, s0, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1001 = get_val_padded<scalar_t>(grid, c, u1, v0, s0, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0101 = get_val_padded<scalar_t>(grid, c, u0, v1, s0, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1101 = get_val_padded<scalar_t>(grid, c, u1, v1, s0, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0011 = get_val_padded<scalar_t>(grid, c, u0, v0, s1, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1011 = get_val_padded<scalar_t>(grid, c, u1, v0, s1, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c0111 = get_val_padded<scalar_t>(grid, c, u0, v1, s1, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    scalar_t c1111 = get_val_padded<scalar_t>(grid, c, u1, v1, s1, t1, c_stride, u_stride, v_stride, s_stride, U, V, S, T);
    
    scalar_t c000 = c0000 * (1 - du) + c1000 * du;
    scalar_t c100 = c0100 * (1 - du) + c1100 * du;
    scalar_t c010 = c0010 * (1 - du) + c1010 * du;
    scalar_t c110 = c0110 * (1 - du) + c1110 * du;
    scalar_t c001 = c0001 * (1 - du) + c1001 * du;
    scalar_t c101 = c0101 * (1 - du) + c1101 * du;
    scalar_t c011 = c0011 * (1 - du) + c1011 * du;
    scalar_t c111 = c0111 * (1 - du) + c1111 * du;

    scalar_t c00 = c000 * (1 - dv) + c100 * dv;
    scalar_t c10 = c010 * (1 - dv) + c110 * dv;
    scalar_t c01 = c001 * (1 - dv) + c101 * dv;
    scalar_t c11 = c011 * (1 - dv) + c111 * dv;

    scalar_t c0 = c00 * (1 - ds) + c10 * ds;
    scalar_t c1 = c01 * (1 - ds) + c11 * ds;

    scalar_t val = c0 * (1 - dt) + c1 * dt;

    out[idx] = val;
}

template <typename scalar_t>
__global__ void grid_interpolate_4d_backward_kernel(
    const scalar_t* __restrict__ grad_out,  // [N * C], dL/d(Output)
    const scalar_t* __restrict__ rays,      // [N, 4]
    scalar_t* __restrict__ grad_grid,       // [C, U, V, S, T], dL/d(grid)
    int C, int U, int V, int S, int T,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C) return;

    int n = idx / C;  // ray index
    int c = idx % C;  // channel index

    scalar_t gVal = grad_out[idx]; // dL/d(Output[n,c])

    // 동일하게 ray -> (u_f,v_f,s_f,t_f)
    scalar_t u_f = ((rays[n * 4 + 0] + 1) / 2) * (U - 1);
    scalar_t v_f = ((rays[n * 4 + 1] + 1) / 2) * (V - 1);
    scalar_t s_f = ((rays[n * 4 + 2] + 1) / 2) * (S - 1);
    scalar_t t_f = ((rays[n * 4 + 3] + 1) / 2) * (T - 1);

    int u0 = floorf(u_f);
    int v0 = floorf(v_f);
    int s0 = floorf(s_f);
    int t0 = floorf(t_f);

    int u1 = u0 + 1;
    int v1 = v0 + 1;
    int s1 = s0 + 1;
    int t1 = t0 + 1;

    scalar_t du = u_f - u0;
    scalar_t dv = v_f - v0;
    scalar_t ds = s_f - s0;
    scalar_t dt = t_f - t0;

    int c_stride = U * V * S * T;
    int u_stride = V * S * T;
    int v_stride = S * T;
    int s_stride = T;

    auto idx_5d = [&](int cc, int uu, int vv, int ss, int tt) {
        return cc * (size_t)c_stride 
             + uu * (size_t)u_stride
             + vv * (size_t)v_stride
             + ss * (size_t)s_stride
             + tt;
    };
    auto add_grad = [&](int uu, int vv, int ss, int tt, scalar_t w) {
    if (uu < 0 || uu >= U || vv < 0 || vv >= V || ss < 0 || ss >= S || tt < 0 || tt >= T)
        return 0;
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
        atomicAdd((float*)&grad_grid[idx_5d(c, uu, vv, ss, tt)], static_cast<float>(w * gVal));
    #else
        atomicAdd(&grad_grid[idx_5d(c, uu, vv, ss, tt)], w * gVal);
    #endif
    };
    
    add_grad(u0, v0, s0, t0, (1 - du)*(1 - dv)*(1 - ds)*(1 - dt));
    add_grad(u1, v0, s0, t0, du*(1 - dv)*(1 - ds)*(1 - dt));
    add_grad(u0, v1, s0, t0, (1 - du)*dv*(1 - ds)*(1 - dt));
    add_grad(u1, v1, s0, t0, du*dv*(1 - ds)*(1 - dt));
    add_grad(u0, v0, s1, t0, (1 - du)*(1 - dv)*ds*(1 - dt));
    add_grad(u1, v0, s1, t0, du*(1 - dv)*ds*(1 - dt));
    add_grad(u0, v1, s1, t0, (1 - du)*dv*ds*(1 - dt));
    add_grad(u1, v1, s1, t0, du*dv*ds*(1 - dt));
    add_grad(u0, v0, s0, t1, (1 - du)*(1 - dv)*(1 - ds)*dt);
    add_grad(u1, v0, s0, t1, du*(1 - dv)*(1 - ds)*dt);
    add_grad(u0, v1, s0, t1, (1 - du)*dv*(1 - ds)*dt);
    add_grad(u1, v1, s0, t1, du*dv*(1 - ds)*dt);
    add_grad(u0, v0, s1, t1, (1 - du)*(1 - dv)*ds*dt);
    add_grad(u1, v0, s1, t1, du*(1 - dv)*ds*dt);
    add_grad(u0, v1, s1, t1, (1 - du)*dv*ds*dt);
    add_grad(u1, v1, s1, t1, du*dv*ds*dt);
}



void grid_interpole_4d_forward_cuda(
    torch::Tensor grid,  // [C, U, V, S, T]
    torch::Tensor rays,  // [N, 4], in [0,1]
    torch::Tensor output // [N, C]
) {
    int C = grid.size(0);
    int U = grid.size(1);
    int V = grid.size(2);
    int S = grid.size(3);
    int T = grid.size(4);

    int N = rays.size(0);
    // rays.size(1) should be 4

    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "grid_interpole_4d_forward_cuda", ([&] {
        grid_interpolate_4d_forward_kernel<scalar_t><<<blocks, threads>>>(
            grid.data_ptr<scalar_t>(),
            rays.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C, U, V, S, T,
            N
        );
    }));
}

void grid_interpole_4d_backward_cuda(
    torch::Tensor grad_out,  // [N, C]
    torch::Tensor rays,      // [N, 4]
    torch::Tensor grad_grid  // [C, U, V, S, T]
) {
    int C = grad_grid.size(0);
    int U = grad_grid.size(1);
    int V = grad_grid.size(2);
    int S = grad_grid.size(3);
    int T = grad_grid.size(4);

    int N = grad_out.size(0);
    // grad_out.size(1) == C

    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "grid_interpole_4d_backward_cuda", ([&] {
        grid_interpolate_4d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            rays.data_ptr<scalar_t>(),
            grad_grid.data_ptr<scalar_t>(),
            C, U, V, S, T,
            N
        );
    }));
}
