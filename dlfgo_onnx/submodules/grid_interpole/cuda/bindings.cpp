#include <torch/extension.h>

void grid_interpole_1d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_1d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);

void grid_interpole_1d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    grid_interpole_1d_forward_cuda(grid, rays, output);
}

void grid_interpole_1d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_1d_backward_cuda(grad_out, rays, grad_grid);
}

torch::Tensor grid_interpolate_1d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    TORCH_CHECK(grid.dim() == 2, "grid must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 1, "rays must be of shape [N, 1]");
    auto output = torch::zeros({rays.size(0), grid.size(1)}, grid.options());
    grid_interpole_1d_forward_cuda(grid, rays, output);
    return output;
}

torch::Tensor grid_interpolate_1d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 1, "rays must be of shape [N, 1]");
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0]}, grad_out.options());
    grid_interpole_1d_backward_cuda(grad_out, rays, grad_grid);
    return grad_grid;
}


// 여기서 위 .cu의 두 함수를 extern으로 선언
void grid_interpole_4d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_4d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);


// 래퍼
void grid_interpole_4d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    // 원하는 경우 CHECK_CUDA 등 체크
    grid_interpole_4d_forward_cuda(grid, rays, output);
}

void grid_interpole_4d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_4d_backward_cuda(grad_out, rays, grad_grid);
}

// 4D 그리드 보간 forward 함수
torch::Tensor grid_interpolate_4d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    // 입력 텐서의 크기 확인
    TORCH_CHECK(grid.dim() == 5, "grid must be 5D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 4, "rays must be of shape [N, 4]");
    
    // 출력 텐서 생성
    auto output = torch::zeros({rays.size(0), grid.size(0)}, grid.options());
    
    // CUDA 함수 호출
    grid_interpole_4d_forward_cuda(grid, rays, output);
    
    return output;
}

// 4D 그리드 보간 backward 함수
torch::Tensor grid_interpolate_4d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    // 입력 텐서의 크기 확인
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 4, "rays must be of shape [N, 4]");
    
    // 출력 텐서 생성
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0], grid_size[1], grid_size[2], grid_size[3]}, 
                                 grad_out.options());
    
    // CUDA 함수 호출
    grid_interpole_4d_backward_cuda(grad_out, rays, grad_grid);
    
    return grad_grid;
}

// pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_1d", &grid_interpole_1d_forward, "1D grid interpolate forward (CUDA)");
    m.def("backward_1d", &grid_interpole_1d_backward, "1D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_1d", &grid_interpolate_1d, "1D grid interpolation (CUDA)");
    m.def("grid_interpolate_1d_backward", &grid_interpolate_1d_backward, "Backward pass for 1D grid interpolation (CUDA)");

    m.def("forward_4d", &grid_interpole_4d_forward, "4D grid interpolate forward (CUDA)");
    m.def("backward_4d", &grid_interpole_4d_backward, "4D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_4d", &grid_interpolate_4d, "4D grid interpolation (CUDA)");
    m.def("grid_interpolate_4d_backward", &grid_interpolate_4d_backward, "Backward pass for 4D grid interpolation (CUDA)");
}
