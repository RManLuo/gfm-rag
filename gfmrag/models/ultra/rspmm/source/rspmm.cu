#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#include "util.cuh"
#include "operator.cuh"
#include "rspmm.h"

namespace at {

// Memory & time efficient implementation of generalized spmm
// Much of the code is inspired by GE-SpMM
// https://github.com/hgyhungry/ge-spmm

namespace {

const int kCoarseningFactor = 2;
const int kThreadPerBlock = 256;

} // namespace anonymous

template <class scalar_t, class NaryOp, class BinaryOp>
__global__
void rspmm_forward_out_cuda(const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
                            const scalar_t *weight, const scalar_t *relation, const scalar_t *input,
                            scalar_t *output,
                            int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    int64_t *layer_ind_buf = buffer + blockDim.y * warpSize;
    scalar_t *weight_buf = reinterpret_cast<scalar_t *>(layer_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    layer_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;
    scalar_t out[kCoarseningFactor];
#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++)
        out[i] = get_nary_zero<NaryOp, scalar_t>();

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            layer_ind_buf[threadIdx.x] = layer_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            int64_t layer = layer_ind_buf[offset_ptr];
            scalar_t w = weight_buf[offset_ptr];
#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                scalar_t x = BinaryOp::forward(relation[layer * dim + d], input[col * dim + d]);
                scalar_t y = w * x;
                out[i] = NaryOp::forward(out[i], y);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        int64_t d = d_start + i * warpSize;
        if (d >= dim)
            break;
        output[row * dim + d] = out[i];
    }
}

// mixed precision version with weight gradients - FP32 accumulation + local accumulation
template <class input_t, class grad_t, class NaryOp, class BinaryOp>
__global__
void rspmm_backward_mixed_precision_cuda(const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
                                         const input_t *weight, const input_t *relation, const input_t *input,
                                         const input_t *output, const input_t *output_grad,
                                         grad_t *weight_grad, grad_t *relation_grad, grad_t *input_grad,
                                         int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    int64_t *layer_ind_buf = col_ind_buf + blockDim.y * warpSize;
    input_t *weight_buf = reinterpret_cast<input_t *>(layer_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    layer_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    // local accumulation buffers to reduce atomic operations
    grad_t local_rel_grad[kCoarseningFactor];
    grad_t local_in_grad[kCoarseningFactor];
    int64_t last_layer = -1;
    int64_t last_col = -1;

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        local_rel_grad[i] = 0;
        local_in_grad[i] = 0;
    }

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            layer_ind_buf[threadIdx.x] = layer_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            int64_t layer = layer_ind_buf[offset_ptr];
            input_t w = weight_buf[offset_ptr];
            grad_t w_grad = 0;

            // flush local accumulation if indices change
            if (layer != last_layer && last_layer != -1) {
#pragma unroll
                for (int64_t i = 0; i < kCoarseningFactor; i++) {
                    int64_t d = d_start + i * warpSize;
                    if (d < dim && local_rel_grad[i] != 0) {
                        atomicAdd(&relation_grad[last_layer * dim + d], local_rel_grad[i]);
                        local_rel_grad[i] = 0;
                    }
                }
            }
            if (col != last_col && last_col != -1) {
#pragma unroll
                for (int64_t i = 0; i < kCoarseningFactor; i++) {
                    int64_t d = d_start + i * warpSize;
                    if (d < dim && local_in_grad[i] != 0) {
                        atomicAdd(&input_grad[last_col * dim + d], local_in_grad[i]);
                        local_in_grad[i] = 0;
                    }
                }
            }

#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                input_t rel = relation[layer * dim + d];
                input_t in = input[col * dim + d];
                input_t out = output[row * dim + d];
                input_t out_grad = output_grad[row * dim + d];
                input_t x = BinaryOp::forward(rel, in);
                input_t y = w * x;
                input_t dx_drel = BinaryOp::backward_lhs(rel, in);
                input_t dx_din = BinaryOp::backward_rhs(rel, in);
                input_t dout_dy = NaryOp::backward(out, y);
                input_t dy_dw = x;
                input_t dy_dx = w;
                w_grad += grad_t(out_grad * dout_dy * dy_dw);

                local_rel_grad[i] += grad_t(out_grad * dout_dy * dy_dx * dx_drel);
                local_in_grad[i] += grad_t(out_grad * dout_dy * dy_dx * dx_din);
            }
            w_grad = warp_reduce(w_grad);
            if (threadIdx.x == 0)
                atomicAdd(&weight_grad[block_ptr + offset_ptr], w_grad);

            last_layer = layer;
            last_col = col;
        }
        __syncwarp();
    }

    if (last_layer != -1) {
#pragma unroll
        for (int64_t i = 0; i < kCoarseningFactor; i++) {
            int64_t d = d_start + i * warpSize;
            if (d < dim && local_rel_grad[i] != 0) {
                atomicAdd(&relation_grad[last_layer * dim + d], local_rel_grad[i]);
            }
        }
    }
    if (last_col != -1) {
#pragma unroll
        for (int64_t i = 0; i < kCoarseningFactor; i++) {
            int64_t d = d_start + i * warpSize;
            if (d < dim && local_in_grad[i] != 0) {
                atomicAdd(&input_grad[last_col * dim + d], local_in_grad[i]);
            }
        }
    }
}

// mixed precision version without weight gradients - FP32 accumulation + local accumulation
template <class input_t, class grad_t, class NaryOp, class BinaryOp>
__global__
void rspmm_backward_mixed_precision_cuda(const int64_t *row_ptr, const int64_t *col_ind, const int64_t *layer_ind,
                                         const input_t *weight, const input_t *relation, const input_t *input,
                                         const input_t *output, const input_t *output_grad,
                                         grad_t *relation_grad, grad_t *input_grad,
                                         int64_t num_row, int64_t nnz, int64_t dim) {
    // for best optimization, the following code is compiled with constant warpSize
    assert(blockDim.x == warpSize);

    extern __shared__ int64_t buffer[];
    int64_t *col_ind_buf = buffer;
    int64_t *layer_ind_buf = col_ind_buf + blockDim.y * warpSize;
    input_t *weight_buf = reinterpret_cast<input_t *>(layer_ind_buf + blockDim.y * warpSize);
    col_ind_buf += threadIdx.y * warpSize;
    layer_ind_buf += threadIdx.y * warpSize;
    weight_buf += threadIdx.y * warpSize;

    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_row)
        return;
    int64_t d_start = blockIdx.y * warpSize * kCoarseningFactor + threadIdx.x;
    int64_t ptr_start = row_ptr[row];
    int64_t ptr_end = row + 1 < num_row ? row_ptr[row + 1] : nnz;

    grad_t local_rel_grad[kCoarseningFactor];
    grad_t local_in_grad[kCoarseningFactor];
    int64_t last_layer = -1;
    int64_t last_col = -1;

#pragma unroll
    for (int64_t i = 0; i < kCoarseningFactor; i++) {
        local_rel_grad[i] = 0;
        local_in_grad[i] = 0;
    }

    for (int64_t block_ptr = ptr_start; block_ptr < ptr_end; block_ptr += warpSize) {
        int64_t ptr = block_ptr + threadIdx.x;
        if (ptr < ptr_end) {
            col_ind_buf[threadIdx.x] = col_ind[ptr];
            layer_ind_buf[threadIdx.x] = layer_ind[ptr];
            weight_buf[threadIdx.x] = weight[ptr];
        }
        __syncwarp();

        int64_t max_offset = warpSize < ptr_end - block_ptr ? warpSize : ptr_end - block_ptr;
        for (int64_t offset_ptr = 0; offset_ptr < max_offset; offset_ptr++) {
            int64_t col = col_ind_buf[offset_ptr];
            int64_t layer = layer_ind_buf[offset_ptr];
            input_t w = weight_buf[offset_ptr];

            if (layer != last_layer && last_layer != -1) {
#pragma unroll
                for (int64_t i = 0; i < kCoarseningFactor; i++) {
                    int64_t d = d_start + i * warpSize;
                    if (d < dim && local_rel_grad[i] != 0) {
                        atomicAdd(&relation_grad[last_layer * dim + d], local_rel_grad[i]);
                        local_rel_grad[i] = 0;
                    }
                }
            }
            if (col != last_col && last_col != -1) {
#pragma unroll
                for (int64_t i = 0; i < kCoarseningFactor; i++) {
                    int64_t d = d_start + i * warpSize;
                    if (d < dim && local_in_grad[i] != 0) {
                        atomicAdd(&input_grad[last_col * dim + d], local_in_grad[i]);
                        local_in_grad[i] = 0;
                    }
                }
            }

#pragma unroll
            for (int64_t i = 0; i < kCoarseningFactor; i++) {
                int64_t d = d_start + i * warpSize;
                if (d >= dim)
                    break;
                input_t rel = relation[layer * dim + d];
                input_t in = input[col * dim + d];
                input_t out = output[row * dim + d];
                input_t out_grad = output_grad[row * dim + d];
                input_t x = BinaryOp::forward(rel, in);
                input_t y = w * x;
                input_t dx_drel = BinaryOp::backward_lhs(rel, in);
                input_t dx_din = BinaryOp::backward_rhs(rel, in);
                input_t dout_dy = NaryOp::backward(out, y);
                input_t dy_dx = w;

                local_rel_grad[i] += grad_t(out_grad * dout_dy * dy_dx * dx_drel);
                local_in_grad[i] += grad_t(out_grad * dout_dy * dy_dx * dx_din);
            }

            last_layer = layer;
            last_col = col;
        }
        __syncwarp();
    }

    if (last_layer != -1) {
#pragma unroll
        for (int64_t i = 0; i < kCoarseningFactor; i++) {
            int64_t d = d_start + i * warpSize;
            if (d < dim && local_rel_grad[i] != 0) {
                atomicAdd(&relation_grad[last_layer * dim + d], local_rel_grad[i]);
            }
        }
    }
    if (last_col != -1) {
#pragma unroll
        for (int64_t i = 0; i < kCoarseningFactor; i++) {
            int64_t d = d_start + i * warpSize;
            if (d < dim && local_in_grad[i] != 0) {
                atomicAdd(&input_grad[last_col * dim + d], local_in_grad[i]);
            }
        }
    }
}

template <template<class> class NaryOp, template<class> class BinaryOp>
Tensor rspmm_forward_cuda(const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_,
                          const Tensor &relation_, const Tensor &input_) {
    constexpr const char *fn_name = "rspmm_forward_cuda";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), relation_arg(relation_, "relation", 4),
              input_arg(input_, "input", 5);

    rspmm_forward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, relation_arg, input_arg);
    checkAllSameGPU(fn_name, {edge_index_arg, edge_type_arg, edge_weight_arg, relation_arg, input_arg});

    const Tensor edge_index = edge_index_.contiguous();
    const Tensor edge_type = edge_type_.contiguous();
    // Convert tensors to input type for mixed precision support
    const Tensor edge_weight = edge_weight_.to(input_.scalar_type()).contiguous();
    const Tensor relation = relation_.to(input_.scalar_type()).contiguous();
    const Tensor input = input_.contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);
    Tensor output = at::empty({num_row, dim}, input.options());

    Tensor row_ind = edge_index.select(0, 0);
    Tensor row_ptr = ind2ptr(row_ind, num_row);
    Tensor col_ind = edge_index.select(0, 1);
    Tensor layer_ind = edge_type;

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rspmm_forward_cuda", [&]() {
        const int memory_size = kThreadPerBlock * (sizeof(int64_t) * 2 + sizeof(scalar_t));
        rspmm_forward_out_cuda<scalar_t, NaryOp<scalar_t>, BinaryOp<scalar_t>>
            <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
            row_ptr.data_ptr<int64_t>(),
            col_ind.data_ptr<int64_t>(),
            layer_ind.data_ptr<int64_t>(),
            edge_weight.data_ptr<scalar_t>(),
            relation.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_row, nnz, dim
        );
    });

    return output;
}

template <template<class> class NaryOp, template<class> class BinaryOp>
std::tuple<Tensor, Tensor, Tensor> rspmm_backward_cuda(
        const Tensor &edge_index_, const Tensor &edge_type_, const Tensor &edge_weight_,
        const Tensor &relation_, const Tensor &input_, const Tensor &output_, const Tensor &output_grad_) {
    constexpr const char *fn_name = "rspmm_backward_cuda";
    TensorArg edge_index_arg(edge_index_, "edge_index", 1), edge_type_arg(edge_type_, "edge_type", 2),
              edge_weight_arg(edge_weight_, "edge_weight", 3), relation_arg(relation_, "relation", 4),
              input_arg(input_, "input", 5), output_arg(output_, "output", 6),
              output_grad_arg(output_grad_, "output_grad", 7);

    rspmm_backward_check(fn_name, edge_index_arg, edge_type_arg, edge_weight_arg, relation_arg, input_arg,
                         output_arg, output_grad_arg);
    checkAllSameGPU(fn_name, {edge_index_arg, edge_type_arg, edge_weight_arg, relation_arg, input_arg, output_arg,
                              output_grad_arg});

    const Tensor edge_index = edge_index_.contiguous();
    const Tensor edge_type = edge_type_.contiguous();
    // Convert tensors to input type for mixed precision support
    const Tensor edge_weight = edge_weight_.to(input_.scalar_type()).contiguous();
    const Tensor relation = relation_.to(input_.scalar_type()).contiguous();
    const Tensor input = input_.contiguous();
    const Tensor output = output_.to(input_.scalar_type()).contiguous();
    const Tensor output_grad = output_grad_.to(input_.scalar_type()).contiguous();

    int64_t nnz = edge_index.size(1);
    int64_t num_row = input.size(0);
    int64_t dim = input.size(1);
    // Always use FP32 for gradient accumulation to avoid slow bfloat16 atomics
    Tensor weight_grad = at::zeros_like(edge_weight.to(at::ScalarType::Float));
    Tensor relation_grad = at::zeros_like(relation.to(at::ScalarType::Float));
    Tensor input_grad = at::zeros_like(input.to(at::ScalarType::Float));

    Tensor row_ind = edge_index.select(0, 0);
    Tensor row_ptr = ind2ptr(row_ind, num_row);
    Tensor col_ind = edge_index.select(0, 1);
    Tensor layer_ind = edge_type;

    cudaSetDevice(input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int dim_per_block = 32; // warpSize
    const int num_dim_block = (dim + dim_per_block * kCoarseningFactor - 1) / (dim_per_block * kCoarseningFactor);
    const int row_per_block = kThreadPerBlock / dim_per_block;
    const int num_row_block = (num_row + row_per_block - 1) / row_per_block;

    if (edge_weight.requires_grad()) {
        // Use mixed-precision kernel: input precision for inputs, FP32 for gradients
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rspmm_backward_cuda", [&]() {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) * 2 + sizeof(scalar_t));
            rspmm_backward_mixed_precision_cuda<scalar_t, float, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_ind.data_ptr<int64_t>(),
                layer_ind.data_ptr<int64_t>(),
                edge_weight.data_ptr<scalar_t>(),
                relation.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_grad.data_ptr<scalar_t>(),
                weight_grad.data_ptr<float>(),
                relation_grad.data_ptr<float>(),
                input_grad.data_ptr<float>(),
                num_row, nnz, dim
            );
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "rspmm_backward_cuda", [&]() {
            const int memory_size = kThreadPerBlock * (sizeof(int64_t) * 2 + sizeof(scalar_t));
            rspmm_backward_mixed_precision_cuda<scalar_t, float, NaryOp<scalar_t>, BinaryOp<scalar_t>>
                <<<dim3(num_row_block, num_dim_block), dim3(dim_per_block, row_per_block), memory_size, stream>>>(
                row_ptr.data_ptr<int64_t>(),
                col_ind.data_ptr<int64_t>(),
                layer_ind.data_ptr<int64_t>(),
                edge_weight.data_ptr<scalar_t>(),
                relation.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                output_grad.data_ptr<scalar_t>(),
                relation_grad.data_ptr<float>(),
                input_grad.data_ptr<float>(),
                num_row, nnz, dim
            );
        });
    }

    // Convert gradients back to original types for mixed precision support
    Tensor weight_grad_orig = weight_grad.to(edge_weight_.scalar_type());
    Tensor relation_grad_orig = relation_grad.to(relation_.scalar_type());
    Tensor input_grad_orig = input_grad.to(input_.scalar_type());

    return std::make_tuple(weight_grad_orig, relation_grad_orig, input_grad_orig);
}

#define DECLARE_FORWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    Tensor rspmm_##ADD##_##MUL##_forward_cuda(                                                            \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,                 \
            const Tensor &relation, const Tensor &input) {                                                \
        return rspmm_forward_cuda<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight, relation, input); \
    }

#define DECLARE_BACKWARD_IMPL(ADD, MUL, NARYOP, BINARYOP) \
    std::tuple<Tensor, Tensor, Tensor> rspmm_##ADD##_##MUL##_backward_cuda(                                 \
            const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,                   \
            const Tensor &relation, const Tensor &input, const Tensor &output, const Tensor &output_grad) { \
        return rspmm_backward_cuda<NARYOP, BINARYOP>(edge_index, edge_type, edge_weight, relation, input,   \
                                                     output, output_grad);                                  \
    }

DECLARE_FORWARD_IMPL(add, mul, NaryAdd, BinaryMul)
DECLARE_BACKWARD_IMPL(add, mul, NaryAdd, BinaryMul)

DECLARE_FORWARD_IMPL(min, mul, NaryMin, BinaryMul)
DECLARE_BACKWARD_IMPL(min, mul, NaryMin, BinaryMul)

DECLARE_FORWARD_IMPL(max, mul, NaryMax, BinaryMul)
DECLARE_BACKWARD_IMPL(max, mul, NaryMax, BinaryMul)

DECLARE_FORWARD_IMPL(add, add, NaryAdd, BinaryAdd)
DECLARE_BACKWARD_IMPL(add, add, NaryAdd, BinaryAdd)

DECLARE_FORWARD_IMPL(min, add, NaryMin, BinaryAdd)
DECLARE_BACKWARD_IMPL(min, add, NaryMin, BinaryAdd)

DECLARE_FORWARD_IMPL(max, add, NaryMax, BinaryAdd)
DECLARE_BACKWARD_IMPL(max, add, NaryMax, BinaryAdd)

} // namespace at
