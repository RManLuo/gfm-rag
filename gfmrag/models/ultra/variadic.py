# mypy: ignore-errors
import torch


def native_scatter_softmax(src, index, dim=-1, eps=1e-12):
    """
    Implementation of scatter_softmax using PyTorch's native scatter operations.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: -1)
        eps (float, optional): Small value for numerical stability. (default: 1e-12)

    Returns:
        Tensor: Output tensor with softmax applied over elements sharing the same index.
    """
    max_value_per_index = torch.zeros_like(src)
    max_value_per_index = max_value_per_index.scatter_reduce(
        dim=dim,
        index=index,
        src=src,
        reduce="amax",
        include_self=False,
    )
    gathered_max = max_value_per_index.gather(dim, index)
    src_stable = src - gathered_max

    exp_src = torch.exp(src_stable)

    sum_per_index = torch.zeros_like(src)
    sum_per_index = sum_per_index.scatter_add(dim=dim, index=index, src=exp_src)

    gathered_sum = sum_per_index.gather(dim, index) + eps

    return exp_src / gathered_sum


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def native_scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """
    Implements torch_scatter style functions using PyTorch's native  method.

    Args:
        src (torch.Tensor): Source tensor containing values to scatter
        index (torch.Tensor): Index tensor specifying where to scatter values.
                             Must have the same shape as src.
        dim (int, optional): Dimension along which to scatter. Default: 0
        dim_size (int, optional): Size of the output tensor in the given dimension.
                                 If None, will be automatically determined. Default: None

    Returns:
        torch.Tensor: Result of scatter operation
    """
    if index.shape != src.shape:
        index = broadcast(index, src, dim)  # Try to broadcast index to src shape

    assert index.shape == src.shape, "Index and source tensors must have the same shape"

    if dim < 0:
        dim = src.dim() + dim

    if dim_size is None:
        dim_size = index.max().item() + 1

    output_shape = list(src.shape)
    output_shape[dim] = dim_size

    output = torch.zeros(tuple(output_shape), device=src.device, dtype=src.dtype)
    output.scatter_reduce_(
        dim=dim, index=index, src=src, reduce=reduce, include_self=False
    )

    return output


"""
Some variadic functions adopted from TorchDrug
https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/layers/functional/functional.py
"""


def masked_mean(input, mask, dim=None, keepdim=False):
    """
    Masked mean of a tensor.

    Parameters:
        input (Tensor): input tensor
        mask (BoolTensor): mask tensor
        dim (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not
    """
    input = input.masked_scatter(~mask, torch.zeros_like(input))  # safe with nan
    if dim is None:
        return input.sum() / mask.sum().clamp(1)
    return input.sum(dim, keepdim=keepdim) / mask.sum(dim, keepdim=keepdim).clamp(1)


def mean_with_nan(input, dim=None, keepdim=False):
    """
    Mean of a tensor. Ignore all nan values.

    Parameters:
        input (Tensor): input tensor
        dim (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not
    """
    mask = ~torch.isnan(input)
    return masked_mean(input, mask, dim, keepdim)


def multi_slice(starts, ends):
    """
    Compute the union of indexes in multiple slices.

    Example::

        >>> mask = multi_slice(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([0, 1, 2, 4, 5]).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    slices, order = slices.sort()
    values = values[order]
    depth = values.cumsum(0)
    valid = ((values == 1) & (depth == 1)) | ((values == -1) & (depth == 0))
    slices = slices[valid]

    starts, ends = slices.view(-1, 2).t()
    size = ends - starts
    indexes = variadic_arange(size)
    indexes = indexes + starts.repeat_interleave(size)
    return indexes


def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Example::

        >>> mask = multi_slice_mask(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([1, 1, 1, 0, 1, 1])).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    if slices.numel():
        assert slices.min() >= 0 and slices.max() <= length
    mask = native_scatter(values, slices, dim=0, dim_size=length + 1, reduce="sum")[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def _extend(data, size, input, input_size):
    """
    Extend variadic-sized data with variadic-sized input.
    This is a variadic variant of ``torch.cat([data, input], dim=-1)``.

    Example::

        >>> data = torch.tensor([0, 1, 2, 3, 4])
        >>> size = torch.tensor([3, 2])
        >>> input = torch.tensor([-1, -2, -3])
        >>> input_size = torch.tensor([1, 2])
        >>> new_data, new_size = _extend(data, size, input, input_size)
        >>> assert (new_data == torch.tensor([0, 1, 2, -1, 3, 4, -2, -3])).all()
        >>> assert (new_size == torch.tensor([4, 4])).all()

    Parameters:
        data (Tensor): variadic data
        size (LongTensor): size of data
        input (Tensor): variadic input
        input_size (LongTensor): size of input

    Returns:
        (Tensor, LongTensor): output data, output size
    """
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(
        new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device
    )
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def variadic_sum(input, size):
    """
    Compute sum over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = native_scatter(input, index2sample, dim=0, reduce="sum")
    return value


def variadic_mean(input, size):
    """
    Compute mean over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = native_scatter(input, index2sample, dim=0, reduce="mean")
    return value


def variadic_softmax(input, size):
    """
    Compute softmax over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): number of categories of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    log_likelihood = native_scatter_softmax(input, index2sample, dim=0)
    return log_likelihood


def variadic_sort(input, size, descending=False):
    """
    Sort elements in sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        descending (bool, optional): return ascending or descending order

    Returns
        (Tensor, LongTensor): sorted values and indexes
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    abs_max = input[mask].abs().max().item()
    # special case: max = min
    gap = max - min + abs_max * 1e-6
    safe_input = input.clamp(min - gap, max + gap)
    offset = gap * 4
    if descending:
        offset = -offset
    input_ext = safe_input + offset * index2sample
    index = input_ext.argsort(dim=0, descending=descending)
    value = input.gather(0, index)
    index = index - (size.cumsum(0) - size)[index2sample]
    return value, index


def variadic_arange(size):
    """
    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range
