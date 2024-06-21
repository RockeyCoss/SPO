import torch
import torch.distributed as dist

def gather_tensor_with_diff_shape(input_tensor, primary_dim_size_list):
    gathered_tensor_list = [
        input_tensor.new_zeros(
            primary_dim_size, *input_tensor.shape[1:],
        )
        for primary_dim_size in primary_dim_size_list
    ]
    dist.all_gather(gathered_tensor_list, input_tensor)
    gathered_tensor = torch.cat(gathered_tensor_list, dim=0)
    return gathered_tensor
