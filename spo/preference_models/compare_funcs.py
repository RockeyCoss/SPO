import torch
from .builder import COMPARE_FUNCS

@COMPARE_FUNCS.register_module()
def preference_score_compare(scores, threshold):
    # scores: num_sample_per_step, b
    scores, indices = torch.sort(scores, dim=0, descending=True)
    # 2, b
    indices = indices[[0, -1], :]
    scores = scores[[0, -1], :]
    scores = scores.softmax(dim=0)
    # b
    valid_samples = scores[0] - scores[1] > threshold
    return indices, valid_samples
