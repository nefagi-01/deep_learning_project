import torch


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10((((x - y)/255) ** 2).mean((1, 2, 3))).mean()

