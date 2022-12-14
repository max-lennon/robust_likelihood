import torch
from torch.distributions import Normal
from scipy.stats import chi2
import numpy as np

PI = torch.tensor(np.pi)


def ring_prob(y, batch_size, mode="ring_95"):

    d = y.shape[1]
    device = y.device

    y_square = torch.square(y)

    r = torch.sqrt(torch.sum(y_square, axis=1))    
    
    if mode == "ring_50":
        gauss_r_25 = np.sqrt(chi2.isf(0.75, d))
        gauss_r_75 = np.sqrt(chi2.isf(0.25, d))
        mu_r = (gauss_r_25 + gauss_r_75) / 2
        sd = (gauss_r_75 - gauss_r_25) / 4

    elif mode == "ring_95":
        gauss_r_95 = np.sqrt(chi2.isf(0.05, d))
        mu_r = gauss_r_95 + 2
        sd = 1

    norm_dist = Normal(mu_r, sd)
    norm_prob = norm_dist.log_prob(r).to(device)

    logpx = norm_prob - (np.log(2) + torch.log(PI.to(device)) * (d-1))

    y_sum_square_num = torch.sum(torch.triu(torch.unsqueeze(y_square, 1).expand(batch_size, d-2, d), diagonal=1), axis=2)
    y_sum_square_denom = torch.sum(torch.triu(torch.unsqueeze(y_square, 1).expand(batch_size, d-2, d), diagonal=0), axis=2)

    log_sin_arctan_theta = 0.5 * (torch.log(y_sum_square_num) - torch.log(y_sum_square_denom))

    power = torch.unsqueeze(torch.Tensor(range(d-2, 0, -1)), 0).expand(batch_size, d-2).to(device)
    
    log_pow_sin_arctan_theta = torch.mul(power, log_sin_arctan_theta)

    polar_det_inv = (d-1) * torch.log(r) + torch.sum(log_pow_sin_arctan_theta, axis=1)
    polar_det = -1 * polar_det_inv

    return logpx, polar_det


def offset_gauss_prob(y, batch_size, direction, offset):
    # -log(zero-mean gaussian) + log determinant
    # -log p_x = log(pz(f(x))) + log(det(\partial f/\partial x))
    # -log p_x = 0.5 * y**2 + s1 + s2 + ... + batch_norm_scalers + l2_regularizers(scale)
    logpx = -torch.sum(0.5 * torch.log(2 * PI) + 0.5 * (y-direction*offset)**2)
    det = torch.ones(batch_size)
    return logpx, det


def loss_fn_ood(like_function, y, s, norms, scale, batch_size):
    # -log(zero-mean gaussian) + log determinant
    # -log p_x = log(pz(f(x))) + log(det(\partial f/\partial x))
    # -log p_x = 0.5 * y**2 + s1 + s2 + ... + batch_norm_scalers + l2_regularizers(scale)
    det = torch.sum(s)

    s_batch = s.view((batch_size, -1))
    det_batch = torch.sum(s_batch, axis=1)

    logpx, polar_det = like_function(y, batch_size)

    logpx = torch.sum(logpx)
    polar_det = torch.sum(polar_det)

    norms = torch.sum(norms)
    reg = 5e-5 * torch.sum(scale ** 2)
    loss = -(logpx + polar_det + det + norms) + reg
    return torch.div(loss, batch_size), (-logpx, -polar_det, -det, -norms, reg)

def loss_fn(y, s, norms, scale, batch_size):
    # -log(zero-mean gaussian) + log determinant
    # -log p_x = log(pz(f(x))) + log(det(\partial f/\partial x))
    # -log p_x = 0.5 * y**2 + s1 + s2 + ... + batch_norm_scalers + l2_regularizers(scale)
    logpx = -torch.sum(0.5 * torch.log(2 * PI) + 0.5 * y**2)
    det = torch.sum(s)
    norms = torch.sum(norms)
    reg = 5e-5 * torch.sum(scale ** 2)
    loss = -(logpx + det + norms) + reg
    return torch.div(loss, batch_size), (-logpx, -det, -norms, reg)