import torch
import torch.nn.functional as F
import numpy as np


def gaussian_nll_loss(mu, log_var, target, marvel_unc, mask, eps=1e-8):
    """
    Heteroscedastic loss with a safety epsilon to prevent NaN.
    """
    mu = mu[mask]
    log_var = log_var[mask]
    target = target[mask]

    # Clip experimental uncertainty to a minimum of eps
    sigma_exp_sq = torch.clamp(marvel_unc[mask], min=eps) ** 2

    # Predicted variance
    sigma_pred_sq = torch.exp(log_var)

    # Total variance with safety buffer
    total_var = sigma_pred_sq + sigma_exp_sq + eps

    # Negative Log Likelihood
    # Using 0.5 * log is numerically more stable than log(sqrt)
    loss = 0.5 * (torch.log(total_var) + (target - mu) ** 2 / total_var)

    return loss.mean()


def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    mu, log_var = model(data.x, data.edge_index)
    loss = gaussian_nll_loss(mu, log_var, data.y, data.unc, data.train_mask)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def mc_dropout_predict(model, data, n_passes=50):
    """
    Performs multiple forward passes to estimate Bayesian uncertainty.
    """
    model.eval()
    mu_list = []
    log_var_list = []

    for _ in range(n_passes):
        # We force dropout to remain active
        mu, log_var = model(data.x, data.edge_index, force_dropout=True)
        mu_list.append(mu)
        log_var_list.append(torch.exp(log_var))

    # Stack and calculate statistics
    mu_stack = torch.stack(mu_list)  # [n_passes, n_nodes, 1]

    # Final Mean = Average of means
    final_mu = mu_stack.mean(dim=0)

    # Final Uncertainty = Var(means) [Epistemic] + Mean(vars) [Aleatoric]
    epistemic_unc = mu_stack.var(dim=0)
    aleatoric_unc = torch.stack(log_var_list).mean(dim=0)
    total_unc = torch.sqrt(epistemic_unc + aleatoric_unc)

    return final_mu, total_unc


def get_real_mae(model, data, mask_name="train_mask"):
    model.eval()
    mask = getattr(data, mask_name)

    with torch.no_grad():
        mu_norm, _ = model(data.x, data.edge_index)

        # Ensure std and mean are on the same device as the prediction
        y_std = data.y_std.to(mu_norm.device)
        y_mean = data.y_mean.to(mu_norm.device)

        # Unscale
        pred_raw = (mu_norm[mask] * y_std) + y_mean
        target_raw = (data.y[mask] * y_std) + y_mean

        mae = torch.abs(pred_raw - target_raw).mean()

    return mae.item(), pred_raw, target_raw
