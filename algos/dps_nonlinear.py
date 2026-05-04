# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
from omegaconf import DictConfig

from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.degredations import build_degredation_model
from .ddim import DDIM


class DPSNonlinear(DDIM):
    """DPS with nonlinear Poisson CT likelihood guidance.

    This implements the CT-specific idea from Li et al., "CT Reconstruction
    using Diffusion Posterior Sampling conditioned on a Nonlinear Measurement
    Model": use the raw transmission-count Poisson likelihood and normalize
    the likelihood-gradient step by its squared norm.
    """

    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        self.model = model
        self.diffusion = model.diffusion
        self.H = build_degredation_model(cfg)
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.cond_awd = cfg.algo.cond_awd
        self.eta = cfg.algo.eta
        self.likelihood_weight = float(
            getattr(cfg.algo, "likelihood_weight", getattr(cfg.algo, "grad_term_weight", 310.0))
        )
        self.gradient_eps = float(getattr(cfg.algo, "gradient_eps", 1.0e-12))
        self.init_mode = str(getattr(cfg.algo, "init_mode", "noise"))

    def sample(self, x, y, ts, **kwargs):
        y_0 = kwargs["y_0"]
        metric_callback = kwargs.get("metric_callback")
        n = x.size(0)
        H = self.H

        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        xt_s = [x.detach().cpu()]
        x0_s = []

        xt = x
        max_iter = len(ts)
        for step, (ti, si) in enumerate(zip(reversed(ts), reversed(ss)), start=1):
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            xt = xt.clone().to(x.device).requires_grad_(True)

            if self.cond_awd:
                scale = alpha_s.sqrt() / (
                    alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt()
                )
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            et, x0_pred = self.model(xt, y, t, scale=scale)
            if not self.awd:
                et = (xt - x0_pred * alpha_t.sqrt()) / (1 - alpha_t).sqrt()

            if not hasattr(H, "measurement_loss_per_sample"):
                raise RuntimeError(
                    "DPSNonlinear requires a CT operator with measurement_loss_per_sample()."
                )

            # Negative log likelihood for y ~ Poisson(I0 * exp(-A mu(x0))).
            # Subtracting its gradient is equivalent to ascending log p(y | x0).
            nll_per_sample = H.measurement_loss_per_sample(x0_pred, y_0)
            nll = nll_per_sample.sum()
            grad_nll = torch.autograd.grad(nll, xt, retain_graph=False)[0]
            grad_nll = grad_nll.detach()

            grad_norm_sq = grad_nll.flatten(1).pow(2).sum(dim=1)
            coeff = self.likelihood_weight / grad_norm_sq.clamp_min(self.gradient_eps)
            coeff = coeff.reshape(-1, 1, 1, 1)

            xs = (
                alpha_s.sqrt() * x0_pred.detach()
                + c1 * torch.randn_like(xt)
                + c2 * et.detach()
                - coeff * grad_nll
            )
            xt_s.append(xs.detach().cpu())
            x0_s.append(x0_pred.detach().cpu())
            if metric_callback is not None:
                metric_callback(step=step, sample=xs.detach(), max_iter=max_iter, timestep=int(ti))
            xt = xs.detach()

        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize(self, x, y, ts, **kwargs):
        if self.init_mode == "pinv":
            y_0 = kwargs["y_0"]
            return self.H.H_pinv(y_0).view(*x.size()).detach()
        if self.init_mode != "noise":
            raise ValueError(f"Unsupported DPSNonlinear init_mode: {self.init_mode}")
        return torch.randn_like(x)
