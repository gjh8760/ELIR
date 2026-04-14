"""ElirRetinex: Phase-2 model.

Retinex decomposition (frozen) + I-branch (pixel-space LCFM) + R-branch (latent LCFM).
Mirrors Elir's API: forward / inference / collapse / load_weights.
"""
import torch
import torch.nn as nn

from ELIR.models.elir import pos_emb
from ELIR.models.load_model import get_model


class ElirRetinex(nn.Module):
    def __init__(
        self,
        R_fm_cfg,  R_fmir_cfg, R_mmse_cfg, R_enc_cfg, R_dec_cfg,
        I_fm_cfg,  I_fmir_cfg, I_mmse_cfg,
        decomposer_cfg,
    ):
        super().__init__()

        # --- Decomposer (frozen) ---
        self.decomposer = get_model(decomposer_cfg)
        for p in self.decomposer.parameters():
            p.requires_grad_(False)
        self.decomposer.eval()

        # --- R-branch (latent space, mirrors Phase-1 Elir) ---
        self.R_fmir = get_model(R_fmir_cfg)
        self.R_mmse = get_model(R_mmse_cfg)
        self.R_enc  = get_model(R_enc_cfg)
        self.R_dec  = get_model(R_dec_cfg)
        self.R_K          = R_fm_cfg.get("k_steps")
        self.R_dt         = 1.0 / self.R_K
        self.R_sigma_s    = R_fm_cfg.get("sigma_s", 0.025)
        self.R_t_emb_dim  = R_fmir_cfg.get("params", {}).get("t_emb_dim", 160)
        self.R_latent_shape = R_fm_cfg.get("latent_shape", [16, 32, 32])
        self.R_dynamic_noise = R_fm_cfg.get("dynamic_noise", True)

        # --- I-branch (pixel space, 1 channel) ---
        self.I_fmir = get_model(I_fmir_cfg)
        self.I_mmse = get_model(I_mmse_cfg)
        self.I_K          = I_fm_cfg.get("k_steps")
        self.I_dt         = 1.0 / self.I_K
        self.I_sigma_s    = I_fm_cfg.get("sigma_s", 0.01)
        self.I_t_emb_dim  = I_fmir_cfg.get("params", {}).get("t_emb_dim", 160)
        self.I_dynamic_noise = I_fm_cfg.get("dynamic_noise", True)

    # Decomposer must never leave eval mode.
    def train(self, mode: bool = True):
        super().train(mode)
        self.decomposer.eval()
        for p in self.decomposer.parameters():
            p.requires_grad_(False)
        return self

    def collapse(self):
        for sub in (self.R_fmir, self.R_mmse, self.I_fmir, self.I_mmse):
            if hasattr(sub, "collapse"):
                sub.collapse()

    def load_weights(self, path):
        if not path:
            return
        state_dict = torch.load(path, map_location="cpu")
        if path.endswith(".ckpt"):
            key_map = {
                "state_dict_R_fmir": self.R_fmir,
                "state_dict_R_mmse": self.R_mmse,
                "state_dict_R_enc":  self.R_enc,
                "state_dict_R_dec":  self.R_dec,
                "state_dict_I_fmir": self.I_fmir,
                "state_dict_I_mmse": self.I_mmse,
            }
            for key, sub in key_map.items():
                if key in state_dict:
                    sub.load_state_dict(state_dict[key])
            self.collapse()
        else:
            # `elir.pth`-style collapsed full state_dict (ElirRetinex.state_dict()).
            self.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def decompose(self, x):
        return self.decomposer(x)

    def _euler_solve_I(self, I_low, device):
        if self.I_dynamic_noise:
            noise = self.I_sigma_s * torch.randn_like(I_low)
        else:
            noise = self.I_sigma_s * torch.randn_like(I_low)
        i0 = self.I_mmse(I_low) + noise
        t = 0.0
        for _ in range(self.I_K):
            i0 = i0 + self.I_dt * self.I_fmir(i0, pos_emb(t, self.I_t_emb_dim).to(device))
            t += self.I_dt
        return i0

    def _euler_solve_R(self, R_low, device):
        z = self.R_enc(R_low)
        if self.R_dynamic_noise:
            noise = self.R_sigma_s * torch.randn_like(z)
        else:
            noise = self.R_sigma_s * torch.randn_like(z)
        z0 = self.R_mmse(z) + noise
        t = 0.0
        for _ in range(self.R_K):
            z0 = z0 + self.R_dt * self.R_fmir(z0, pos_emb(t, self.R_t_emb_dim).to(device))
            t += self.R_dt
        return z0

    def forward(self, x):
        self.to(x.device)
        R_low, I_low = self.decompose(x)

        i_hat = self._euler_solve_I(I_low, x.device)
        z_R   = self._euler_solve_R(R_low, x.device)
        R_hat = self.R_dec(z_R)

        I_hat = i_hat.clamp(min=1e-4, max=1.0)
        R_hat = R_hat.clamp(0.0, 1.0)
        y = (R_hat * I_hat).clamp(0.0, 1.0)
        return y

    def inference(self, x):
        return self.forward(x)
