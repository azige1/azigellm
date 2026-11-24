from typing import Any, Mapping, Sequence

import torch


def compute_token_weighted_std(values: torch.Tensor, mask: torch.Tensor) -> float:
    """按有效 token 数加权计算张量的标准差。"""
    valid = mask.bool()
    if not valid.any():
        return 0.0
    selected = values[valid].float()
    return float(selected.std(unbiased=False).item())


class DomainAdvantageNormalizer:
    """4.5 按领域维护 token 数加权的 advantage 滑动标准差并归一化。"""

    def __init__(
        self,
        decay: float = 0.9,
        sigma_floor: float = 0.2,
        raw_clip: float = 5.0,
        post_norm_clip: float = 3.0,
        initial_stds: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("滑动衰减系数必须位于 [0, 1)")
        if sigma_floor <= 0.0:
            raise ValueError("标准差下限必须为正")
        if raw_clip <= 0.0 or post_norm_clip <= 0.0:
            raise ValueError("裁剪阈值必须为正")
        self.decay = float(decay)
        self.sigma_floor = float(sigma_floor)
        self.raw_clip = float(raw_clip)
        self.post_norm_clip = float(post_norm_clip)
        self.running_stds: dict[str, float] = {
            str(k): float(v) for k, v in (initial_stds or {}).items()
        }

    def update_domain_std(self, domain: str, step_std: float) -> float:
        key = str(domain)
        val = float(step_std)
        if key not in self.running_stds:
            self.running_stds[key] = val
        else:
            self.running_stds[key] = self.decay * self.running_stds[key] + (1.0 - self.decay) * val
        return self.running_stds[key]

    def effective_sigma(self, domain: str) -> float:
        return max(self.running_stds.get(str(domain), self.sigma_floor), self.sigma_floor)

    def normalize_batch(
        self,
        raw_advantages: torch.Tensor,
        response_mask: torch.Tensor,
        domains: Sequence[str],
        clip_after_norm: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if raw_advantages.shape != response_mask.shape:
            raise ValueError("raw_advantages 与 response_mask 必须等形")
        if len(domains) != raw_advantages.shape[0]:
            raise ValueError("domains 长度必须与 batch 大小一致")

        clipped_raw = torch.clamp(raw_advantages.float(), min=-self.raw_clip, max=self.raw_clip)
        normalized = torch.zeros_like(clipped_raw, dtype=torch.float32)
        mask_bool = response_mask.bool()

        rows_by_domain: dict[str, list[int]] = {}
        for idx, domain in enumerate(domains):
            rows_by_domain.setdefault(str(domain), []).append(idx)

        step_stds: dict[str, float] = {}
        running_stds: dict[str, float] = {}
        effective_sigmas: dict[str, float] = {}

        for domain, indices in rows_by_domain.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=clipped_raw.device)
            domain_vals = clipped_raw.index_select(0, idx_tensor)
            domain_mask = mask_bool.index_select(0, idx_tensor)
            if domain_mask.any():
                s_d = compute_token_weighted_std(domain_vals, domain_mask)
                running = self.update_domain_std(domain, s_d)
                sigma = max(running, self.sigma_floor)
                step_stds[domain] = s_d
                running_stds[domain] = running
                effective_sigmas[domain] = sigma

                scaled = torch.where(domain_mask, domain_vals / sigma, torch.zeros_like(domain_vals))
                if clip_after_norm:
                    scaled = torch.clamp(scaled, min=-self.post_norm_clip, max=self.post_norm_clip)
                normalized.index_copy_(0, idx_tensor, scaled)

        return normalized, {
            "step_stds": step_stds,
            "running_stds": running_stds,
            "effective_sigmas": effective_sigmas,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "sigma_floor": self.sigma_floor,
            "raw_clip": self.raw_clip,
            "post_norm_clip": self.post_norm_clip,
            "running_stds": dict(self.running_stds),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.sigma_floor = float(state.get("sigma_floor", self.sigma_floor))
        self.raw_clip = float(state.get("raw_clip", self.raw_clip))
        self.post_norm_clip = float(state.get("post_norm_clip", self.post_norm_clip))
        self.running_stds = {
            str(k): float(v) for k, v in dict(state.get("running_stds", {})).items()
        }
