import random
from collections import defaultdict
from typing import Any

from . import DEFAULT_DOMAIN_SHARES


def distillation_shares() -> dict[str, float]:
    """返回默认的八教师加 self_anchor（8%）静态采样份额。"""
    return dict(DEFAULT_DOMAIN_SHARES)


class CrossStepQuota:
    def __init__(self, shares: dict[str, float]) -> None:
        if not shares or any(value < 0 for value in shares.values()) or sum(shares.values()) <= 0:
            raise ValueError("固定份额无效")
        normalizer = sum(shares.values())
        self.shares = {name: value / normalizer for name, value in shares.items()}
        self.credits = {name: 0.0 for name in shares}

    def counts(self, batch_size: int) -> dict[str, int]:
        if batch_size < 0:
            raise ValueError("batch 大小不能为负")
        for name, share in self.shares.items():
            self.credits[name] += batch_size * share
        result = {name: int(self.credits[name]) for name in self.shares}
        remainder = batch_size - sum(result.values())
        order = sorted(self.shares, key=lambda name: (self.credits[name] - result[name], name), reverse=True)
        for name in order[:remainder]:
            result[name] += 1
        for name in self.shares:
            self.credits[name] -= result[name]
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"shares": dict(self.shares), "credits": dict(self.credits)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["shares"] != self.shares:
            raise ValueError("checkpoint 的固定份额与当前配置不一致")
        self.credits = {name: float(state["credits"][name]) for name in self.shares}


class FixedDomainSampler:
    def __init__(self, domain_indices: dict[str, list[int]], shares: dict[str, float], seed: int = 17) -> None:
        if set(domain_indices) != set(shares):
            raise ValueError("领域索引池与固定份额必须一一对应")
        if any(not values for values in domain_indices.values()):
            raise ValueError("每个有份额的领域都必须有训练样本")
        self.domain_indices = {name: list(values) for name, values in domain_indices.items()}
        self.quota = CrossStepQuota(shares)
        self.rng = random.Random(seed)
        self.cursors = defaultdict(int)

    def sample_indices(self, batch_size: int) -> list[int]:
        result = []
        for name, count in self.quota.counts(batch_size).items():
            pool = self.domain_indices[name]
            for _ in range(count):
                cursor = self.cursors[name]
                if cursor >= len(pool):
                    self.rng.shuffle(pool)
                    cursor = 0
                result.append(pool[cursor])
                self.cursors[name] = cursor + 1
        self.rng.shuffle(result)
        return result

    def update_shares(self, shares: dict[str, float]) -> None:
        if set(shares) != set(self.domain_indices):
            raise ValueError("新配比改变了可用领域集合")
        credits = dict(self.quota.credits)
        self.quota = CrossStepQuota(shares)
        self.quota.credits = {name: credits[name] for name in shares}

    def state_dict(self) -> dict[str, Any]:
        return {
            "quota": self.quota.state_dict(),
            "rng": self.rng.getstate(),
            "cursors": dict(self.cursors),
            "domain_indices": self.domain_indices,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["domain_indices"] != self.domain_indices:
            raise ValueError("checkpoint 的领域索引池与当前数据不一致")
        saved_quota = state["quota"]
        if set(saved_quota["shares"]) != set(self.domain_indices):
            raise ValueError("checkpoint 的配比领域与当前数据不一致")
        self.quota = CrossStepQuota(saved_quota["shares"])
        self.quota.credits = {name: float(saved_quota["credits"][name]) for name in self.domain_indices}
        self.rng.setstate(state["rng"])
        self.cursors = defaultdict(int, state["cursors"])
