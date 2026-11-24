import os
from collections import defaultdict
from typing import Any

import ray
import torch
import torch.nn.functional as F

from .schemas import PositionScore, ScoreRequest, ScoreResponse


class ResidentModel:
    """常驻教师 prefill 服务模型：按第 i 行隐状态预测第 i+1 个 token 并分块计算 log-softmax。"""

    def __init__(
        self,
        teacher_id: str,
        item: dict[str, str],
        token_budget: int = 32768,
        row_chunk_size: int = 256,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.teacher_id = teacher_id
        self.version = item.get("version", item["path"])
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.model = AutoModelForCausalLM.from_pretrained(
            item["path"], dtype=torch.bfloat16, device_map={"": self.device.index}
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(item["path"])
        self.token_budget = token_budget
        self.row_chunk_size = row_chunk_size

    @torch.inference_mode()
    def score(self, request: ScoreRequest) -> ScoreResponse:
        if any(position <= 0 or position >= len(request.token_ids) for position in request.positions):
            raise ValueError("评分位置必须有前驱且对应学生响应 token")
        if request.candidate_token_ids is not None and len(request.candidate_token_ids) != len(request.positions):
            raise ValueError("candidate_token_ids 长度必须与 positions 一致")

        ids = torch.tensor([request.token_ids], dtype=torch.long, device=self.device)
        hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        # 第 i 行隐状态预测第 i+1 个 token
        positions = torch.tensor([position - 1 for position in request.positions], device=self.device)
        sampled = torch.tensor([request.token_ids[position] for position in request.positions], device=self.device)
        rows = hidden[positions]
        output = self.model.get_output_embeddings()

        logprobs: list[float] = []
        topk_ids_all: list[list[int]] = []
        topk_logprobs_all: list[list[float]] = []
        candidate_maps_all: list[dict[int, float]] = []

        for start in range(0, len(rows), self.row_chunk_size):
            logits = F.linear(rows[start : start + self.row_chunk_size].float(), output.weight.float())
            values = F.log_softmax(logits, dim=-1)
            selected = values.gather(-1, sampled[start : start + self.row_chunk_size, None]).squeeze(-1)
            logprobs.extend(selected.tolist())
            if request.top_k > 0:
                k = min(int(request.top_k), values.shape[-1])
                top_values, top_indices = torch.topk(values, k=k, dim=-1)
                topk_ids_all.extend(top_indices.tolist())
                topk_logprobs_all.extend(top_values.tolist())
            else:
                topk_ids_all.extend([[] for _ in range(values.shape[0])])
                topk_logprobs_all.extend([[] for _ in range(values.shape[0])])

            if request.candidate_token_ids is not None:
                for offset in range(values.shape[0]):
                    cand_ids = [int(token_id) for token_id in request.candidate_token_ids[start + offset]]
                    if cand_ids:
                        cand_tensor = torch.tensor(cand_ids, dtype=torch.long, device=values.device)
                        cand_values = values[offset, cand_tensor].tolist()
                        candidate_maps_all.append(dict(zip(cand_ids, cand_values, strict=True)))
                    else:
                        candidate_maps_all.append({})
            else:
                candidate_maps_all.extend([{} for _ in range(values.shape[0])])

        scores = [
            PositionScore(
                position=position,
                sampled_id=request.token_ids[position],
                sampled_logprob=value,
                top_k_ids=top_ids,
                top_k_logprobs=top_lps,
                candidate_logprobs=cand_map,
            )
            for position, value, top_ids, top_lps, cand_map in zip(
                request.positions, logprobs, topk_ids_all, topk_logprobs_all, candidate_maps_all, strict=True
            )
        ]
        return ScoreResponse(
            request_id=request.request_id,
            teacher_id=self.teacher_id,
            scores=scores,
            model_version=self.version,
            prefill_tokens=len(request.token_ids),
        )


@ray.remote
class ResidentRank:
    def __init__(self, manifest: dict[str, dict[str, str]], token_budget: int, row_chunk_size: int) -> None:
        if len(manifest) != 2:
            raise ValueError("每张教师卡必须恰好常驻两个 Qwen3-8B 教师")
        self.models = {
            name: ResidentModel(name, item, token_budget, row_chunk_size)
            for name, item in manifest.items()
        }

    def ready(self) -> list[str]:
        return sorted(self.models)

    def versions(self) -> dict[str, str]:
        return {name: model.version for name, model in self.models.items()}

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed = [ScoreRequest.model_validate(value) for value in requests]
        unknown = {request.teacher_id for request in parsed} - set(self.models)
        if unknown:
            raise ValueError(f"本卡未部署教师: {sorted(unknown)}")
        results = {}
        for request in parsed:
            results[request.request_id] = self.models[request.teacher_id].score(request).model_dump()
        return [results[request.request_id] for request in parsed]


class ResidentCoordinator:
    def __init__(self, manifest: dict[str, dict[str, Any]]) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        groups = defaultdict(dict)
        for teacher_id, item in manifest.items():
            groups[int(item.get("gpu", len(groups)))][teacher_id] = item
        if any(len(group) != 2 for group in groups.values()):
            raise ValueError("教师 manifest 必须按每卡两个教师分组")
        token_budget = int(os.environ.get("TEACHER_TOKEN_BUDGET", "32768"))
        row_chunk = int(os.environ.get("TEACHER_ROW_CHUNK_SIZE", "256"))
        self.ranks = [
            ResidentRank.options(num_gpus=1).remote(group, token_budget, row_chunk)
            for _, group in sorted(groups.items())
        ]
        self._teacher_ranks = {}
        for rank, (_, group) in zip(self.ranks, sorted(groups.items()), strict=True):
            for teacher_id in group:
                self._teacher_ranks[teacher_id] = rank
        ray.get([rank.ready.remote() for rank in self.ranks])

    def layouts(self) -> list[list[str]]:
        return ray.get([rank.ready.remote() for rank in self.ranks])

    def score_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped = defaultdict(list)
        for request in requests:
            grouped[self._teacher_ranks[request["teacher_id"]]].append(request)
        outputs = ray.get([rank.score_batch.remote(group) for rank, group in grouped.items()])
        by_id = {response["request_id"]: response for values in outputs for response in values}
        return [by_id[request["request_id"]] for request in requests]
