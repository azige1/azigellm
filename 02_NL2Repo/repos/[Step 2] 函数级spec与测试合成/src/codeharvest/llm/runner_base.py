"""大模型 runner 基类：批量请求与缓存。"""

import json
from abc import ABC, abstractmethod

from tqdm import tqdm

from codeharvest.llm.config import LLMConfig
from codeharvest.llm.cache import ResponseCache
from codeharvest.llm.models import ModelSpec
from codeharvest.parallel import run_jobs_parallel

class BaseLLMRunner(ABC):
    def __init__(self, args: LLMConfig, model: ModelSpec):
        self.args = args
        self.model = model
        self.client_kwargs: dict[str, str] = {}

        if self.args.use_cache:
            self.cache = ResponseCache()
        else:
            self.cache = None

    def save_cache(self):
        if self.cache is not None:
            self.cache.save_cache()

    @abstractmethod
    def config(self) -> dict:
        pass

    @abstractmethod
    def _run_single(self, payload) -> list[str]:
        return []

    @staticmethod
    def run_single(combined_args) -> list[str]:
        cache: ResponseCache | None
        call_method: callable  
        payload, cache, args, config, call_method = combined_args

        if cache is not None:
            cache_result = cache.get_from_cache(json.dumps([payload, config]))
            if cache_result is not None:
                return cache_result

        result = call_method(payload)
        return result

    def run_batch(self, payloads: list) -> list[list[str]]:
        outputs = []
        config = self.config()
        arguments = [
            (
                payload,
                self.cache,  
                self.args,  
                config,
                self._run_single,  
            )
            for payload in payloads
        ]
        if self.args.multiprocess > 1:
            parallel_outputs = run_jobs_parallel(
                self.run_single,
                arguments,
                self.args.multiprocess,
                use_progress_bar=True,
            )
            for output in parallel_outputs:
                if output.is_success():
                    outputs.append(output.result)
                else:
                    print("Failed to run the model for some payload")
                    print(output.status)
                    print(output.exception_tb)
                    outputs.extend([""] * self.args.n)
        else:
            outputs = [self.run_single(argument) for argument in tqdm(arguments)]

        if self.cache is not None:
            for payload, output in zip(payloads, outputs):
                self.cache.add_to_cache(
                    json.dumps([payload, config]), output
                )  
            self.save_cache()

        return outputs

    def run_main(self, payloads: list) -> list[list[str]]:
        if self.cache is not None:
            outputs = []
            batch_size = self.args.cache_batch_size
            for i in range(0, len(payloads), batch_size):
                payload_batch = payloads[i : i + batch_size]
                outputs_batch = self.run_batch(payload_batch)
                outputs.extend(outputs_batch)
        else:
            outputs = self.run_batch(payloads)
        return outputs
