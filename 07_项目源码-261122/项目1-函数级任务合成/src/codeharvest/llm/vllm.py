"""本地 vLLM 推理的 runner。"""

from vllm import LLM, SamplingParams  

from codeharvest.llm.config import LLMConfig
from codeharvest.llm.runner_base import BaseLLMRunner
from codeharvest.llm.models import ModelSpec

class LocalVLLMRunner(BaseLLMRunner):
    def __init__(self, args: LLMConfig, model: ModelSpec):
        super().__init__(args, model)
        model_tokenizer_path = (
            model.model_name if args.local_model_path is None else args.local_model_path
        )
        self.llm = LLM(
            model=model_tokenizer_path,
            tokenizer=model_tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            max_model_len=args.vllm_max_model_len,
            enable_prefix_caching=args.enable_prefix_caching,
            disable_custom_all_reduce=False,
        )

        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=self.args.stop,
        )

    def _run_single(self, payload):  
        pass

    def run_batch(self, payloads: list[str]) -> list[list[str]]:
        outputs: list[list[str]] = [None for _ in payloads]  
        remaining_payloads = []
        remaining_indices = []
        for payload_index, payload in enumerate(payloads):
            if self.cache is not None:
                cache_result = self.cache.get_from_cache(payload)
                if cache_result is not None:
                    outputs[payload_index] = cache_result
                    continue
            remaining_payloads.append(payload)
            remaining_indices.append(payload_index)
        if remaining_payloads:
            vllm_outputs = self.llm.generate(remaining_payloads, self.sampling_params)
            if self.cache is not None:
                assert len(remaining_payloads) == len(vllm_outputs)
                for index, remaining_payload, vllm_output in zip(
                    remaining_indices, remaining_payloads, vllm_outputs
                ):
                    output_texts = [o.text for o in vllm_output.outputs]
                    self.cache.add_to_cache(remaining_payload, output_texts)
                    outputs[index] = output_texts
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs
