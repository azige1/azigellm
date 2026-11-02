"""按模型分派批量补全请求的入口。"""

from codeharvest.llm.config import LLMConfig
from codeharvest.llm.models import ModelBackend, MODEL_REGISTRY

class CompletionEngine:

    @staticmethod
    def run_chat_completions(args: LLMConfig, payloads: list) -> list[list[str]]:
        model_name = args.model_name
        matched_lang_model = [
            model for model in MODEL_REGISTRY if model.model_name == model_name
        ]
        assert len(matched_lang_model) == 1
        model = matched_lang_model[0]

        if model.style == ModelBackend.OpenAI:
            from codeharvest.llm.openai import OpenAIChatRunner

            runner = OpenAIChatRunner(args, model)
            return runner.run_main(payloads)

        raise ValueError(f"Unsupported model style: {model.style}")

    
    
    
    
    
    
    
    

    

    

    

    

    
    
    
    
    
