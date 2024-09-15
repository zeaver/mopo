__all__ = ["AzureGPTConfig", "AzureGPT"]

import os
import yaml
from typing import Dict

from dataclasses import dataclass, field

import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion


@dataclass
class AzureGPTConfig:
    azure_endpoint: str = field(
        default="https://example.azurewebsites.net",
        metadata={"help": "Azure endpoint URL"}
    )
    api_key: str = field(
        default="your-api-key",
        metadata={"help": "API key for authentication"}
    )
    api_version: str = field(
        default="v1",
        metadata={"help": "API version"}
    )
    model_name: str = field(
        default="your-model-name",
        metadata={"help": "Name of the GPT model"}
    )

class AzureGPT:
    def __init__(self, config: AzureGPTConfig):
        assert int(openai.__version__.split(".")[0]) >= 1,\
            f"requires OpenAI Python library version 1.0.0 or higher, but install {openai.__version__}"
        self.config = config
        self.client = self.load_client(config)

    # def __call__(self) -> AzureOpenAI:
    #     return self.client
    
    @property
    def init_messages(self):
        return {"role":"system","content":"You are an AI assistant that helps people find information."}

    def load_client(self, config) -> AzureOpenAI:
        # assert "azure_endpoint" in config 
        # assert "api_key" in config 
        # assert "api_version" in config 

        self.deployment_model_name = config.model_name

        return AzureOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )
    
    def custom_completion(self, 
                          query, 
                          init_message=None,
                          **kwargs) -> ChatCompletion:
        # response data structure:
        # https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        # demo:
        # ```
        # {
        #   "choices": [
        #     {
        #       "finish_reason": "stop",
        #       "index": 0,
        #       "message": {
        #         "content": "The 2021 World Series was played in Texas at Globe Life Field in Arlington.",
        #         "role": "assistant"
        #       },
        #       "logprobs": null
        #     }
        #   ],
        #   "created": 1677664795,
        #   "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        #   "model": "gpt-3.5-turbo-0613",
        #   "object": "chat.completion",
        #   "usage": {
        #     "completion_tokens": 17,
        #     "prompt_tokens": 57,
        #     "total_tokens": 74
        #   }
        # }
        # ```

        if init_message is None:
            init_message = self.init_messages
        message_text = [
            init_message,
            {"role":"user","content":query}
        ]
        
        try:
            response =  self.client.chat.completions.create(
                model = self.deployment_model_name,
                messages = message_text,
                temperature=0.1,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            return True, response
        except Exception as e:
            return False, str(e)

if __name__ == "__main__":
    with open(r"x_retrieval/pre_processing/data_synthesis/config.yaml", 'r') as f:
        azure_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    gpt_config = AzureGPTConfig(**azure_config["azure_gpt"])
    gpt = AzureGPT(gpt_config)
    res = gpt.custom_completion("帮我使用python写一份快速排序代码")
    print(res)

    
