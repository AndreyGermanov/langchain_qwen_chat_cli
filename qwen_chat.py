from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema import ChatResult, ChatGeneration

from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

# Tiny wrapper class to make LangChain Chat instance from Qwen model
class LocalQwenChat(BaseChatModel):

    # Using Qwen-7B model
    model_name: str = "Qwen/Qwen-7B-Chat"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer: Any = None  
    model: Any = None      

    # Create the model instance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map={"": self.device},
            bf16=True
        ).eval()

    # Respond to queries
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n"
        prompt += "Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs_on_device = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        outputs = self.model.generate(**inputs_on_device, max_new_tokens=2048)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]

        generation = ChatGeneration(message=AIMessage(content=response))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "local_qwen_chat"