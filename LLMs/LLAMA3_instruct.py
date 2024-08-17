from torch import nn
from transformers import BitsAndBytesConfig, pipeline
import torch
from peft import PeftModel
import os


class LLAMA3Instruct(nn.Module):
    def __init__(self, args):
        super(LLAMA3Instruct, self).__init__()
        self.args = args
        if not args.train:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

            self.pipeline = pipeline(
                "text-generation",
                model=model_id,
                token='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',
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

    def forward(self, prompt):
        if not self.args.train:
            messages = [
                {"role": "system", "content": "Be a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            output = self.pipeline(messages, max_new_tokens=256)[0]["generated_text"][-1]
            output = output.replace('</s>', '')
            output = output.replace("['", '')
            output = output.replace("']", '')
            output = output.replace('["', '')
            output = output.replace('"]', '')
            output = output.split(':')[-1]
            return output
        # return self.model(**inputs)
