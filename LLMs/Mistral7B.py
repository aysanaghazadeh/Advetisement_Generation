from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


class Mistral7B(nn.Module):
    def __init__(self, args):
        super(Mistral7B, self).__init__()
        self.args = args
        if not args.train:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                              device_map="auto")
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                                              device_map='auto',
                                                              quantization_config=nf4_config)

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(generated_ids)[0]
        return self.model(**inputs)
