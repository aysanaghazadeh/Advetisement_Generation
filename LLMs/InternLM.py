from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class InternLM(nn.Module):
    def __init__(self, args):
        super(InternLM, self).__init__()
        self.args = args
        if not args.train:
            model_path = "internlm/internlm2_5-7b-chat"
            self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                              torch_dtype=torch.float16,
                                                              trust_remote_code=True,
                                                              load_in_8bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            self.model = self.model.eval()

    def forward(self, prompt):
        if not self.args.train:

            length = 0
            for response, history in self.model.stream_chat(self.tokenizer, prompt, history=[], temperature=0):
                output = history[0][-1]
                length = len(response)
            return output
