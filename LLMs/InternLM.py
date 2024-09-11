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
            # model_path = "internlm/internlm2_5-7b-chat"
            # self.model = AutoModelForCausalLM.from_pretrained(model_path,
            #                                                   torch_dtype=torch.float16,
            #                                                   trust_remote_code=True,
            #                                                   load_in_8bit=True)
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            #
            # self.model = self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-base-7b", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-base-7b",
                                                              torch_dtype=torch.float16,
                                                              load_in_8bit=True,
                                                              trust_remote_code=True)
            self.model = self.model.eval()

    def forward(self, prompt):
        if not self.args.train:

            # length = 0
            # for response, history in self.model.stream_chat(self.tokenizer, prompt, history=[]):
            #     output = history[0][-1]
            #     length = len(response)
            # return output
            inputs = self.tokenizer([prompt], return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            gen_kwargs = {"temperature": 0.8, "do_sample": True}
            output = self.model.generate(**inputs, **gen_kwargs)
            output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            return output


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


print(output)