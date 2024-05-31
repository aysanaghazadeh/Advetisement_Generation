from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class Phi(nn.Module):
    def __init__(self, args):
        super(Phi, self).__init__()
        self.args = args
        if not args.train:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                           token='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',
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                              token='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',
                                                              device_map="auto",
                                                              trust_remote_code=True)
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_phi_model'))

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=5)
            first_line = 'Describe an advertisement image that conveys the following message in 2 sentences:'
            return self.tokenizer.batch_decode(generated_ids)[0].split(first_line)[-1]
        # return self.model(**inputs)
