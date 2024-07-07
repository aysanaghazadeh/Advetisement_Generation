from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class Vicuna(nn.Module):
    def __init__(self, args):
        super(Vicuna, self).__init__()
        self.args = args
        if not args.train:
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5",
                                                           token='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')
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5",
                                                              token='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',
                                                              device_map="auto")
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_ppo_model_DMD_batch_size_1'))

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = output.replace('</s>', '')
            output = output.replace("['", '')
            output = output.replace("']", '')
            output = output.replace('["', '')
            output = output.replace('"]', '')
            output = output.split(':')[-1]
            return output
        # return self.model(**inputs)
