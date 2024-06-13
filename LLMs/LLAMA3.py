from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


class LLAMA3(nn.Module):
    def __init__(self, args):
        super(LLAMA3, self).__init__()
        self.args = args
        if not args.train:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                           token='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')
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                              token='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',
                                                              device_map="auto")
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path,
                                                                                'my_ppo_model'))

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=250)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = output.replace('</s>', '')
            output = output.replace("['", '')
            output = output.replace("']", '')
            output = output.replace('["', '')
            output = output.replace('"]', '')
            output = output.split('Description of the image:')[-1]
            return output
