from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
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
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_LLAMA3_model'))
        # else:
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     # bnb_4bit_quant_type="nf4",
            #     # bnb_4bit_use_double_quant=True,
            #     bnb_8bit_compute_dtype=torch.bfloat16
            # )
            # self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
            #                                                   device_map='auto',
            #                                                   quantization_config=bnb_config)
            # self.model.gradient_checkpointing_enable()
            # if torch.cuda.device_count() > 1:
            #     self.model.is_parallelizable = True
            #     self.model.model_parallel = True

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(output)
            return output
        # return self.model(**inputs)
