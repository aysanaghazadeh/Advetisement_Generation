from torch import nn
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
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
            if args.fine_tuned:
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-instruct",
                                                             token='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',
                                                             device_map='auto')
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-instruct",
                                                          token='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')
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
                self.model = PeftModel.from_pretrained(self.model,
                                                       os.path.join(args.model_path,
                                                                    'my_LLAMA3_CPO/checkpoint-5000/'))
            else:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

                self.pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    token='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',
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )

    def forward(self, prompt):
        if not self.args.fine_tuned:
            messages = [
                {"role": "system", "content": "Be a helpful assistant"},
                {"role": "user", "content": prompt},
            ]
            output = self.pipeline(messages, max_new_tokens=256)
            output = output[0]["generated_text"][-1]['content'].split(':')[-1]
            return output
        else:
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(device=self.args.device)
            inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True).to(device=self.args.device)
            # inputs = self.tokenizer(inputs, return_tensors="pt").to(device='cuda:1')
            generated_ids = self.model.generate(**inputs, max_new_tokens=20)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = output.replace('</s>', '')
            output = output.replace("['", '')
            output = output.replace("']", '')
            output = output.replace('["', '')
            output = output.replace('"]', '')
            output = output
            return output
        # return self.model(**inputs)
