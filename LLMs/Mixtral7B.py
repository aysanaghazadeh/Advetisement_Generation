from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mixtral7B(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
                                                         device_map="auto")
        # self.model = self.model.to(device=self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    def forward(self, prompt):
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]


