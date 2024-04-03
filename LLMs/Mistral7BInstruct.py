from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mistral7BInstruct(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                         device_map="auto")
        self.model = self.model.to(device=self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    def forward(self, prompt):
        messages = {'role':"user", 'content':prompt}
        model_inputs = self.tokenizer([messages], return_tensors="pt").to(device=self.args.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]


