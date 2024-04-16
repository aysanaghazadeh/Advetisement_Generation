from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Mistral7B(nn.Module):
    def __init__(self,):
        super(Mistral7B, self).__init__()
        # self.args = args
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def forward(self, inputs):
        # if not self.args.train:
        inputs = self.tokenizer(inputs, return_tensors="pt")
        # generated_ids = self.model.generate(**inputs, max_new_tokens=300)
        # return self.tokenizer.batch_decode(generated_ids)[0]
        return self.model(**inputs)

model = Mistral7B()
prompt = 'hello mistral'
output = model(prompt)
print(type(output))
# print(output)