from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Mistral7B(nn.Module):
    def __init__(self, args):
        super(Mistral7B, self).__init__()
        self.args = args
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
        if not args.train:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=300)
            return self.tokenizer.batch_decode(generated_ids)[0]
        return self.model(**inputs)


class Mistral7BPipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pipe = pipeline("text-generation",
                             model="mistralai/Mistral-7B-v0.1",
                             device_map='auto',
                             max_length=300)

    def forward(self, prompt):
        output = self.pipe(prompt)[0]['generated_text'].split('Description:')[-1]
        return output
