from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Mistral7B(nn.Module):
    def __init__(self,):
        super(Mistral7B, self).__init__()
        # self.args = args
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    def forward(self, inputs):
        # if not self.args.train:
        inputs = self.tokenizer(inputs, return_tensors="pt").to(device='cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=300)
        return self.tokenizer.batch_decode(generated_ids)[0]
        # return self.model(**inputs)

model = Mistral7B()
prompt = """
Question: Describe an advertisement image that conveys the following message:
I should drink absolute vodka because it is as good as medicine. 

Example:
Question: Describe an advertisement image that conveys the following message:
I should drink Carling black beer because it is light. 
Response: The image features a vintage advertisement for Carling's Black Label Beer. The advertisement showcases a bottle of beer with a feather on top of it, giving it a unique and eye-catching appearance. The bottle is prominently displayed in the center of the image, with the feather extending from the top of the bottle.The advertisement also includes a quote, possibly a slogan, that reads "Light... as a Carling's." This phrase emphasizes the refreshing and light nature of the beer. The overall design of the advertisement is reminiscent of an old-fashioned poster, giving it a nostalgic and classic feel.
"""
output = model(prompt)
print(output)
# print(output)