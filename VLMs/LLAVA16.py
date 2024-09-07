from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


class LLAVA16(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf",
                                                                       torch_dtype=torch.float16,
                                                                       load_in_8bit=True,
                                                                       low_cpu_mem_usage=True)

    def forward(self, image, prompt, generate_kwargs={"max_new_tokens": 250}):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {"type": "image"},
                    ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
        output = self.model.generate(**inputs,
                                     max_new_tokens=generate_kwargs["max_new_tokens"],
                                     temperture=0)
        output = self.processor.decode(output[0], skip_special_tokens=True)
        output = output.split('ASSISTANT:')[-1]
        return output