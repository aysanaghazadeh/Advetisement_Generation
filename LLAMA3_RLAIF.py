import os.path
from util.data.data_util import get_LLAMA3_RLAIF_Dataloader
from configs.training_config import get_args
from transformers import AutoTokenizer
import torch
from peft import LoraConfig
from T2I_models.T2I_model import T2IModel
from Evaluation.metrics import PersuasivenessMetric
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


class RewardModel:
    def __init__(self, args):
        args.T2I_model = 'DMD'
        self.T2I_model = T2IModel(args)
        self.reward_function = PersuasivenessMetric(args)

    def get_reward(self, prompt, action_reason):
        action_reason = [ar[0] for ar in action_reason]
        print('action-reason:', action_reason)
        print('prompt:', prompt.split(':')[-1])
        prompt = 'Generate the described image:\n' + prompt.split(':')[-1]
        image = self.T2I_model(prompt)

        persuasiveness = self.reward_function.get_persuasiveness_alignment(image, action_reason)
        return persuasiveness - 1


def get_model():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_id = os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/')
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        # device_map='auto',
        peft_config=lora_config,
        load_in_4bit=True
    ).to(device=args.device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        # device_map='auto',
        peft_config=lora_config,
        load_in_4bit=True
    ).to(device='cuda:1')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint'
                                                                            '-4350/'),
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, ref_model


def train(args):
    config = PPOConfig(
        model_name="RLHFlow/LLaMA3-SFT",
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,

    )
    model, tokenizer, ref_model = get_model()
    reward_model = RewardModel(args)
    dataset = get_LLAMA3_RLAIF_Dataloader(args)
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    generation_kwargs = {
        "min_length": 1,
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 125,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for epoch in tqdm(range(args.epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            query_tensors = [torch.stack([torch.tensor(tensor.item()) for tensor in query_tensors])]
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            texts = [r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model.get_reward(texts[0], batch["query"]['action_reason'])
            rewards = [torch.tensor(pipe_outputs).float()]
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            ppo_trainer.save_pretrained(os.path.join(args.model_path, "my_ppo_model_DMD"))

if __name__ == '__main__':
    args = get_args()
    train(args)