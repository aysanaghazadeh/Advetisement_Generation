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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


class RewardModel:
    def __init__(self, args):
        args.T2I_model = 'DMD'
        self.T2I_model = T2IModel(args)
        self.reward_function = PersuasivenessMetric(args)

    def get_reward(self, prompt, action_reason):
        action_reason = [ar for ar in action_reason.split('\n')]
        print('action-reason:', action_reason)
        print('prompt:', prompt.split(':')[-1])
        prompt = 'Generate the described image:\n' + prompt.split(':')[-1]
        image = self.T2I_model(prompt)

        persuasiveness = (self.reward_function.get_persuasiveness_alignment(image, action_reason) * 3 + \
                         self.reward_function.get_persuasiveness_score(image)) / 4
        return (persuasiveness - 3)


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
        peft_config=lora_config,
        load_in_4bit=True
    ).to(device=args.device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        peft_config=lora_config,
        load_in_4bit=True
    ).to(device='cuda:1')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint'
                                                                            '-4350/'),
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, ref_model


def train(args, local_rank):
    dist.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')
    config = PPOConfig(
        model_name="RLHFlow/LLaMA3-SFT",
        learning_rate=1.41e-1,
        batch_size=2,
        mini_batch_size=2,
        log_with='wandb',
    )
    model, tokenizer, ref_model = get_model()
    reward_model = RewardModel(args)
    model = DDP(model.to(device), device_ids=[local_rank])
    ref_model = DDP(ref_model.to(device), device_ids=[local_rank])
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
            batch["input_ids"] = [tokenizer.encode(batch['query']['query'][i], max_length=125) for i in range(len(batch['query']['query']))]
            query_tensors = batch["input_ids"]
            query_tensors = [torch.stack([torch.tensor(tensor) for tensor in query_tensor]) for query_tensor in query_tensors]
            response_tensors = [ppo_trainer.generate(query_tensor, **generation_kwargs) for query_tensor in query_tensors]
            response_tensors = [torch.stack([r.squeeze() for r in response_tensor]).squeeze() for response_tensor in response_tensors ]
            batch["response"] = [''.join([tokenizer.decode(r) for r in response_tensor]) for response_tensor in response_tensors]
            # print(batch['response'])
            texts = [r for r in batch["response"]]
            print(texts)
            pipe_outputs = [reward_model.get_reward(texts[i], batch["query"]['action_reason'][i]) for i in range(len(texts))]
            rewards = [torch.tensor(pipe_output).float() for pipe_output in pipe_outputs]
            print('reward:', rewards)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            ppo_trainer.save_pretrained(os.path.join(args.model_path, "my_ppo_model_DMD_batch_size_2"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = get_args()
    additional_args = parser.parse_args()
    train(args, additional_args.local_rank)
    # args = get_args()
    # train(args)