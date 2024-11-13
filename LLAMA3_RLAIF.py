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
from transformers import Adafactor
import os
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='fp16')
print(accelerator)



class RewardModel:
    def __init__(self, args):
        # args.T2I_model = 'DMD'
        self.T2I_model = T2IModel(args)
        self.reward_function = PersuasivenessMetric(args)
        pass

    def get_reward(self, prompt, action_reason):
        action_reason = [ar for ar in action_reason.split('\n')]
        print('action-reason:', action_reason)
        print('prompt:', prompt.split(':')[-1])
        prompt = 'Generate the described image:\n' + prompt.split(':')[-1]
        image = self.T2I_model(prompt)

        persuasiveness = self.reward_function.get_persuasiveness_alignment(image, action_reason)
        return persuasiveness
        # return 4

def get_model():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    current_device = Accelerator().local_process_index
    # model_id = os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/')
    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        peft_config=lora_config,
        load_in_4bit=True,
    ).to(device=args.device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        peft_config=lora_config,
        load_in_4bit=True,
    ).to(device='cuda')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',
                                              # os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint'
                                              #                               '-4350/'),
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, ref_model


def train(args):
    config = PPOConfig(
        model_name="RLHFlow/LLaMA3-SFT",
        learning_rate=1.41e-1,
        batch_size=2,
        mini_batch_size=2,
        log_with='wandb',
    )
    model, tokenizer, ref_model = get_model()
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
    reward_model = RewardModel(args)
    dataset = get_LLAMA3_RLAIF_Dataloader(args)
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        optimizer=optimizer
    )
    generation_kwargs = {
        "min_length": 1,
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 50,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for epoch in tqdm(range(args.epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            batch["input_ids"] = [tokenizer.encode(batch['query']['query'][i], max_length=125) for i in
                                  range(len(batch['query']['query']))]
            query_tensors = batch["input_ids"]
            query_tensors = [torch.stack([torch.tensor(tensor) for tensor in query_tensor]) for query_tensor in
                             query_tensors]
            response_tensors = [ppo_trainer.generate(query_tensor, **generation_kwargs) for query_tensor in
                                query_tensors]
            response_tensors = [torch.stack([r.squeeze() for r in response_tensor]).squeeze() for response_tensor in
                                response_tensors]
            batch["response"] = [''.join([tokenizer.decode(r) for r in response_tensor]) for response_tensor in
                                 response_tensors]
            # print(batch['response'])
            texts = [r for r in batch["response"]]
            print(texts)
            pipe_outputs = [reward_model.get_reward(texts[i], batch["query"]['action_reason'][i]) for i in
                            range(len(texts))]
            rewards = [torch.tensor(pipe_output).float() for pipe_output in pipe_outputs]
            print('reward:', rewards)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            ppo_trainer.save_pretrained(os.path.join(args.model_path, "my_ppo_model_llama_instruct"))


if __name__ == '__main__':
    args = get_args()
    train(args)
