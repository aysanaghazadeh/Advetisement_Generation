import os.path
from util.data.data_util import get_LLAMA3_DPO_Dataloader
from configs.training_config import get_args
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import LoraConfig
from T2I_models.T2I_model import T2IModel
from Evaluation.metrics import PersuasivenessMetric
from tqdm import tqdm
from trl import DDPOConfig, DPOTrainer


class RewardModel:
    def __init__(self, args):
        args.T2I_model = 'SDXL'
        self.T2I_model = T2IModel(args)
        self.reward_function = PersuasivenessMetric()

    def get_reward(self, prompt):
        prompt = 'Generate the described image:\n' + prompt
        image = self.T2I_model(prompt)
        persuasiveness = self.reward_function.get_persuasiveness_score(image)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token='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',
        device_map='auto',
        # peft_config=lora_config,
        # load_in_8bit=True
    )#.to(device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint'
                                                                            '-4350/'),
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def train(args):
    # config = PPOConfig(
    #     model_name="RLHFlow/LLaMA3-SFT",
    #     learning_rate=1.41e-5,
    #     batch_size=1,
    #     mini_batch_size=1
    # )
    model, tokenizer = get_model()
    # reward_model = RewardModel(args)
    dataset = get_LLAMA3_DPO_Dataloader(args)
    training_args = DDPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=6000,
        logging_steps=10,
        save_steps=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        learning_rate=5e-4,
        eval_strategy="steps",
        eval_steps=10,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama3",
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_length=250,
    )
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 25,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    dpo_trainer.train()
    dpo_trainer.save_model(os.path.join(args.model_path, 'llama3_dpo'))


if __name__ == '__main__':
    args = get_args()
    train(args)