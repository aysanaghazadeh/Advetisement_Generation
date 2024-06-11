import os.path
from util.data.trian_test_split import get_train_data
from util.data.data_util import get_LLAMA3_RLAIF_Dataloader
from configs.training_config import get_args
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from T2I_models.T2I_model import T2IModel
from Evaluation.metrics import PersuasivenessMetric
from torch.optim import Adam
from tqdm import tqdm
import json
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from accelerate import Accelerator


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
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     # bnb_4bit_quant_type="nf4",
    #     # bnb_4bit_use_double_quant=True,
    #     bnb_8bit_compute_dtype=torch.bfloat16
    # )
    # model = AutoModelForCausalLMWithValueHead.from_pretrained("meta-llama/Meta-Llama-3-8B",
    #                                              token='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',
    #                                              device_map='auto',
    #                                              quantization_config=bnb_config)
    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    # peft_config = LoraConfig(inference_mode=False,
    #                          r=8,
    #                          lora_alpha=32,
    #                          lora_dropout=0.2,
    #                          peft_type=TaskType.CAUSAL_LM)
    # model = get_peft_model(model, peft_config)
    # print(f'model\'s trainable parameters: {model.print_trainable_parameters()}')
    # if torch.cuda.device_count() > 1:
    #     print(f'torch cuda count: {torch.cuda.device_count()}')
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_id = "RLHFlow/LLaMA3-SFT"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        token='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',
        # device_map='auto',
        peft_config=lora_config,
        load_in_4bit=True
    ).to(device=args.device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def train(args):
    config = PPOConfig(
        model_name="RLHFlow/LLaMA3-SFT",
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1
    )
    model, tokenizer = get_model()
    reward_model = RewardModel(args)
    dataset = get_LLAMA3_RLAIF_Dataloader(args)
    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    generation_kwargs = {
        # "min_length": -1,
        # "top_k": 0.0,
        # "top_p": 1.0,
        "max_new_tokens": 2,
        # "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    for epoch in tqdm(range(args.epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            print(batch)
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            print(response_tensors)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model.get_reward(texts[0])
            # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            rewards = [torch.tensor(pipe_outputs)]
            print(rewards)
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            print(f'epoch: {epoch} \n {stats}')

    #### Save model
    ppo_trainer.save_pretrained("my_ppo_model")
    # train_images = get_train_data(args)
    # QA = json.load(open(os.path.join(args.data_path, args.test_set_QA)))
    # for epoch in tqdm(range(args.epochs)):
    #     for i, image_url in enumerate(train_images):
    #         action_reason = QA[image_url[0]][0]
    #         prompt = f"""Describe an advertisement image that conveys the following messages in detail:
    #                             {action_reason}
    #                             Description of the image:
    #                         """
    #         inputs = tokenizer(prompt, return_tensors="pt").to(device=args.device)
    #         print(f"Epoch {epoch + 1}/{args.epochs}")
    #         print(f'prompt: {prompt}')
    #         inputs = inputs.to(device=args.device)
    #         generated_ids = model.generate(**inputs, max_new_tokens=10)
    #         description = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #         output = model(**inputs)
    #         logits = output.logits
    #         reward_value = reward_model.get_reward(description)
    #         loss = -reward_value * logits.mean()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch {epoch + 1}/{args.epochs}")
    #         print(f'prompt: {prompt}')
    #         print(f"Generated Description: {description}")
    #         print(f"Loss: {loss.item()}")
    #         print(f"Reward: {reward_value}")
    #         print("-" * 50)
    #         model.save_pretrained(args.model_path + '/my_LLAMA3_RLAIF_model')
    #         tokenizer.save_pretrained(args.model_path + '/my_LLAMA3_RLAIF_tokenizer')

if __name__ == '__main__':
    args = get_args()
    train(args)