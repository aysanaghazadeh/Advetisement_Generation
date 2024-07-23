import os.path
from util.data.data_util import get_RLHF_train_LLAMA3_Dataloader
from configs.training_config import get_args
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import RewardTrainer, SFTTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import Dataset
from random import choices
from tqdm import tqdm
import time
import numpy as np


def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                 token='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',
                                                 device_map='auto',
                                                 quantization_config=bnb_config)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             peft_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    print(f'model\'s trainable parameters: {model.print_trainable_parameters()}')
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.model_path,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=800,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        # optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=10,
        do_eval=True,
        label_names=["input_ids", "labels", "attention_mask"],
        report_to="none",
        logging_dir=os.path.join(args.results, 'logs')
    )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args


def get_RM_training_args(args):
    training_args = TrainingArguments(
        output_dir=f"{args.model_path}/rm_checkpoint",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        learning_rate=1e-5,
        save_total_limit=1,
        no_cuda=True
    )
    return training_args


def get_score(model, tokenizer, prompt, response):
    instructions = tokenizer.encode_plus(prompt,
                                       response,
                                       padding="max_length",
                                       max_length=256,
                                       return_tensors="pt",
                                        truncation=True)
    with torch.no_grad():
        outputs = model(**instructions)

    logits = outputs[0]

    return logits


def train_reward_model(args):
    model, tokenizer = get_model()
    print(model.config)
    training_args = get_RM_training_args(args)
    train_dataset = get_RLHF_train_LLAMA3_Dataloader(args)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    test_dataset = tmp["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(args.model_path + '/rm_checkpoint')
    tokenizer.save_pretrained(args.model_path + '/rm_checkpoint')

def format_dataset(train_dataset):
    txt_in_len = 5
    txt_out_len = 20
    seed = 1
    train_dataset = train_dataset.filter(lambda x: len(x["prompt"]) > 500, batched=False)
    train_dataset = train_dataset.map(lambda x: {"prompt": x["prompt"][:1000]}, batched=False)
    model, tokenizer = get_model()
    txt_in_len = 5
    txt_out_len = 32
    seed = 1
    train_dataset = train_dataset.map(
        lambda x: {"input_ids":
                       tokenizer.encode(" " + x["chosen"], return_tensors="pt", truncation=True, padding="max_length",
                                        max_length=32)[0]},
        batched=False,
    )
    train_dataset = train_dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"])}, batched=False)
    train_dataset = train_dataset[:20480]
    train_dataset = Dataset.from_dict(train_dataset)
    train_dataset.set_format("pytorch")
    return train_dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit

def get_score(model, tokenizer, responses):
    positive_logist = []
    for i in responses:
        instructions = tokenizer.encode_plus(
            i,
            padding="max_length",
            max_length=32,
            return_tensors="pt")
        with torch.no_grad():
            outputs = model(**instructions)

        logits = outputs[0].mean()
        positive_logist.append(logits)

    return positive_logist


if __name__ == '__main__':
    args = get_args()
    # training reward model
    print('training reward model started...')
    train_reward_model(args)
    # train the policy model
    model, tokenizer = get_model()
    sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}
    config = PPOConfig(
        model_name='meta-llama/Meta-Llama-3-8B', steps=51200, learning_rate=1.41e-5, remove_unused_columns=True
    )
    starcoder_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path + '/rm_checkpoint')

    train_dataset = get_RLHF_train_LLAMA3_Dataloader(args)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    test_dataset = tmp["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataset = format_dataset(train_dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    ppo_trainer = PPOTrainer(config, model, model, tokenizer, dataset=train_dataset,
                             data_collator=data_collator, optimizer=optimizer)
    ctrl_str = ["[negative]", "[positive]"]
    ctrl_tokens = dict((s, tokenizer.encode(s, return_tensors="pt").squeeze().to(args.device)) for s in ctrl_str)
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
        "eos_token_id": -1,
    }

    txt_in_len = 5
    txt_out_len = 32
    seed = 1

    for epoch in range(1):
        for batch in tqdm(ppo_trainer.dataloader):
            (logs, game_data,) = (
                dict(),
                dict(),
            )

            print(ctrl_str)
            #### prepend a random control token
            task_list = choices(ctrl_str, k=config.batch_size)
            game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
            query_tensors = [torch.cat((ctrl_tokens[t], input_ids)) for t, input_ids in
                             zip(task_list, batch["input_ids"])]

            #### get response from gpt2
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-txt_out_len:])
            #         print(response_tensors)
            game_data["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### sentiment analysis
            texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
            logits = get_score(model, tokenizer, texts)
            rewards = pos_logit_to_reward(logits, task_list)

            #### Run PPO training
            t = time.time()
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            for cs in ctrl_str:
                key = "env/reward_" + cs.strip("[]")
                stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
            ppo_trainer.log_stats(stats, game_data, rewards)
