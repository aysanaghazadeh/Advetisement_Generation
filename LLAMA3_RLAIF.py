import os.path
from util.data.data_util import get_LLAMA3_RLAIF_Dataloader
from configs.training_config import get_args
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from T2I_models.T2I_model import T2IModel
from Evaluation.metrics import PersuasivenessMetric
from torch.optim import Adam
import tqdm

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
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
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
        print(f'torch cuda count: {torch.cuda.device_count()}')
        model.is_parallelizable = True
        model.model_parallel = True
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              token='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')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.model_path+'/my_LLAMA3_large_sample_model',
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=40000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
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


def train(args):
    model, tokenizer = get_model()
    optimizer = Adam(model.parameters(), lr=args.lr)
    reward_model = RewardModel(args)
    # training_args = get_training_args(args)
    train_loader = get_LLAMA3_RLAIF_Dataloader(args)
    for epoch in tqdm(range(args.epochs)):
        for i, (prompt, inputs) in enumerate(train_loader):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f'prompt: {prompt}')
            inputs = inputs.to(device=args.device)
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            description = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = model(**inputs)
            logits = output.logits
            reward_value = reward_model.get_reward(description)
            loss = -reward_value * logits.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f'prompt: {prompt}')
            print(f"Generated Description: {description}")
            print(f"Loss: {loss.item()}")
            print(f"Reward: {reward_value}")
            print("-" * 50)
            model.save_pretrained(args.model_path + '/my_LLAMA3_RLAIF_model')
            tokenizer.save_pretrained(args.model_path + '/my_LLAMA3_RLAIF_tokenizer')

if __name__ == '__main__':
    args = get_args()
    train(args)