import os.path
from util.data.data_util import get_train_Mistral7B_Dataloader
from configs.training_config import get_args
from LLMs.Mistral7B import Mistral7B
from transformers import TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, LoftQConfig


def get_model(args):
    model = Mistral7B(args)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    return model, tokenizer


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.model_path,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(args.results, 'logs'),
    )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args

if __name__ == '__main__':
    args = get_args()
    model, tokenizer = get_model(args)
    training_args = get_training_args(args)
    train_dataset = get_train_Mistral7B_Dataloader(args)
    loftq_config = LoftQConfig(loftq_bits=4)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        loftq_config=loftq_config
    )
    model = get_peft_model(model, peft_config)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     peft_config=peft_config,
    #     packing=True,
    # )
    # trainer.train()
    model.save_pretrained(args.model_path + '/my_mistral_model')
    tokenizer.save_pretrained(args.model_path + '/my_mistral_tokenizer')
