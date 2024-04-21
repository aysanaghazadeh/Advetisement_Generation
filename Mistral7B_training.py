import os.path
from util.data.data_util import get_train_Mistral7B_Dataloader
from configs.training_config import get_args
from LLMs.Mistral7B import Mistral7B
from transformers import TrainingArguments, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def get_model(args):
    model = Mistral7B(args).model
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def get_training_args(args):
    # training_args = TrainingArguments(
    #     output_dir=args.model_path,
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=1,
    #     weight_decay=args.weight_decay,
    #     save_total_limit=3,
    #     num_train_epochs=args.epochs,
    #     logging_dir=os.path.join(args.results, 'logs'),
    # )
    training_args = TrainingArguments(
        output_dir=args.model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        warmup_steps=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        bf16=True,
        lr_scheduler_type='constant',
        save_total_limit=3,
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
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     tokenizer=tokenizer,
    #     peft_config=peft_config,
    #     packing=True,
    # )
    max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=args,
        train_dataset=train_dataset,
        remove_unused_columns=True
    )
    trainer.train()
    model.save_pretrained(args.model_path + '/my_mistral_model')
    tokenizer.save_pretrained(args.model_path + '/my_mistral_tokenizer')
