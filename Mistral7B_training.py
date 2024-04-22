import os.path
from util.data.data_util import get_train_Mistral7B_Dataloader
from configs.training_config import get_args
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
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
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def get_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.model_path,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=5,
        do_eval=True,
        label_names=["input_ids", "labels", "attention_mask"],
        report_to="none",
        logging_dir=os.path.join(args.results, 'logs')
    )
    # training_args = TrainingArguments(
    #     output_dir=args.model_path,
    #     num_train_epochs=args.epochs,
    #     per_device_train_batch_size=4,
    #     warmup_steps=0.03,
    #     logging_steps=10,
    #     save_strategy="epoch",
    #     evaluation_strategy="epoch",
    #     learning_rate=args.lr,
    #     bf16=True,
    #     lr_scheduler_type='constant',
    #     save_total_limit=3,
    #     logging_dir=os.path.join(args.results, 'logs'),
    #     remove_unused_columns=True
    # )
    if not os.path.exists(os.path.join(args.results, 'logs')):
        os.makedirs(os.path.join(args.results, 'logs'))
    return training_args


if __name__ == '__main__':
    args = get_args()
    model, tokenizer = get_model()
    training_args = get_training_args(args)
    train_dataset = get_train_Mistral7B_Dataloader(args)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    test_dataset = tmp["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args
    )
    trainer.train()
    model.save_pretrained(args.model_path + '/my_mistral_model')
    tokenizer.save_pretrained(args.model_path + '/my_mistral_tokenizer')
