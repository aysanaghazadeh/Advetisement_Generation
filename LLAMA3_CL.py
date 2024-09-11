from util.data.data_util import get_train_LLAMA3_CPO_Dataloader
from configs.training_config import get_args
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import CPOConfig, CPOTrainer, ModelConfig, get_peft_config
import os

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
    training_args = CPOConfig(
        output_dir=args.model_path+'/my_LLAMA3_CPO',
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


if __name__ == '__main__':
    args = get_args()
    cpo_args = get_training_args(args)
    model, tokenizer = get_model()
    cpo_config = CPOConfig(beta=0.1,
                           output_dir=args.model_path+'/my_LLAMA3_CPO',)
    train_dataset = get_train_LLAMA3_CPO_Dataloader(args)
    tmp = train_dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    eval_dataset = tmp["test"]
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = CPOTrainer(
        model,
        args=cpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # train and save the model
    trainer.train()
    trainer.save_model(cpo_args.output_dir)