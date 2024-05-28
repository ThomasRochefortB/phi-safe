import argparse
import os
import requests
import torch
from datasets import load_dataset
from safe.tokenizer import SAFETokenizer
from safe.trainer.collator import SAFECollator
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from safe.tokenizer import SAFESplitter
from trl import SFTTrainer
from tokenizers.pre_tokenizers import Whitespace

def download_file(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        f.write(response.content)

def check_and_download_tokenizer(tokenizer_path):
    tokenizer_url = "https://huggingface.co/datamol-io/safe-gpt/resolve/main/tokenizer.json"
    config_url = "https://huggingface.co/datamol-io/safe-gpt/resolve/main/config.json"
    if not os.path.exists(tokenizer_path):
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        print(f"Downloading {tokenizer_url} to {tokenizer_path}...")
        download_file(tokenizer_url, tokenizer_path)
        print("Download complete.")
    config_path = os.path.join(os.path.dirname(tokenizer_path), "config.json")
    if not os.path.exists(config_path):
        print(f"Downloading {config_url} to {config_path}...")
        download_file(config_url, config_path)
        print("Download complete.")

def get_linear_layer_names(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    return linear_layers

def custom_tokenize(example, splitter):
    if 'input' in example and example['input'] is not None:
        tokens = splitter.tokenize(example['input'])
        return {"input": " ".join(tokens)}
    else:
        return {}

def filter_empty_examples(example):
    return 'input' in example and example['input']

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    check_and_download_tokenizer(args.tokenizer_path)

    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

    tokenizer = SAFETokenizer().load(args.tokenizer_path)
    tokenizer = tokenizer.get_pretrained()
    tokenizer.model_max_length = args.max_seq_length

    model_id = args.model_id
    config = AutoConfig.from_pretrained(model_id)

    config.vocab_size = len(tokenizer)
    config.max_position_embeddings = args.max_seq_length
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.task_type = "causal_lm"

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"{model_path} not found. Creating it from {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.resize_token_embeddings(len(tokenizer))
        model.save_pretrained(model_path)
        print(f"Model saved to {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )

    linear_layer_names = get_linear_layer_names(model)
    print("Linear layers:", linear_layer_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=linear_layer_names
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    embed_tokens_layer = model.base_model.model.model.embed_tokens
    for param in embed_tokens_layer.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if 'embed_tokens' in name:
            print(f"{name} requires_grad: {param.requires_grad}")
    model.print_trainable_parameters()

    trainable_modules = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Trainable modules:")
    for module in trainable_modules:
        print(module)

    splitter = SAFESplitter()
    train_dataset = train_dataset.map(lambda x: custom_tokenize(x, splitter), batched=False, num_proc=32)
    eval_dataset = eval_dataset.map(lambda x: custom_tokenize(x, splitter), batched=False, num_proc=32)

    train_dataset = train_dataset.filter(filter_empty_examples, num_proc=32)
    eval_dataset = eval_dataset.filter(filter_empty_examples, num_proc=32)

    tokenizer._tokenizer.pre_tokenizer = Whitespace()

    training_args = TrainingArguments(
        fp16=args.fp16,
        bf16=args.bf16,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        log_level="info",
        optim="adafactor",
        logging_steps=10,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_accumulation_steps=1,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        report_to="tensorboard",
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=10,
        save_total_limit=5,
        save_steps=100,
        torch_compile=True,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="input",
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=64,
        data_collator=SAFECollator(tokenizer, input_key="input", max_length=args.max_seq_length)
    )

    torch.cuda.empty_cache()

    if args.checkpoint_path:
        train_result = trainer.train(resume_from_checkpoint=args.checkpoint_path)
    else:
        train_result = trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for SAFE GPT model.")
    parser.add_argument("--dataset_name", type=str, default="datamol-io/safe-gpt", help="Name of the dataset.")
    parser.add_argument("--train_split", type=str, default="train[:5%]", help="Training data split.")
    parser.add_argument("--eval_split", type=str, default="test[10%:15%]", help="Evaluation data split.")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json", help="Path to the tokenizer file.")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-1_5", help="Model ID for configuration.")
    parser.add_argument("--model_path", type=str, default="phi1_5_updated", help="Path to the pretrained model.")
    parser.add_argument("--output_dir", type=str, default=".saved_model/phi1_5-safmol_0528", help="Output directory for the trained model.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--learning_rate", type=float, default=2.0e-05, help="Learning rate for training.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for training.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to resume training from a checkpoint.")
    args = parser.parse_args()

    main(args)
