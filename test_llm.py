import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from logger import EnergyLogger

# ============ 功能函数 ============
def compute_flops(model, seq_len=128, batch_size=1, steps=10):
    """简单估算 FLOPs (基于参数量 * 序列长度 * 步数)"""
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * seq_len * steps * batch_size

# ============ 推理模式 ============
def run_inference(model_name, device, output_log, repeats=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # 两种输入场景：短文本 / 长文本
    input_texts = [
        ("short", "Hello world"),
        ("long", "This is a long input example. " * 10),  # ~100 tokens
    ]

    for tag, text in input_texts:
        print(f"\n===== Inference: {model_name}, input={tag}, repeats={repeats} =====")
        inputs = tokenizer(text, return_tensors="pt").to(device)

        logger = EnergyLogger(output_log)
        logger.start(model=model_name, task=f"inference-{tag}")

        start = time.time()
        with torch.no_grad():
            for i in range(repeats):
                _ = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id
                )
        duration = time.time() - start

        flops = compute_flops(model, seq_len=inputs["input_ids"].shape[1],
                              steps=repeats, batch_size=1)
        record = logger.stop(flops=flops, accuracy=None)
        print(f"✅ [{model_name}][{tag}] Done, duration={duration:.2f}s, record={record}")

# ============ 训练模式 ============
def run_training(model_name, device, output_log, batch_size=2, seq_len=128):
    print(f"\n===== Training: {model_name}, batch={batch_size}, seq_len={seq_len} =====")

    dataset = load_dataset("imdb", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True,
                           padding="max_length", max_length=seq_len)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator
    )

    logger = EnergyLogger(output_log)
    logger.start(model=model_name, task=f"training-b{batch_size}-s{seq_len}")

    start = time.time()
    trainer.train()
    duration = time.time() - start

    flops = compute_flops(model, seq_len=seq_len,
                          steps=len(tokenized), batch_size=batch_size)
    record = logger.stop(flops=flops, accuracy=None)
    print(f"✅ Training done, duration={duration:.2f}s, record={record}")

# ============ 主入口 ============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="infer")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="logs/test_llm.jsonl")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "infer":
        run_inference(args.model_name, DEVICE, args.log_file, repeats=args.repeats)
    elif args.mode == "train":
        run_training(args.model_name, DEVICE, args.log_file,
                     batch_size=args.batch_size, seq_len=args.seq_len)
