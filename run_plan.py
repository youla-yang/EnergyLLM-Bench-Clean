import os
import subprocess
from benchmark import build_leaderboard
from predictor import EnergyPredictor

LOG_FILE = "logs/test_llm.jsonl"

def run(cmd):
    print(f"\n===== Running: {cmd} =====")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # -------------------------
    # 1. 推理实验 (Inference)
    # -------------------------
    inference_exps = [
        # 原有
        ("distilgpt2", 10, 128),
        ("distilgpt2", 10, 512),
        ("gpt2", 10, 128),
        ("gpt2", 10, 512),
        # 补充
        ("distilgpt2", 5, 1024),   # 超长文本
        ("gpt2", 5, 1024),
        ("gpt2-medium", 5, 128),   # 新模型
        ("gpt2-medium", 5, 512),
    ]
    for model, repeats, seq in inference_exps:
        run(f"python test_llm.py --mode infer --model_name {model} --repeats {repeats} --seq_len {seq}")

    # -------------------------
    # 2. 训练实验 (Training)
    # -------------------------
    training_exps = [
        # 原有
        ("distilgpt2", 2, 128),
        ("distilgpt2", 4, 128),
        ("distilgpt2", 2, 256),
        ("gpt2", 2, 128),
        ("gpt2", 4, 128),
        ("gpt2", 2, 256),
        # 补充
        ("distilgpt2", 8, 128),    # 大 batch
        ("gpt2", 8, 128),
        ("distilgpt2", 2, 512),    # 长序列
        ("gpt2", 2, 512),
    ]
    for model, bs, seq in training_exps:
        run(f"python test_llm.py --mode train --model_name {model} --batch_size {bs} --seq_len {seq} --repeats 3")

    # -------------------------
    # 3. 生成 Leaderboard
    # -------------------------
    if os.path.exists(LOG_FILE):
        df = build_leaderboard(LOG_FILE, output_csv="leaderboard.csv")
        print("\n✅ Leaderboard saved as leaderboard.csv")
        print(df.head())

    # -------------------------
    # 4. 训练能耗预测器
    # -------------------------
    p = EnergyPredictor()
    p.train(LOG_FILE)
    p.save()
    print("\n✅ Predictor saved as predictor.pkl")

    print("\n🎉 全流程完成！日志在 logs/test_llm.jsonl, 结果在 leaderboard.csv, 模型在 predictor.pkl")

if __name__ == "__main__":
    main()
