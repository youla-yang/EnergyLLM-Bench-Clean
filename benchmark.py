import pandas as pd
import json

def build_leaderboard(jsonl_file, output_csv="leaderboard.csv"):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    # 提取能耗 (兼容 float 和 dict 两种格式)
    def extract_energy(e):
        if isinstance(e, dict):
            return e.get("emissions", {}).get("energy_consumed", None)
        else:
            return e  # 如果已经是 float，就直接用

    df["energy_kWh"] = df["emissions"].apply(extract_energy)

    # 只保留主要指标（有些字段可能日志里没有，需要容错）
    keep_cols = [c for c in ["model", "task", "optimizer", "precision", "flops", "accuracy", "energy_kWh"] if c in df.columns]
    df = df[keep_cols]

    # 保存 leaderboard
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print("✅ Leaderboard 已保存:", output_csv)
    return df
