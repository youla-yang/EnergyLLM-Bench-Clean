import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
import json

class EnergyPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def _extract_energy(self, e):
        """兼容 float 和 dict 两种 emissions 格式"""
        if isinstance(e, dict):
            return e.get("emissions", {}).get("energy_consumed", None)
        else:
            return e  # 如果是 float，就直接返回

    def train(self, jsonl_file):
        """读取 JSONL 日志，训练 FLOPs → 能耗 的预测器"""
        with open(jsonl_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        df = pd.DataFrame(data)

        # 兼容 float 和 dict
        def extract_energy(e):
            if isinstance(e, dict):
                return e.get("emissions", {}).get("energy_consumed", None)
            return e

        df["energy_kWh"] = df["emissions"].apply(extract_energy)

        # 丢掉无效数据
        df = df.dropna(subset=["flops", "energy_kWh"])

        X = df["flops"].values.reshape(-1, 1)
        y = df["energy_kWh"].values

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"训练完成: MAPE={mape:.3f}, R²={r2:.3f}")


    def predict(self, flops):
        return self.model.predict([[flops]])[0]

    def save(self, path="predictor.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="predictor.pkl"):
        self.model = joblib.load(path)
