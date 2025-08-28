import json
import os
from codecarbon import EmissionsTracker
from datetime import datetime

class EnergyLogger:
    def __init__(self, output_file="logs/energy_logs.jsonl"):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.output_file = output_file
        # ⚡ 改这里：把 temp 文件写进 logs 文件夹，避免 PermissionError
        self.tracker = EmissionsTracker(
            output_file=os.path.join("logs", "codecarbon_temp.csv"),
            save_to_file=True
        )
        self.metadata = {}

    def start(self, **kwargs):
        """开始跟踪，并记录实验元数据"""
        self.metadata = kwargs
        self.metadata["timestamp"] = datetime.now().isoformat()
        self.tracker.start()

    def stop(self, flops=None, accuracy=None):
        """停止跟踪，把能耗 + FLOPs + accuracy 写入 JSONL"""
        emissions = self.tracker.stop()
        record = {
            **self.metadata,
            "flops": flops,
            "accuracy": accuracy,
            "emissions": emissions,  # kWh, CO2eq 等
        }
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

