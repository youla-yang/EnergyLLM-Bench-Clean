# EnergyLLM-Bench

**EnergyLLM-Bench** is a system-level benchmark for **measuring, predicting, and comparing** the energy consumption and carbon footprint of large language models (LLMs).  
It provides a **reproducible pipeline**, **standardized logging format**, and an **open leaderboard** to enable transparent and extensible evaluation of LLM energy footprints.

---

## 🔋 Key Features
- **Measurement Layer**  
  Collects GPU/CPU energy usage, FLOPs, latency, and task-specific quality metrics via NVML/RAPL.

- **Standardized Logging**  
  JSONL-based protocol with full metadata (model, optimizer, precision, device, seed) for reproducibility.

- **Prediction Layer**  
  FLOPs-based analytic model calibrated with empirical GPU runs for lightweight energy estimation.

- **Open Leaderboard**  
  Aggregated results across models, tasks, and devices; continuously extensible with community contributions.

---

## 📊 Example Results

### Leaderboard Entries (Individual Runs)
| model      | task            | flops       | accuracy | energy_kWh |
|------------|-----------------|-------------|----------|------------|
| distilgpt2 | inference-short | 1.63e+09    |   –      | 2.3e-04    |
| distilgpt2 | inference-long  | 5.82e+10    |   –      | 2.2e-04    |
| gpt2       | inference-short | 2.49e+09    |   –      | 4.9e-04    |
| gpt2       | inference-long  | 8.83e+10    |   –      | 4.6e-04    |
| …          | …               | …           | …        | …          |

### Aggregated Summary (Across Models & Devices)
| model                  | params | device | flops      | emissions | efficiency | runs |
|------------------------|--------|--------|------------|-----------|------------|------|
| EleutherAI/pythia-1b   | 1.0B   | GPU    | 1.30e+13   | 1.88e-04  | 3.1e-15    | 5    |
| bigscience/bloom-1b1   | 1.1B   | GPU    | 6.91e+12   | 2.92e-04  | 4.0e-15    | 5    |
| distilgpt2             | 82M    | CPU    | 4.60e+12   | 1.47e-03  | 3.2e-14    | 16   |
| facebook/opt-1.3b      | 1.3B   | GPU    | 1.70e+13   | 2.55e-04  | 2.0e-15    | 5    |
| mistralai/Mistral-7B   | 7B     | GPU    | 8.15e+11   | 1.79e-04  | 1.4e-15    | 4    |

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/youla-yang/EnergyLLM-Bench.git
cd EnergyLLM-Bench

**So ready to "change the world one run at a time"? Let's start with a very quick set up.**
conda create -n energyllm python=3.12 -y
conda activate energyllm
pip install -r requirements.txt

python test_llm.py
**this will Load the specified LLM

Run inference with given settings

Collect FLOPs, runtime, latency, CPU energy

Save results into JSONL/CSV logs**




  **these files can be run on the colab**


- **`energyllm_bench_runner.py`**  
  **this will Load the specified LLM

Run inference with given settings

Collect FLOPs, runtime, latency, CPU energy

Save results into JSONL/CSV logs and leadboard.csv and compared with codecarbon**


- **`energy_predictor.py`**  
  FLOPs→emissions predictor. Trains regression models to estimate CO₂ emissions from FLOPs. Evaluates MAPE, R², and produces scatter/error plots.  
  *Implements the **Prediction Layer** of EnergyLLM-Bench.*

- **`dense_vs_moe.py`**  
  Comparative experiment for dense Transformers vs. Mixture-of-Experts (MoE). Measures **J/token**, computes **perplexity (PPL)**, and visualizes the Pareto tradeoff (efficiency vs. quality).  
  *Highlights architectural differences in efficiency within EnergyLLM-Bench.*


📖 Citation

If you use EnergyLLM-Bench in your research, please cite:

@article{yang2025energyllmbench,
  title  = {EnergyLLM-Bench: A Benchmark for Measuring and Predicting the Energy Footprint of Large Language Models},
  author = {Yang, Youla and collaborators},
  year   = {2025},
 
}