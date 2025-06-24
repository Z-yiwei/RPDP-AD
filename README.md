# RPDP-AD: Unsupervised Anomaly Detection for Aircraft PRSOV via Random Projection-Based Inner Product Prediction
This repository provides the official implementation of our IEEE TIM paper:

**Unsupervised Anomaly Detection for Aircraft PRSOV with Random Projection-Based Inner Product Prediction**,  
Accepted to **IEEE Transactions on Instrumentation and Measurement**, 2025.

**RPDP-AD** (Random Projection Distance Prediction for Anomaly Detection) is an **unsupervised** learning method for detecting anomalies in **Pressure Regulated Shutoff Valves (PRSOV)** — a critical component in aircraft Environmental Control Systems (ECS). It features:

- 💡 A novel training signal: **predict inner products in random projection space**
- 📉 Distribution difference indicator for **uncertainty-aware detection**
- 🔬 Theoretical support based on the **Johnson–Lindenstrauss lemma**
- ⚙️ Lightweight architecture via **MLP**, suitable for real-time onboard inference

## 📂 Dataset Description

We use a simulated PRSOV dataset including **8,000 samples** (600 normal training, 400 normal testing, 7 types of faults, 1,000 each). Each sample includes **201 pressure-related features**.

| Type                     | Count |
|--------------------------|-------|
| Normal (train)           | 600   |
| Normal (test)            | 400   |
| Charge Fault             | 1000  |
| Discharge Fault          | 1000  |
| Friction Fault           | 1000  |
| Charge + Discharge       | 1000  |
| Charge + Friction        | 1000  |
| Discharge + Friction     | 1000  |
| Charge + Discharge + Friction | 1000  |

Simulated data is generated via MATLAB Simulink under expert-guided fault injection.

## ⚙️ Environment

- Python 3.8+
- PyTorch >= 1.10
- NumPy, scikit-learn, matplotlib

## 📚 Citation 
If you use this repository or would like to refer to the paper, please use the following:
```bash
@ARTICLE{Peng2025RPDP,
  author={Peng, Dandan and Zhu, Ning and Han, Te and Chen, Zhuyun and Liu, Chenyu},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Unsupervised Anomaly Detection For Aircraft PRSOV With Random Projection-Based Inner Product Prediction}, 
  year={2025},
  volume={74},
  number={},
  pages={1-11},
  doi={10.1109/TIM.2025.3568969}
}
```
