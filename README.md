# AdaKD: LLM-Oriented Token-Adaptive Knowledge Distillation

[![Arxiv](https://img.shield
s.io/badge/Arxiv-2510.11615-B31B1B.svg)](https://arxiv.org/abs/2510.11615)
[![Venue](https://img.shields.io/badge/Venue-AAAI--2026-blue.svg)](https://aaai.org/)

Official implementation of the paper **"LLM-Oriented Token-Adaptive Knowledge Distillation"** (AAAI 2026).

AdaKD is a plug-and-play framework for logit-based distillation that dynamically adapts to the student's learning state. It features two synergistic modules:
*   **Loss-driven Adaptive Token Focusing (LATF):** Concentrates distillation on valuable tokens by monitoring learning stability.
*   **Inverse Difficulty Temperature Scaling (IDTS):** Applies token-level temperatures—low for hard tokens (error correction) and high for easy tokens (better generalization).

---

## 🛠️ Installation

> [!NOTE]
> The `requirements.txt` and specific environment setup scripts are currently being finalized.
> 
---

## 📂 Data Preparation

The training data is based on the **databricks-dolly-15k** dataset. You can download our processed version here:

*   **Processed Data:** [Google Drive Link](https://drive.google.com/file/d/1XVZw7hd7TJ13Z9flGdhvIVld_l0S7c4d/view?usp=drive_link)

Please place the downloaded data in the `data/` directory.

---

## 🚀 Training and Evaluation

> [!IMPORTANT]
> Bash scripts for automated training and evaluation are coming soon.

---


## 🤝 Acknowledgements
Our code is built upon the following open-source projects:
*   [distillm](https://github.com/jongwooko/distillm): Towards Streamlined Distillation for Large Language Models.
*   [minillm](https://github.com/microsoft/LMOps/tree/main/minillm): Knowledge Distillation of Large Language Models.

We thank the authors for their great work!

---

## 📝 Citation
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{xie2026adakd,
  title={LLM-Oriented Token-Adaptive Knowledge Distillation},
  author={Xie, Xurong and Xue, Zhucun and Wu, Jiafu and Li, Jian and Wang, Yabiao and Hu, Xiaobin and Liu, Yong and Zhang, Jiangning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```
