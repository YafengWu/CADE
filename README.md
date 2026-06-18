## Getting Started

### Environment Requirements

Ensure you have Conda installed, then set up the environment:

```bash
conda create -n cade python=3.10.19
conda activate cade
pip install -r requirements.txt
```

### Data Preparation

The `data/` folder already contains the datasets required for training and testing all models.

### Repository Structure

```
├── CADE/                      # CADE model code (proposed method)
├── CADE_wo_SupCon/            # CADE without SupCon loss (ablation)
├── Frozen_random_linear/      # Frozen Random Linear baseline
├── Frozen_timemoe/            # Frozen Time-MoE baseline
├── ITFormer/                  # ITFormer baseline
├── Time-MQA/                  # Time-MQA (LoRA) baseline
├── Time-MQA_Full_FT/          # Time-MQA (Full Fine-Tuning) baseline
├── data/                      # Training and testing datasets
├── results/                   # Generated CSV result files
├── test/                      # Jupyter notebooks for evaluating results
├── CADE.sh                    # Train & test CADE
├── CADE_wo_SupCon.sh          # Train & test CADE w/o SupCon
├── Frozen_random_linear.sh    # Train & test Frozen Random Linear
├── Frozen_timemoe.sh          # Train & test Frozen Time-MoE
├── ITFormer.sh                # Train & test ITFormer
├── Time-MQA.sh                # Train & test Time-MQA (LoRA)
├── Time-MQA_Full_FT.sh        # Train & test Time-MQA (Full FT)
├── deepseekv3.2.py            # DeepSeek-V3 API-based evaluation
└── requirements.txt           # Python dependencies
```

### Reproducing Results

We provide pre-computed results so you can inspect them without retraining. The `results/` folder contains generated CSV files, and the `test/` folder contains executed Jupyter notebooks that evaluate each model's performance. You can directly open the notebooks in `test/` to check the results.

**To train and test the proposed CADE model:**

```bash
sh CADE.sh
```

**To train and test baseline models**, run the corresponding script:

```bash
sh CADE_wo_SupCon.sh        # CADE without SupCon (ablation)
sh Frozen_random_linear.sh  # Frozen Random Linear
sh Frozen_timemoe.sh        # Frozen Time-MoE
sh ITFormer.sh              # ITFormer
sh Time-MQA.sh              # Time-MQA (LoRA)
sh Time-MQA_Full_FT.sh      # Time-MQA (Full Fine-Tuning)
```

Each script will train and test the corresponding model, and the result CSV files will be saved to the `results/` folder. After training, run the relevant Jupyter notebooks in `test/` to evaluate performance.

**If you want to retrain from scratch**, delete the `results/` folder first, then run the desired shell script(s) and the corresponding notebook(s) in `test/`.

**To test with DeepSeek-V3**, run `deepseekv3.2.py` directly. This script evaluates DeepSeek-V3 on the testing dataset in `data/` via API calls. You will need to add your own API key in the script before running:

```bash
python deepseekv3.2.py
```

## Citation

If you find this repo useful, please cite our paper.

```bibtex
@article{wu2026beyond,
  title={Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering},
  author={Wu, Yafeng and Nguyen, Huu Hiep and Nguyen, Thin and Le, Hung},
  journal={arXiv preprint arXiv:2606.18986},
  year={2026}
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Yafeng Wu (s225635478@deakin.edu.au)
- Huu Hiep Nguyen (s225250685@deakin.edu.au)
- Thin Nguyen (thin.nguyen@deakin.edu.au)
- Hung Le (thai.le@deakin.edu.au)