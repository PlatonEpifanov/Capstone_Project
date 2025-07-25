# Food101-CNN Project

Convolutional Neural Network for classifying **101 food categories** from the Food-101 / Food41 dataset.

- **Best Top-1 accuracy:** 83.17 %  
- **Best Top-5 accuracy:** 94.98 %  
- **Train/Test split:** 80% / 20% (official Food-101 split)  
- **Backbone:** EfficientNet-B3 (`timm`)  
- **Optimizer:** AdamW, cosine annealing LR schedule  
- **Loss:** CrossEntropy with label smoothing = 0.1  
- **Augmentations:** Resize → CenterCrop → RandAugment → RandomHorizontalFlip → Normalize, RandomErasing (train only)  
- **Frameworks:** PyTorch, timm, sklearn, matplotlib/seaborn  
- **Extras:** Grad‑CAM visualizations & confusion matrix

---

## 1. Motivation & Problem

Recognizing dishes from photos is useful for nutrition tracking, restaurant apps and computer-vision demos. The **goal** was to build an end‑to‑end, reproducible CNN pipeline that reaches strong accuracy on a well-known benchmark and to practice modern DL tooling (timm, AMP, schedulers, Grad‑CAM).

## 2. Dataset

- **Name:** Food-101 (also mirrored as “Food41” on Kaggle)  
- **Kaggle link:** https://www.kaggle.com/datasets/kmader/food41  
- **Size:** 101 classes, 1,000 images each (75k train / 25k test in the official split).

In Colab, the dataset was accessed from the auto-mounted Kaggle path: `/kaggle/input/food41/`.

## 3. Notebook

Full training & analysis is in Google Colab:  
**https://colab.research.google.com/drive/19ehVMaSTRCIs6neUP5jqUClf8BR7-VV0?usp=sharing**

Open it, run all cells (GPU recommended), and you will reproduce the results.

## 4. Quick Start (Colab)

```python
# 0) Optional: install extra packages
!pip -q install timm==0.9.16 torchinfo seaborn scikit-learn

# 1) Set paths
from pathlib import Path
DATA_ROOT = Path('/kaggle/input/food41')
IMG_DIR   = DATA_ROOT/'images'
META_DIR  = DATA_ROOT/'meta'

# 2) Run the rest of the notebook cells (model, training, evaluation)
```

If you trained before and have a checkpoint:
```python
model.load_state_dict(torch.load('/content/best_food101.pth', map_location=device))
```

## 5. Training Details

| Item                | Value                           |
|---------------------|---------------------------------|
| Backbone            | EfficientNet-B3 (pretrained)    |
| Image size          | 224 × 224                       |
| Batch size          | 64 (GPU), 32 if CPU             |
| Epochs              | 12 (7 done first session, 5 resumed) |
| Optimizer           | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler           | CosineAnnealingLR (T_max=epochs)|
| Loss                | CrossEntropy + label smoothing  |
| AMP                 | `torch.cuda.amp` (mixed precision)|
| num_workers         | 2 (set 0 on Mac/CPU if needed)  |

## 6. Results

- **Top‑1 Accuracy:** **83.17 %**  
- **Top‑5 Accuracy:** **94.98 %**  
- Confusion matrix & Grad‑CAM samples are shown in the notebook.

`best_food101.pth` is saved whenever validation accuracy improves.

## 7. Repository Structure

```
.
├── README.md                # this file
├── notebook.ipynb           # (optional) exported Colab
├── presentation.pptx        # project slides
├── best_food101.pth         # trained weights (optional, large file)
└── src/                     # any helper scripts if added later
```

*(Large binaries can go to Releases or Google Drive instead of the repo if they exceed GitHub limits.)*

## 8. Run Locally

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision timm scikit-learn seaborn matplotlib torchinfo
# download & unpack Food-101 so that data/food-101/{images,meta} exist
python train.py            # if you convert notebook to a script
```

## 9. Acknowledgements

- Lukas Bossard et al., **Food-101 – Mining Discriminative Components with Random Forests**, ECCV 2014.  
- Kaggle user **kmader** for hosting the dataset mirror.  
- Ross Wightman for `timm` models.

---

> Questions / suggestions? Open an issue or ping me on GitHub.
