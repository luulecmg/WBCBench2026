# Decoupled Learning for White Blood Cell Classification

## Method Description

This work proposes a **two-stage decoupled learning strategy** combined with **ensemble inference** to address class imbalance in white blood cell (WBC) classification.

---

## Stage 1: Backbone Training

- Train the **entire network** (backbone + classifier head)
- Use **standard Cross-Entropy Loss**
- Apply **random sampling** to learn general feature representations
- Use **early stopping** with patience-based monitoring

---

## Stage 2: Classifier Fine-tuning

- **Freeze the backbone**
- Reinitialize and train **only the classifier head**
- Use **class-balanced sampling** to mitigate class imbalance
- Optimize with **Combined Loss**:
  - Cross-Entropy Loss  
  - Focal Loss  
  - Effective number–based class weighting

---

## Ensemble & Inference

- Train two backbone architectures:
  - **ResNet50** 
  - **ResNet152**
- Apply **Macenko stain normalization** as preprocessing
- Perform **8× Test-Time Augmentation (TTA)**:
  - Original image (×2, higher weight)
  - Horizontal / vertical flips
  - Rotations: 90°, 180°, 270°
- Ensemble predictions using **weighted probability averaging**
- Apply **post-processing adjustment** for rare classes based on the training class distribution

# Install
### Step 1: Create Conda Environment
```bash
conda create -n wbc python=3.8 -y
```
### Step 2: Activate Environment
```bash
conda activate wbc
```
### Step 3: Install Dependencies
```bash
pip install pip install -r requirements.txt
```

# Commands
## Training
```bash
jupyter notebook training.ipynb
# Run all cells sequentially
```

## Inference
```bash
jupyter notebook inference.ipynb
# Run all cells sequentially

# Required checkpoints:
#   - checkpoints/final_model_resnet50.pt
#   - checkpoints/final_model_resnet152.pt
```

# Compute Environment

All experiments were conducted on **Kaggle** with the following hardware configuration:

- **GPU**: 2 × NVIDIA Tesla T4 (16 GB VRAM each)
- **CUDA**: 12.6
- **cuDNN**: 9.1
- **Framework**: PyTorch 2.8.0
- **Operating System**: Linux (Kaggle notebook environment)

### Training
- Models were trained using **dual-GPU execution (2× Tesla T4)**
- Mixed precision training (AMP) was enabled
- Deterministic settings and fixed random seeds were applied for reproducibility

### Inference
- Inference was performed on **Tesla T4 GPUs**
- Ensemble inference and Test-Time Augmentation (TTA) were executed sequentially
- No multi-GPU parallelism was used during inference to ensure stable and reproducible predictions


# Seed Settings & Determinism
```bash
SEED = 42

# Applied to:
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic settings:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DataLoader worker seeding via seed_worker() and Generator
```