# Deep Learning for Drug–Target Binding Affinity Prediction

## Abstract
Predicting drug–target binding affinity (DTA) is a crucial task in computational drug discovery, aiming to accelerate lead compound identification and reduce experimental costs. In this work, we present a deep learning pipeline for DTA prediction on the **BindingDB Kd dataset**, leveraging a **CNN-based protein sequence encoder** and **drug molecular fingerprints**. We conduct **ablation studies** (drug-only, protein-only) and benchmark against a **Random Forest baseline**. Results demonstrate that the combined model effectively integrates drug and protein features, achieving competitive predictive performance.

---

## Dataset
- **Source**: [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)  
- **Task**: Binding affinity regression (Kd values, logarithmic scale pKd)  
- **Dataset**: BindingDB_Kd  
- **Final sample sizes (after subsampling for speed):**
  - **Train**: 14,000 pairs
  - **Validation**: 1,999 pairs
  - **Test**: 4,001 pairs

---

## Methods

### 1. Combined CNN Model (Protein + Drug)
- **Drug representation**: Extended-connectivity fingerprints (ECFP, 1024 bits)  
- **Protein representation**: CNN-based encoder over amino acid sequences  
- **Fusion**: Concatenation of drug + protein embeddings  
- **Prediction head**: Fully-connected MLP → regression output (affinity score)

### 2. Ablation Studies
- **Drug-only model**: Fully-connected network on fingerprints  
- **Protein-only model**: CNN encoder without drug features  

### 3. Baseline
- **Random Forest** using concatenated fingerprints and simple sequence features

---

## Results
Evaluation metrics: **MSE, RMSE, R², Pearson correlation, Spearman correlation**  

| Model             | MSE      | RMSE     | MAE    | R²      | Pearson | Spearman |
|------------------|----------|----------|--------|---------|---------|----------|
| **Combined (CNN+DrugFP)** | 0.9791  | 0.9895  | 0.6888 | 0.4873  | 0.7033  | 0.6699   |
| **Drug-only**     | 1.2174  | 1.1034  | –      | 0.3625  | 0.6102  | –        |
| **Protein-only**  | 1.4597  | 1.2082  | –      | 0.2356  | 0.5550  | –        |
| **Random Forest** | **0.8521** | **0.9231** | –      | **0.5538** | **0.7444** | –        |

**Key Findings:**
- Combined CNN model outperforms single-modality models, validating the importance of **joint drug–protein representation learning**.  
- Random Forest achieved the **lowest MSE**, suggesting handcrafted fingerprints and tree ensembles remain strong baselines for DTA tasks.  
- Ablation studies confirm that **drug fingerprints contribute more predictive power** than protein sequences alone, but their integration is necessary for robust performance.

---

## Visualization
**Scatter plot: True pKd vs Predicted pKd**  
<img width="604" height="782" alt="drug-target-cnn-plot" src="https://github.com/user-attachments/assets/2d915213-d0e8-4ec8-b7c6-4c62bb38c82f" />



---

## Reproducibility

### Installation
```bash
pip install pytdc torch torchvision scikit-learn rdkit-pypi
```

### Run
```bash
python train_dti.py
```

### Outputs

- Trained models (.pt)

- Metrics logs (results_summary.csv)

- Example plots

## Conclusion

This project shows that deep learning (CNN + fingerprints) can predict drug–target binding affinities with competitive performance, but traditional ML baselines remain strong. Future work includes:

- Exploring Transformer-based protein embeddings (ProtBERT, ESM2)

- Leveraging Graph Neural Networks (GNNs) for molecular graph encoding

- Scaling training on the full BindingDB dataset
