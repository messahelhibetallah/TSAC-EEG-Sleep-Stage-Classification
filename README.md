# EEG Sleep Stage Classification (Time Series Project)

> Automatic sleep stage classification from single-channel EEG signals using Time Series Analysis & Machine Learning

---

## Project Overview

This project was developed for the **Time Series Analysis & Classification (TSAC)** course. We analyze EEG time series data to automatically classify sleep stages, replacing manual scoring by experts.

**Problem:** Manual sleep staging is time-consuming, expensive, and subjective.

**Solution:** Build ML models that learn time series patterns in EEG signals to classify 5 sleep stages.

### Sleep Stages

| Stage | Name | EEG Characteristics |
|-------|------|-------------------|
| 0 | Wake | Low amplitude, mixed frequencies |
| 1 | NREM 1 | Theta waves (4-8 Hz) |
| 2 | NREM 2 | Sleep spindles, K-complexes |
| 3 | NREM 3 | High amplitude delta waves (<4 Hz) |
| 4 | REM | Mixed frequency, rapid eye movements |

---

## Team Members

- **Messahel Hibetallah**
- **BOUDJELAL Mounir**
- **GHENAM Douaa**
- **BELKEBIR Mustapha**

---

## Dataset

- **Training:** 4,999 EEG epochs × 178 time steps
- **Test:** 999 EEG epochs × 178 time steps  
- **Sampling Rate:** 100 Hz
- **Epoch Duration:** 30 seconds

---

##  Methodology

### Time Series Preprocessing
```python
# Z-score normalization per epoch
def normalize_epoch(x):
    return (x - x.mean()) / x.std()
```

### Feature Extraction from Time Series

| Method | Description |
|--------|-------------|
| **Raw Signal** | Direct time series values |
| **Spectrogram** | Time-frequency representation (STFT) |
| **Wavelet Features** | Multi-resolution decomposition features |
| **PCA** | Dimensionality reduction (95% variance) |

### Models Tested

| Model | Type | Best Accuracy |
|-------|------|---------------|
| KNN | Distance-based | 58% |
| SVM | Kernel-based | 63% |
| Random Forest | Tree-based | 63% |
| CNN-LSTM | Deep Learning | 66% |
| EEGNet | Deep Learning | 65% |
| **XGBoost** | **Boosting** | **75%** 🏆 |

---

## Best Model: XGBoost

**Why it worked best for time series data:**
- Handles temporal patterns effectively
- Works well with engineered time-frequency features
- Robust to class imbalance
- Fast training on tabular features

### Performance Details

```
Classification Report:
              precision    recall  f1-score   support
Wake             0.93      0.82      0.87       339
NREM 1           0.82      0.63      0.71       509
NREM 2           0.71      0.86      0.78      1313
NREM 3           0.65      0.62      0.63       661
REM              0.63      0.50      0.56       378

Average Accuracy: 75%
```

---

##  Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost tensorflow pywavelets matplotlib seaborn
```

### 2. Run the Notebook
Open `The full processing.ipynb` and run all cells

### 3. Make Predictions
```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('xgboost_model.pkl')
pca = joblib.load('pca_transformer.pkl')

# Load your EEG time series (178 time points per epoch)
X_new = load_eeg_data()  # Shape: (n_epochs, 178)

# Normalize
X_new = (X_new - X_new.mean(axis=1, keepdims=True)) / X_new.std(axis=1, keepdims=True)

# Apply PCA
X_new_pca = pca.transform(X_new)

# Predict sleep stages
predictions = model.predict(X_new_pca)
print(f"Sleep stages: {predictions}")
```

---

## Key Findings

### Confusion Matrix (XGBoost)
```
         Wake  NREM1  NREM2  NREM3   REM
Wake      82%     8%     9%     0%    1%
NREM1      7%    63%    27%     1%    2%
NREM2      3%     4%    86%     7%    0%
NREM3      0%     0%    35%    62%    3%
REM        1%     7%    37%    13%   42%
```

### What Makes This Hard
- **Similar patterns:** NREM1 and REM look alike in EEG
- **Individual differences:** Brain waves vary per person
- **Class imbalance:** Some stages are more common

---

## Key Insights from Time Series Analysis

### Spectral Analysis by Stage
| Stage | Spectral Entropy | Peak Frequency |
|-------|------------------|----------------|
| Wake | 4.30 bits | 0.56 Hz |
| NREM 1 | 3.97 bits | 0.56 Hz |
| NREM 2 | 3.64 bits | 1.12 Hz |
| NREM 3 | 4.10 bits | 1.69 Hz |
| REM | 3.22 bits | 1.12 Hz |

### Transition Patterns (Hypnogram Analysis)
- **Wake → Wake:** 53% (most stable)
- **NREM 1 → Wake:** 39% (fragmentation)
- **NREM 3 → NREM 1:** 100% (deep to light sleep)

**Average bout durations:** 1-2 minutes (suggests sleep fragmentation)

---

## Future Improvements

- Add multi-channel EEG (more data)
- Try Transformer architectures for time series
- Incorporate additional signals (EMG, EOG, ECG)
- Build real-time mobile application
- Use more subjects for generalization

---

*Made with ❤️ for the TSAC Course*

---

**⭐ Star this repo if you found it useful!**
