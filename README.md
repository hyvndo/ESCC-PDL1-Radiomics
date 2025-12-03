# Multivariable Radiomics Model for Predicting PD-L1 Expression in Esophageal Squamous Cell Carcinoma

This repository contains the code implementation for predicting PD-L1 expression levels in esophageal squamous cell carcinoma (ESCC) using radiomics features extracted from CT images after neoadjuvant chemoradiotherapy (nCRT).

## 📄 Paper Information

**Title:** Multivariable Radiomics Model for Predicting Programmed Death-Ligand 1 Expression After Neoadjuvant Chemoradiotherapy in Esophageal Squamous Cell Carcinoma

**Authors:** [Your names and affiliations]

**Journal:** [Journal name - to be added upon acceptance]

## 🎯 Overview

This project implements three different modeling approaches for predicting PD-L1 expression categories (<1%, 1-10%, >10%):

1. **Traditional Radiomics** with machine learning (PyCaret/Logistic Regression)
2. **TabNet** deep learning on tabular features
3. **ResNet18** deep learning on CT images

Our findings demonstrate that traditional radiomics approaches outperform deep learning methods on small medical imaging datasets (n=101).

## 📁 Repository Structure

```
ESCC-PDL1-Radiomics/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── preprocessing/
│   └── featureExtractor.py           # PyRadiomics feature extraction
├── models/
│   ├── resnet_training.py            # ResNet18 training script
│   ├── tabnet_cleaned.ipynb          # TabNet training notebook
│   └── pycaret_pipeline_cleaned.ipynb # PyCaret ML pipeline
├── data/                              # Place your data here (not included)
└── results/                           # Model outputs saved here
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+ (with CUDA for GPU training)
- PyRadiomics
- PyCaret
- TabNet

### Installation

```bash
# Clone the repository
git clone https://github.com/[hyvndo]/ESCC-PDL1-Radiomics.git
cd ESCC-PDL1-Radiomics

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

**⚠️ Important:** This repository does not include patient data. You need to provide your own data in the following format:

#### For PyRadiomics Feature Extraction:
- CT images in NIfTI format (.nii.gz)
- Tumor segmentation masks in NIfTI format
- Organized by patient ID

#### For Model Training:
- Extracted radiomics features (CSV format)
- Clinical data (optional, for combined models)
- Labels file with PD-L1 categories: 0 (<1%), 1 (1-10%), 2 (>10%)

Place your data in the `data/` directory:
```
data/
├── train_features.csv
├── test_features.csv
├── train_labels.csv
├── test_labels.csv
└── clinical_data.xlsx (optional)
```

## 🔬 Usage

### 1. Feature Extraction with PyRadiomics

```python
from preprocessing.featureExtractor import radiomics_extract
import numpy as np

# Load your CT image and mask (as numpy arrays)
image_array = ...  # Your CT image
mask_array = ...   # Your segmentation mask

# Extract features
features, feature_names = radiomics_extract(image_array, mask_array)

print(f"Extracted {len(features)} features")
```

**Configuration:**
- binWidth: 2.8
- interpolator: sitkGaussian
- resampledPixelSpacing: [0, 0, 0] (no resampling)
- force2D: True
- Feature classes: firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
- Image types: Original, LoG, Wavelet

### 2. Training with PyCaret (Recommended)

See `models/pycaret_pipeline_cleaned.ipynb` for the complete workflow:

1. Load radiomics features and clinical data
2. Feature selection using stability selection
3. PyCaret setup and model comparison
4. Train final logistic regression model
5. Evaluate on hold-out test set

**Key Results:**
- AUROC: 0.8214
- Balanced Accuracy: 0.778
- Macro F1-score: 0.776

### 3. Training TabNet

See `models/tabnet_cleaned.ipynb` for TabNet implementation on tabular features.

**Configuration:**
- Learning rate: 0.02
- Batch size: 2048
- Early stopping patience: 20
- Virtual batch size: 512

**Note:** TabNet showed poor generalization on our small dataset (AUROC=0.51).

### 4. Training ResNet18

```bash
python models/resnet_training.py \
    --data_dir ./data/pre_train \
    --post_data_dir ./data/post_train \
    --test_dir ./data/pre_test \
    --post_test_dir ./data/post_test \
    --label_file ./data/train_labels.csv \
    --test_label_file ./data/test_labels.csv \
    --model_type resnet18 \
    --batch_size 64 \
    --num_epochs 30 \
    --learning_rate 0.000927 \
    --output_dir ./results
```

**Configuration:**
- Optimizer: Adam (lr=0.000927)
- Scheduler: CosineAnnealingLR (T_max=25, eta_min=1e-6)
- Early stopping patience: 5
- Dropout: 0.5
- Weight decay: 1e-4
- No data augmentation (normalization only)

**Note:** ResNet18 also showed poor performance on our small dataset (AUROC=0.48).

## 📊 Expected Results

Based on our study with n=101 patients (80 training, 21 test):

| Model | AUROC | Balanced Acc | Macro F1 |
|-------|--------|--------------|----------|
| **Final Combined (Radiomics + Clinical)** | **0.8214** | **0.778** | **0.776** |
| All Radiomics (Pre+Post+Delta) | 0.7008 | 0.648 | 0.622 |
| Clinical Only | 0.8171 | 0.574 | 0.595 |
| Delta-Radiomics Only | 0.7937 | 0.722 | 0.667 |
| Post-Radiomics Only | 0.7286 | 0.593 | 0.590 |
| TabNet | 0.5111 | 0.407 | 0.353 |
| ResNet18 | 0.4778 | 0.287 | 0.137 |

**Key Findings:**
- Traditional radiomics consistently outperforms deep learning on small datasets
- Delta-radiomics (temporal changes) are particularly informative for low PD-L1 (<1%) detection
- Clinical factors alone provide reasonable predictive power
- Deep learning approaches fail to generalize with limited data

## 🔧 Customization

### Modify PyRadiomics Settings

Edit `preprocessing/featureExtractor.py`:

```python
settings = {}
settings['binWidth'] = 2.8  # Adjust discretization bin width
settings['interpolator'] = 'sitkGaussian'  # Change interpolation method
settings['force2D'] = True  # Set to False for 3D features
```

### Adjust Model Hyperparameters

For PyCaret, modify the notebook cell:
```python
pdl1_clf = setup(
    data=train_data,
    target='encoded_PDL1',
    session_id=123,
    # Add your custom settings here
)
```

For ResNet18, use command-line arguments (see `python models/resnet_training.py --help`)

## ⚠️ Important Notes

1. **Small Dataset Performance:** Our results demonstrate that deep learning approaches (ResNet, TabNet) perform poorly on small medical imaging datasets (n=101). Traditional radiomics with machine learning is more appropriate for such scenarios.

2. **Data Privacy:** This repository contains NO patient data. All data paths are placeholders. Users must provide their own data.

3. **Reproducibility:** We provide exact hyperparameters and configurations used in our study to facilitate reproduction.

4. **Hardware Requirements:** 
   - ResNet18 training: GPU recommended (CUDA-capable)
   - TabNet/PyCaret: CPU sufficient

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Multivariable Radiomics Model for Predicting Programmed Death-Ligand 1 Expression After Neoadjuvant Chemoradiotherapy in Esophageal Squamous Cell Carcinoma},
  author={[Your names]},
  journal={[Journal name]},
  year={2024},
  doi={[DOI]}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions about the code or paper:
- GitHub: [@hyvndo](https://github.com/hyvndo)
- Email: [Your email]

## 🙏 Acknowledgments

- PyRadiomics developers for the excellent feature extraction library
- PyCaret team for the automated machine learning framework
- All co-authors and institutions involved in this research

## 📖 Additional Resources

- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/)
- [PyCaret Documentation](https://pycaret.org/)
- [TabNet Paper](https://arxiv.org/abs/1908.07442)
- [Our Paper (link to be added)]

---

**Last Updated:** December 2024
