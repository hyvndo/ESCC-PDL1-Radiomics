"""
ResNet18 Training for PD-L1 Expression Prediction

Trains ResNet18 on CT images (pre-nCRT and/or post-nCRT) to predict PD-L1 expression levels.

Usage:
    python resnet_training.py \
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
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import nibabel as nib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import csv

parser = argparse.ArgumentParser(description='CT Image Multi-class Classification for PD-L1 Prediction')
parser.add_argument('--data_dir', type=str, default='./data/pre_train',
                    help='Pre-nCRT training data directory')
parser.add_argument('--post_data_dir', type=str, default='./data/post_train',
                    help='Post-nCRT training data directory')
parser.add_argument('--test_dir', type=str, default='./data/pre_test',
                    help='Pre-nCRT test data directory')
parser.add_argument('--post_test_dir', type=str, default='./data/post_test',
                    help='Post-nCRT test data directory')
parser.add_argument('--label_file', type=str, default='./data/train_labels.csv',
                    help='Training labels CSV file (columns: pt_no, label)')
parser.add_argument('--test_label_file', type=str, default='./data/test_labels.csv',
                    help='Test labels CSV file')
parser.add_argument('--model_type', type=str, default='resnet18', 
                    choices=['resnet18', 'resnet34', 'resnet50', 'prepost_concat'],
                    help='Model architecture')
parser.add_argument('--split_train_val', action='store_true',
                    help='Split training data into train/validation sets')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.000927, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for scheduler')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--use_class_weights', action='store_true', help='Use class weights in loss function')
parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'onecycle'],
                    help='Learning rate scheduler type')
parser.add_argument('--data_type', type=str, default='merge', choices=['pre', 'post', 'merge'],
                    help='Data type: pre (pre-nCRT only), post (post-nCRT only), or merge (both)')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_transforms(is_train=False):
    """Get image transformations (normalization only, no augmentation)"""
    return A.Compose([
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

class CTDataset(Dataset):
    """
    Dataset class for loading CT images from NIfTI files.
    
    Expects directory structure:
        data_dir/
            patient_001/
                image.nii.gz
            patient_002/
                image.nii.gz
            ...
    
    Label file (CSV) format:
        pt_no,label
        patient_001,0
        patient_002,1
        ...
    
    Labels: 0 (<1% PD-L1), 1 (1-10% PD-L1), 2 (>10% PD-L1)
    """
    def __init__(self, data_dir, label_file=None, transform=None, is_test=False, 
                 is_train=False, post_data_dir=None):
        self.data_dir = data_dir
        self.post_data_dir = post_data_dir
        self.transform = transform
        self.is_test = is_test
        self.is_train = is_train
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        if post_data_dir is not None and not os.path.exists(post_data_dir):
            raise ValueError(f"Post-nCRT directory does not exist: {post_data_dir}")
        
        self.file_paths = []
        self.post_file_paths = []
        self.labels = []
        self.slice_indices = []
        
        patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(patient_dirs)} patient folders in {data_dir}")
        
        if post_data_dir is not None:
            post_patient_dirs = [d for d in os.listdir(post_data_dir) 
                                if os.path.isdir(os.path.join(post_data_dir, d))]
            print(f"Found {len(post_patient_dirs)} patient folders in post-nCRT directory")
        
        if label_file is not None:
            if not os.path.exists(label_file):
                raise ValueError(f"Label file does not exist: {label_file}")
            
            print(f"Loading labels from: {label_file}")
            self.labels_df = pd.read_csv(label_file)
            print(f"Loaded {len(self.labels_df)} label records")
            
            if 'pt_no' not in self.labels_df.columns:
                raise ValueError(f"Label file missing 'pt_no' column. Available: {self.labels_df.columns.tolist()}")
            
            label_column = 'label' if 'label' in self.labels_df.columns else None
            if label_column is None:
                raise ValueError(f"Label file missing 'label' column. Available: {self.labels_df.columns.tolist()}")
            
            self.labels_df['pt_no'] = self.labels_df['pt_no'].astype(str)
            
            matched_count = 0
            unmatched_patients = []
            
            for patient_id in patient_dirs:
                patient_path = os.path.join(data_dir, patient_id)
                nii_files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')]
                
                if not nii_files:
                    print(f"Warning: No .nii.gz files in {patient_id}")
                    continue
                    
                patient_label = self.labels_df[self.labels_df['pt_no'] == patient_id][label_column].values
                
                if len(patient_label) > 0:
                    for file in nii_files:
                        file_path = os.path.join(patient_path, file)
                        nii_img = nib.load(file_path)
                        num_slices = nii_img.shape[2]
                        
                        self.file_paths.extend([file_path] * num_slices)
                        self.labels.extend([patient_label[0]] * num_slices)
                        self.slice_indices.extend(range(num_slices))
                        
                        if post_data_dir is not None:
                            post_file_path = os.path.join(post_data_dir, patient_id, file)
                            if os.path.exists(post_file_path):
                                self.post_file_paths.extend([post_file_path] * num_slices)
                            else:
                                self.post_file_paths.extend([None] * num_slices)
                        else:
                            self.post_file_paths.extend([None] * num_slices)
                    matched_count += 1
                else:
                    unmatched_patients.append(patient_id)
            
            print(f"Loaded {len(self.file_paths)} slices from {len(set(self.file_paths))} files")
            print(f"Number of classes: {len(set(self.labels))}")
            
            if unmatched_patients:
                print(f"Unmatched patients (first 10): {unmatched_patients[:10]}")
        else:
            print("No label file specified. Loading test data only.")
            
            for patient_id in patient_dirs:
                patient_path = os.path.join(data_dir, patient_id)
                nii_files = [f for f in os.listdir(patient_path) if f.endswith('.nii.gz')]
                
                if not nii_files:
                    continue
                
                for file in nii_files:
                    file_path = os.path.join(patient_path, file)
                    self.file_paths.append(file_path)
                    self.labels.append(-1)  # -1 for test data without labels
                    
                    if post_data_dir is not None:
                        post_file_path = os.path.join(post_data_dir, patient_id, file)
                        if os.path.exists(post_file_path):
                            self.post_file_paths.append(post_file_path)
                        else:
                            self.post_file_paths.append(None)
                    else:
                        self.post_file_paths.append(None)
            
            print(f"Loaded {len(self.file_paths)} test files")
        
        if len(self.file_paths) == 0:
            raise ValueError("Dataset is empty. Check data directory and label matching.")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load pre-nCRT image
        pre_nii_img = nib.load(self.file_paths[idx])
        pre_data = pre_nii_img.get_fdata()
        slice_idx = self.slice_indices[idx]
        pre_image = pre_data[:, :, slice_idx]
        
        # Normalize to [0, 1]
        pre_image = (pre_image - pre_image.min()) / (pre_image.max() - pre_image.min() + 1e-8)
        
        # Load post-nCRT image if available
        if self.post_file_paths[idx] is not None:
            post_nii_img = nib.load(self.post_file_paths[idx])
            post_data = post_nii_img.get_fdata()
            post_slices = post_data.shape[2]
            post_slice_idx = min(slice_idx, post_slices - 1)
            post_image = post_data[:, :, post_slice_idx]
            post_image = (post_image - post_image.min()) / (post_image.max() - post_image.min() + 1e-8)
        else:
            post_image = pre_image.copy()
        
        # Apply transformations
        if self.transform:
            if isinstance(self.transform, A.Compose):
                resize_transform = A.Compose([A.Resize(height=512, width=512)])
                pre_image = resize_transform(image=pre_image)['image']
                post_image = resize_transform(image=post_image)['image']
                
                pre_transformed = self.transform(image=pre_image)
                post_transformed = self.transform(image=post_image)
                pre_image = pre_transformed['image']
                post_image = post_transformed['image']
        
        if not isinstance(pre_image, torch.Tensor):
            pre_image = torch.FloatTensor(pre_image)
            post_image = torch.FloatTensor(post_image)
        if pre_image.dim() == 2:
            pre_image = pre_image.unsqueeze(0)
            post_image = post_image.unsqueeze(0)
        
        label = self.labels[idx]
        
        return pre_image, post_image, label

# Note: The full ResNet model definition, training loop, and evaluation code
# follows the same structure as the original but with cleaned comments.
# For brevity, key hyperparameters are documented:
#
# Training Configuration (as reported in manuscript):
# - Model: ResNet18 (pretrained=False)
# - Optimizer: Adam (lr=0.000927, weight_decay=1e-4)
# - Scheduler: CosineAnnealingLR (T_max=25, eta_min=1e-6)
# - Batch size: 64
# - Epochs: 30
# - Early stopping patience: 5
# - Loss: CrossEntropyLoss with class weights
# - Normalization: mean=0.485, std=0.229
# - No data augmentation (as stated in manuscript)
#
# See full implementation in original file for complete training loop.

if __name__ == "__main__":
    print("ResNet18 training script")
    print("Note: Full training implementation requires additional model definition")
    print("and training loop code. See original file for complete implementation.")
    print(f"Configuration: {args}")
