# Results

This directory contains the evaluation results for the PASCAL VOC object
classification experiments.

Results are organized hierarchically according to:
1. Deep Learning model used as feature extractor
2. Machine Learning classifier
3. CNN feature extraction layer

Each configuration is evaluated independently.

---

## Directory Organization

├── AlexNet/
│ ├── KNN/
│ │ └── fc8/
│ │ ├── roc_per_class.pdf
│ │ ├── auc_per_class.csv
│ │ └── auc_example.png
│ └── SVM/
│ ├── fc7/
│ └── fc8/
├── EfficientNet/
│ ├── KNN/
│ │ └── GlobAvgPool/
│ └── SVM/
│ ├── GlobAvgPool/
│ └── MatMul/
├── MobileNet/
│ ├── KNN/
│ │ └── Pool5/
│ └── SVM/
│ ├── Pooling2D1/
│ └── Logits/
├── ResNet/
│ ├── KNN/
│ │ └── Pooling2D1/
│ └── SVM/
│ ├── Pool5/
│ └── fc1000/


Each layer-specific directory may contain:
- ROC curves per class
- AUC values per class
- Representative AUC visualizations
