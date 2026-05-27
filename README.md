# Fish and Shrimp Disease Classification

## Explainable Domain-Separated Deep Learning for Fish and Shrimp Disease Classification Using Vision Transformers and CNN Backbones

This repository contains the Google Colab notebook **`Fish_Shrimp_Classification.ipynb`**, developed for a publication-oriented deep learning workflow on fish and shrimp disease image classification.

Repository:  
[https://github.com/ParthaPRay/Fish_Shrimp_Classification](https://github.com/ParthaPRay/Fish_Shrimp_Classification)

The workflow performs domain-separated image classification for aquaculture disease recognition. Fish disease images and shrimp disease images are treated as separate classification problems so that model learning, evaluation, and interpretation remain biologically and experimentally meaningful.

---

## Overview

Aquaculture disease detection is an important computer vision problem because early identification of visible disease patterns can support monitoring, screening, and decision-support systems in fish and shrimp farming. This project provides a complete deep learning pipeline for disease classification using transfer learning, transformer-based vision models, convolutional neural networks, statistical evaluation, and explainability methods.

The notebook is designed as a research-ready experimental workflow rather than a simple classification script. It includes dataset loading, domain-wise filtering, model training, validation, testing, metric computation, best-model saving, visualization, statistical analysis, and explainable AI outputs.

---

## Dataset

This work uses the **BD Fish & Shrimp Disease Dataset** hosted on Hugging Face:

[https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset](https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset)

The dataset contains labeled aquaculture images covering fish and shrimp disease conditions as well as healthy samples. In this workflow, the original dataset is loaded directly using the Hugging Face `datasets` library. The official dataset split is preserved, and the notebook performs separate filtering for fish and shrimp domains.

The dataset should be cited as:

```bibtex
@dataset{saon110_bd_fish_disease_2025,
  author = {Sijon Chisty Saon},
  title = {BD Fish & Shrimp Disease Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset}
}
````

Users should also review the dataset page for licensing and usage conditions before reuse.

---

## Repository Contents

```text
Fish_Shrimp_Classification/
│
├── Fish_Shrimp_Classification.ipynb
│   └── Main Google Colab notebook for training, evaluation,
│       visualization, statistical analysis, and explainability.
│
└── README.md
    └── Repository documentation.
```

During execution, the notebook creates structured output folders for saved models, result tables, predictions, figures, and explainability outputs.

---

## Main Features

The notebook implements the following components:

* automatic package installation and runtime checking;
* reproducible experiment configuration;
* Hugging Face dataset loading;
* domain-wise filtering into fish and shrimp classification tasks;
* label remapping for domain-specific training;
* transfer learning using pretrained transformer and CNN backbones;
* training with validation monitoring and early stopping;
* best-model saving for each experiment;
* test-set evaluation with multiple classification metrics;
* raw and normalized confusion matrices;
* copy-pasteable result tables;
* training, validation, memory, and CPU/GPU resource plots;
* inference-time measurement;
* bootstrap confidence interval estimation;
* paired statistical testing across models;
* Grad-CAM visualization for CNN models;
* attention-based visualization for transformer models;
* manuscript-ready tables and figures.

---

## Models Used

The workflow evaluates both transformer-based and CNN-based visual backbones.

### Transformer and Hugging Face models

* `google/vit-base-patch16-224`
* `Falconsai/nsfw_image_detection`
* `microsoft/swin-tiny-patch4-window7-224`

### CNN backbones from torchvision

* ResNet-50
* EfficientNet-B0
* MobileNetV3-Large
* DenseNet-121

The `Falconsai/nsfw_image_detection` model is included only as a transfer-learning baseline. It should not be interpreted as an aquaculture-pretrained disease model.

---

## Workflow Summary

The notebook follows a complete research pipeline.

### 1. Environment setup

The workflow begins by checking the runtime, installing required packages, importing libraries, setting reproducibility controls, and creating output directories.

### 2. Dataset loading

The dataset is loaded from Hugging Face using:

```python
from datasets import load_dataset

raw_ds = load_dataset("Saon110/bd-fish-disease-dataset")
```

### 3. Domain separation

The original dataset contains both fish and shrimp categories. The notebook separates them into two independent experimental domains:

* fish disease classification;
* shrimp disease classification.

This avoids mixing biologically distinct visual categories in a single uncontrolled classification task.

### 4. Label remapping

After domain filtering, class labels are remapped into contiguous domain-specific labels. This ensures compatibility with PyTorch loss functions and model heads.

### 5. Data preprocessing

Two dataset wrappers are used:

* one for Hugging Face image processors;
* one for torchvision transforms.

The torchvision pipeline applies resizing, augmentation for training, tensor conversion, and ImageNet normalization.

### 6. Model construction

The notebook defines a unified model builder. Hugging Face models are loaded with a replaced classification head. Torchvision models are loaded with ImageNet pretrained weights and a new domain-specific classifier layer.

### 7. Training

Each model is trained separately for fish and shrimp domains. The training loop includes:

* class-weighted cross-entropy loss;
* AdamW optimization;
* cosine learning-rate scheduling;
* mixed-precision training when available;
* gradient clipping;
* validation monitoring;
* early stopping;
* best-model checkpointing.

### 8. Evaluation

After training, the best saved model is reloaded and evaluated on the test split. The notebook computes a broad set of metrics, including:

* accuracy;
* balanced accuracy;
* precision;
* recall;
* macro-F1;
* weighted-F1;
* Matthews correlation coefficient;
* Cohen’s kappa;
* ROC-AUC;
* PR-AUC;
* log loss;
* Brier score;
* expected calibration error;
* inference speed;
* memory usage.

The repository intentionally does not hard-code results in this README, since results may change with runtime, seed, package version, and experiment configuration.

### 9. Confusion matrix reporting

The workflow generates both raw and normalized confusion matrices. The figures are designed with light colors so that matrix values remain visible. Copy-pasteable tables are also printed and saved.

### 10. Statistical analysis

The notebook includes statistical support for model comparison, including:

* bootstrap confidence intervals;
* McNemar pairwise comparison;
* Cochran’s Q test across classifiers.

These outputs help support manuscript-level claims beyond simple metric reporting.

### 11. Explainability

The workflow includes explainability modules:

* Grad-CAM for CNN models;
* attention heatmap visualization for transformer-based models.

Both correctly classified and misclassified examples can be inspected. The notebook saves original images, heatmaps, overlays, and explainability summary tables.

---

## Output Structure

After running the notebook, outputs are organized under:

```text
aqua_disease_publication_outputs/
│
├── saved_models/
│   └── best saved model checkpoints
│
├── tables/
│   └── CSV and Excel result tables
│
├── figures/
│   └── training curves, comparison plots, confusion matrices
│
├── predictions/
│   └── image-wise prediction tables
│
└── explainability/
    └── Grad-CAM and attention visualization outputs
```

The generated files are suitable for manuscript preparation, supplementary material, and reproducibility documentation.

---

## How to Run

Open the notebook in Google Colab:

```text
Fish_Shrimp_Classification.ipynb
```

Recommended steps:

1. Select a GPU runtime.
2. Run the cells sequentially.
3. If the dataset requires authentication, add a Hugging Face token in the optional login cell.
4. Allow the notebook to complete training and evaluation for both domains.
5. Download the generated output folder or compressed archive.

---

## Requirements

The notebook installs the required packages automatically, including:

```text
datasets
transformers
accelerate
evaluate
timm
torchmetrics
scikit-learn
statsmodels
pynvml
psutil
openpyxl
matplotlib
pandas
numpy
pillow
tqdm
```

The workflow is designed for Google Colab but can be adapted to local GPU environments.

---

## Reproducibility

The notebook includes seed setting for Python, NumPy, and PyTorch. Output directories are created automatically. Model checkpoints, prediction tables, metric summaries, confusion matrices, and explainability figures are saved for later inspection.

For stronger manuscript-level reproducibility, users may rerun the notebook with multiple random seeds and report aggregated statistics.

---

## Suggested Paper Title

**Explainable Domain-Separated Deep Learning for Fish and Shrimp Disease Classification Using Vision Transformers and CNN Backbones**

---

## Possible Applications

This workflow may support research and development in:

* aquaculture disease screening;
* fish and shrimp health monitoring;
* visual disease recognition;
* AI-assisted farm decision support;
* comparative vision model benchmarking;
* explainable AI for agricultural and aquatic image analysis;
* edge and cloud-based aquaculture intelligence systems.

---

## Limitations

This repository provides an image-level classification workflow. It does not perform lesion segmentation, pathogen confirmation, or clinical diagnosis. Model outputs should be interpreted as decision-support results rather than definitive biological diagnosis.

External validation on independent field-acquired datasets is recommended before deployment in real-world aquaculture settings.

---

## Ethical and Responsible Use

The models trained through this workflow should be used responsibly. Automated disease classification should not replace expert veterinary, aquaculture, or laboratory assessment. The workflow is intended for research, education, and decision-support development.

Users should comply with the dataset license and should not use the dataset or trained models beyond the terms permitted by the original dataset provider.

---

## Citation

If you use this repository, notebook, or workflow, please cite the dataset and the author of this workflow.

### Dataset citation

```bibtex
@dataset{saon110_bd_fish_disease_2025,
  author = {Sijon Chisty Saon},
  title = {BD Fish & Shrimp Disease Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/Saon110/bd-fish-disease-dataset}
}
```

### Workflow citation

```bibtex
@misc{ray_fish_shrimp_classification_2026,
  author = {Partha Pratim Ray},
  title = {Fish and Shrimp Disease Classification Using Explainable Domain-Separated Deep Learning},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/ParthaPRay/Fish_Shrimp_Classification},
  note = {Google Colab notebook: Fish_Shrimp_Classification.ipynb}
}
```

---

## Author

**Partha Pratim Ray**
Department of Computer Applications
Sikkim University, India

Email:
`parthapratimray1986@gmail.com`
`ppray@cus.ac.in`

GitHub:
[https://github.com/ParthaPRay](https://github.com/ParthaPRay)

---

## Acknowledgment

The author acknowledges the creator of the BD Fish & Shrimp Disease Dataset and the open-source ecosystem, including Hugging Face, PyTorch, torchvision, scikit-learn, and the broader machine learning research community.

```

This README is aligned with your uploaded notebook title, author information, and workflow structure. :contentReference[oaicite:0]{index=0}
```
