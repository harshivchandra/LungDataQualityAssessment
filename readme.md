# Lung Data Quality Assessment Tool

This repository provides a common framework for evaluating the quality of lung image datasets. The tool performs multiple assessments and generates performance metrics, including classification, unsupervised Region of Interest (ROI) generation, and segmentation. 

## Citation 

For an indepth understanding of the code, please refer to :

```ref
Rajasekar, Elakkiya, Harshiv Chandra, Nick Pears, Subramaniyaswamy Vairavasundaram,
and Ketan Kotecha. "Lung image quality assessment and diagnosis using generative
autoencoders in unsupervised ensemble learning." Biomedical Signal Processing
and Control 102 (2025): 107268.
```

<a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809424013260">Elsevier Link</a>


## Key Features

- **Classifier**: A Deep CNN model to predict disease from input images.  
  Metrics: F1 Score, Classification Matrix, Precision, Recall, Accuracy, AUC-ROC.
  
- **Segmentation**: Evaluates segmentation quality using an ensemble model (U-Net, U-Net++, Segnet, FCN, NASNet).    
  Metrics: F1 Score, Precision, Recall, DICE.

- **Unsupervised ROI Generation**: Standardised segmentation generator for datasets to understand segmentation performance.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/harshivchandra/LungDataQualityAssessment.git
```

2. Navigate to the git directory and run Jupyter Notebook:

```bash
cd directory_github/LungDataQualityAssessment
jupyter notebook EnsembleLearning.ipynb
```





