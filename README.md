# Ocular Disease Recognition Using Fundus Images

This repository contains code and resources for predicting ocular diseases using high-resolution fundus images from the ODIR-5k dataset. The project applies machine learning techniques to classify ocular conditions based on patient data and diagnostic keywords.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Ocular diseases can lead to severe vision impairment or blindness if not detected early. Fundus images provide essential information about the eye’s condition, helping in diagnosing diseases. This project focuses on using machine learning to automatically predict eight different ocular conditions from the ODIR-5k dataset. The goal is to build a model that can assist medical professionals in early detection and diagnosis.

### Key Objectives:
1. Perform exploratory data analysis (EDA) to understand the dataset.
2. Preprocess the data to be used for machine learning models.
3. Train machine learning classifiers to predict ocular diseases based on patient information and diagnostic images.
4. Evaluate the model's performance using standard classification metrics.

## Dataset
The [ODIR-5k](https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k) dataset consists of:
- **Patient Information**: Age, sex, etc.
- **Fundus Images**: High-resolution images of the left and right eyes.
- **Diagnostic Keywords**: Conditions identified by specialists.
- **Disease Labels**: Labels for 8 specific ocular diseases:
  - N: Normal
  - D: Diabetes
  - G: Glaucoma
  - C: Cataract
  - A: Age-related Macular Degeneration (AMD)
  - H: Hypertension
  - M: Myopia
  - O: Other conditions

## Installation
### Prerequisites
- Python 3.8 or above
- Jupyter Notebook or Jupyter Lab

### Required Libraries
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow (or pytorch)
```

### Dataset Setup
The dataset can be downloaded from [Kaggle ODIR-5k](https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k). After downloading, place the dataset in the `data/` directory as follows:

```
data/
  └── ODIR-5k/
      ├── Training Set/
      └── Testing Set/
```

## Project Structure
The repository is structured as follows:

```
├── data/                   # Contains the dataset
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code for the project
├── results/                # Contains results such as model metrics and plots
├── models/                 # Saved models
├── README.md               # Project overview and instructions
└── requirements.txt        # List of dependencies
```

## Exploratory Data Analysis
Initial analysis is performed to understand the data. This includes:
- Visualizing the distribution of patient age and sex.
- Exploring diagnostic keywords and disease labels.
- Sample visualizations of fundus images.

The notebook `notebooks/eda.ipynb` contains the code for the EDA.

## Model Training
The machine learning model is built using either **TensorFlow/Keras** or **PyTorch**. Steps include:
1. **Data Preprocessing**: Handling missing data, encoding categorical variables, image preprocessing, etc.
2. **Model Selection**: Training models such as Convolutional Neural Networks (CNNs) for image classification, or traditional ML algorithms like Random Forest and SVM for tabular data.
3. **Training**: The model is trained on the training set and validated on the validation set.

To train the model, run:
```bash
python src/train.py
```

## Evaluation
The model’s performance is evaluated using metrics like:
- Accuracy
- Precision, Recall, and F1 Score
- Confusion Matrix

The evaluation code is available in `notebooks/evaluation.ipynb`. To evaluate the trained model, run:
```bash
python src/evaluate.py
```

## Results
Results include:
- **Accuracy**: XX%
- **Confusion Matrix**: The confusion matrix shows how well the model predicts each disease.
- **Example Predictions**: Example cases where the model successfully or incorrectly classified the disease.

## Contributing
We welcome contributions to improve the model or expand the analysis. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License.
