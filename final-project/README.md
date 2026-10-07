# DATAX504 Final Project
## Smartphone-Based Human Activity Recognition Using Deep Learning

### Student
Neha Singh

### Course
DATAX504 final project

## Project Overview

This project develops and evaluates a deep learning solution for smartphone-based human activity recognition.
The task is a six-class classification problem using the UCI Human Activity Recognition Using Smartphones dataset. The model predicts one of the following activities:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

The project follows the Chapter 6 deep learning workflow used in the course: problem framing, data preparation, baseline modelling, neural-network development, evaluation, error analysis, and discussion of limitations.

## Research Question

How effectively can a deep neural network classify six human activities from smartphone sensor-derived features, and how does its performance compare with simpler baseline models?

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones dataset**.

The dataset contains:

- 10,299 observations
- 30 participants
- 561 engineered sensor-derived features
- 6 activity classes
- accelerometer and gyroscope measurements
- separate official training and test partitions

The prepared features include time-domain and frequency-domain measurements derived from smartphone inertial sensor data.

### Data Split Strategy

The official UCI train/test split was retained.

The official training partition was then further divided into:

- development training participants
- validation participants

The split was performed at the participant level rather than randomly at the observation level.

This prevents observations from the same participant appearing in both the training and validation sets and reduces the risk of participant leakage.

The official test set was kept separate until final model evaluation.

## Project Structure

```text
final-project/
├── README.md
├── requirements.txt
├── data/
│   └── UCI HAR Dataset/
├── models/
│   └── regularized_har_model.keras
├── notebooks/
│   └── HAR_Final_Project.ipynb
├── results/
│   ├── final_model_comparison.csv
│   ├── final_test_results.csv
│   ├── largest_confusion_pairs.csv
│   ├── regularized_mlp_class_metrics.csv
│   ├── regularized_mlp_test_confusion_matrix.png
│   └── top_confident_errors.csv
└── src/

##Models Compared
The project compares several models of increasing complexity.
1. Majority-Class Baseline
A naive classifier that always predicts the most frequent training activity.
2. Logistic Regression
A strong classical linear baseline using all 561 standardised features.
3. Keras Linear Softmax Model
A TensorFlow/Keras model containing only a six-class Softmax output layer.
4. Small MLP
A neural network with one hidden Dense layer containing 64 ReLU units.
5. Deep MLP
An unregularised neural network with hidden layers of: 256 → 128 → 64

6. Regularised Deep MLP
The final deep learning model uses: Input: 561 features
↓
Dense 256 + ReLU + L2
↓
Dropout
↓
Dense 128 + ReLU + L2
↓
Dropout
↓
Dense 64 + ReLU + L2
↓
Dense 6 + Softmax
Regularisation includes:
- dropout
- L2 weight regularisation
- early stopping
- learning-rate reduction

## Preprocessing
The 561 numerical features were standardised using StandardScaler.
The scaler was fitted only on the development training set.
The same fitted transformation was then applied to:
- validation data
- official test data
This avoids preprocessing leakage from the validation or test sets.
Labels were converted from the original UCI representation: 1–6 to 0–5 for compatibility with Keras sparse_categorical_crossentropy.

## Reproducibility
A fixed random seed was used for Python, NumPy, and TensorFlow.
SEED = 42
The project records the versions of the main libraries used during execution.
The notebook is intended to be run from top to bottom in sequence.

## Environment Setup
Create and activate a Python virtual environment.
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

Install the required packages:
pip install -r requirements.txt

## Running the Project
Open the notebook:
notebooks/HAR_Final_Project.ipynb

Run all cells from top to bottom.
The notebook performs:
1. dataset loading
2. integrity checks
3. exploratory data analysis
4. participant-safe splitting
5. preprocessing
6. baseline training
7. neural-network training
8. model comparison
9. final test evaluation
10. confusion-matrix analysis
11. confidence analysis
12. result export

## Validation Results
| Model | Validation Accuracy |
|---|---:|
| Majority Class Baseline | 19.39% |
| Logistic Regression | 95.61% |
| Keras Linear Softmax | 94.80% |
| Small MLP (64 ReLU) | 94.59% |
| Deep MLP (256-128-64) | 93.72% |
| Regularised Deep MLP | 95.20% |
The strongest validation model was Logistic Regression.
The unregularised deep network performed worse than the simpler models, while regularisation improved the deep model substantially.

## Final Test Results
Only the selected Logistic Regression baseline and Regularised Deep MLP were evaluated on the official held-out test set.
| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 94.84% |
| Regularised Deep MLP | **95.08%** |
The Regularised Deep MLP achieved the strongest final test accuracy.
The performance difference was small, showing that the engineered UCI HAR features are already highly effective for linear classification.

## Regularisation Ablation
The unregularised Deep MLP achieved: 93.72% validation accuracy

The Regularised Deep MLP achieved: 95.20% validation accuracy

This is an improvement of approximately: 1.48 percentage points

The experiment suggests that additional model capacity alone did not improve generalisation, while dropout and L2 regularisation helped control overfitting.

## Error Analysis
The final Regularised Deep MLP achieved strong overall performance, but errors were concentrated among similar activities.
The largest confusion patterns included:
- SITTING → STANDING
- STANDING → SITTING
- WALKING_DOWNSTAIRS → WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS → WALKING
LAYING was classified perfectly on the official test set.
The weakest class by F1-score was: SITTING
with an F1-score of approximately: 0.92
The strongest class was: LAYING
with an F1-score of: 1.00

## Confidence Analysis
The Regularised Deep MLP produced:
Mean confidence for correct predictions:   0.9804
Mean confidence for incorrect predictions: 0.8388

Median confidence for correct predictions:   1.0000
Median confidence for incorrect predictions: 0.8909

There were also:
50 incorrect predictions with confidence above 0.95

This shows that incorrect predictions generally had lower confidence, but the model could still be confidently wrong.
Softmax confidence therefore should not be interpreted directly as a calibrated probability of correctness.

## Main Findings
The main findings from the project are:
- all learned models substantially outperformed the majority-class baseline
- Logistic Regression provided a very strong classical baseline
- increasing neural-network depth alone did not improve performance
- the unregularised deep model showed poorer generalisation
- dropout and L2 regularisation substantially improved the deep model
- the final Regularised Deep MLP achieved the highest held-out test accuracy
- most errors occurred between physically similar activities
- simple linear models remain highly competitive when strong engineered features are available

## Limitations
The project has several limitations:
- only 30 participants are included in the dataset
- the data were collected in a controlled experimental setting
- the project uses engineered features rather than raw sensor sequences
- participant characteristics may not represent broader populations
- similar activities remain difficult to distinguish
- Softmax confidence is not calibrated
- the project does not evaluate real-time smartphone deployment

## Future Work
Possible extensions include:
- training directly from raw accelerometer and gyroscope sequences
- using 1D convolutional neural networks
- experimenting with recurrent neural networks
- using subject-wise cross-validation
- testing additional regularisation strategies
- investigating probability calibration
- evaluating more diverse real-world participants
- comparing engineered-feature models with end-to-end raw-signal deep learning

AI Tool Disclosure
ChatGPT was used as a supporting tool during the project, mainly for troubleshooting Python/Keras errors, clarifying TensorFlow and machine-learning concepts, reviewing code structure, and suggesting ways to improve the presentation of results.
AI suggestions were not accepted automatically. Code was run and checked in the local development environment, outputs were verified against the dataset, and model performance was evaluated using the actual validation and test results produced during execution.
The final experimental decisions, model comparisons, interpretation of results, error analysis, conclusions, and project submission were reviewed and understood by the author.
The author remains responsible for the accuracy of the submitted code, analysis, and written work.

## References
- UCI Machine Learning Repository. Human Activity Recognition Using Smartphones Dataset.
- TensorFlow / Keras documentation.
- Scikit-learn documentation.
- Chollet, F. Deep Learning with Python, 3rd edition.