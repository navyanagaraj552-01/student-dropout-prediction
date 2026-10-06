# Student Academic Outcome Prediction

## Dataset: UCI 697 - 4424 students
Target: Dropout / Enrolled / Graduate

## Leakage Handling (Imp)
Dropped all 1st & 2nd sem curricular units as per core expectation.
Used only enrollment-time features for leakage-safe baseline.

## Models
1. Baseline: Logistic Regression
2. Advanced: RandomForest (150 trees)

## Evaluation
- Accuracy, Precision, Recall, F1 per class
- Macro F1 (main metric for imbalanced 3-class)
- Confusion Matrix

## How to Run
pip install -r requirements.txt
python main.py