# Model Comparison and Evaluation Report

## 1. Performance Overview

| Model               |   Accuracy |   Precision |   Recall |     F1 |   CV F1 |
|:--------------------|-----------:|------------:|---------:|-------:|--------:|
| KNN                 |     0.6875 |      0.7164 |   0.8889 | 0.7934 |  0.7996 |
| Random Forest       |     0.6875 |      0.7458 |   0.8148 | 0.7788 |  0.7957 |
| Logistic Regression |     0.7375 |      0.8    |   0.8148 | 0.8073 |  0.7674 |
| SVM                 |     0.7    |      0.7885 |   0.7593 | 0.7736 |  0.7506 |
| Decision Tree       |     0.675  |      0.7917 |   0.7037 | 0.7451 |  0.6835 |

## 2. Selected Best Model: **KNN**
- **Selection Criterion:** Highest 5-Fold Stratified Cross-Validation F1-score on training set
- **Test Set Accuracy:** 68.8%
- **Test Set F1-Score:** 0.793
- **Cross-Validated F1:** 0.800

## 3. Classification Report (Test Set)
```
              precision    recall  f1-score   support

  Not placed       0.54      0.27      0.36        26
      Placed       0.72      0.89      0.79        54

    accuracy                           0.69        80
   macro avg       0.63      0.58      0.58        80
weighted avg       0.66      0.69      0.65        80

```

## 4. Key Factors Influencing Placement

| Feature       |   Importance |
|:--------------|-------------:|
| degree_p      |       0.2261 |
| ssc_p         |       0.1941 |
| hsc_p         |       0.1551 |
| mba_p         |       0.1067 |
| etest_p       |       0.0872 |
| hsc_b_Central |       0.0211 |
| workex_No     |       0.0203 |
| workex_Yes    |       0.0187 |
| hsc_b_Others  |       0.0165 |
| gender_M      |       0.0154 |

## 5. Artifacts Generated
- `models/placement_model.joblib`: Serialized inference pipeline
- `models/metadata.json`: Feature schemas and benchmark metrics
- `reports/results.csv`: Complete metrics comparison
- `reports/feature_importance.csv`: Ranked predictor importance
- `reports/figures/confusion_matrix.png`: Test set confusion matrix
- `reports/figures/model_comparison.png`: Accuracy vs CV F1 chart
- `reports/figures/feature_importance.png`: Top predictors bar chart