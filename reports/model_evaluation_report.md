# Model Comparison and Comprehensive Evaluation Report

## 1. Benchmarking Overview across 6 Algorithms

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   CV F1 |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|--------:|
| KNN                 |     0.7    |      0.7143 |   0.9259 | 0.8065 |    0.6531 |  0.8065 |
| Random Forest       |     0.75   |      0.75   |   0.9444 | 0.8361 |    0.683  |  0.7957 |
| XGBoost             |     0.7375 |      0.7463 |   0.9259 | 0.8264 |    0.6781 |  0.7878 |
| Logistic Regression |     0.725  |      0.7667 |   0.8519 | 0.807  |    0.745  |  0.7816 |
| Gradient Boosting   |     0.75   |      0.7429 |   0.963  | 0.8387 |    0.6766 |  0.7792 |
| SVM                 |     0.7625 |      0.7778 |   0.9074 | 0.8376 |    0.7457 |  0.7554 |

## 2. Champion Model Selected: **KNN**
- **Selection Criterion:** Highest 5-Fold Stratified Cross-Validation F1 score on training data.
- **Optimal Decision Threshold:** `0.44` (Standard: `0.50`)
- **Test Accuracy:** `70.0%`
- **Test F1-Score:** `0.8065`
- **ROC-AUC:** `0.6531`
- **5-Fold CV F1:** `0.8065`

## 3. Top Influencing Factors (Feature Importance)

N/A

## 4. Generated Artifacts
- `models/placement_model.joblib`: Serialized inference pipeline
- `models/all_models.joblib`: Serialized dictionary of all 6 trained algorithms
- `models/metadata.json`: Feature schemas, optimal threshold, and benchmark metrics
- `reports/results.csv`: Complete metrics comparison
- `reports/feature_importance.csv`: Ranked predictor importance
- `reports/figures/model_comparison.png`: Accuracy, F1, and CV F1 comparison chart
- `reports/figures/confusion_matrix.png`: Test set confusion matrix
- `reports/figures/roc_curve.png`: ROC Curve for champion model
- `reports/figures/feature_importance.png`: Top predictors bar chart