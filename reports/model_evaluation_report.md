# Model Comparison and Comprehensive Evaluation Report

## 1. Benchmarking Overview across 6 Algorithms

| Model               |   Accuracy |   Balanced Acc |   Macro F1 |   At-Risk Recall |   Placed Recall |   ROC-AUC |   CV BalAcc |   Optimal Threshold |
|:--------------------|-----------:|---------------:|-----------:|-----------------:|----------------:|----------:|------------:|--------------------:|
| Logistic Regression |     0.7125 |         0.6774 |     0.6755 |           0.5769 |          0.7778 |    0.745  |      0.7356 |                0.69 |
| SVM                 |     0.725  |         0.6766 |     0.68   |           0.5385 |          0.8148 |    0.7457 |      0.7218 |                0.66 |
| Random Forest       |     0.6375 |         0.6118 |     0.6048 |           0.5385 |          0.6852 |    0.7009 |      0.705  |                0.7  |
| XGBoost             |     0.6375 |         0.6019 |     0.5983 |           0.5    |          0.7037 |    0.6603 |      0.6857 |                0.64 |
| KNN                 |     0.65   |         0.5912 |     0.5927 |           0.4231 |          0.7593 |    0.6531 |      0.6556 |                0.67 |
| Gradient Boosting   |     0.65   |         0.641  |     0.6267 |           0.6154 |          0.6667 |    0.6781 |      0.6302 |                0.7  |

## 2. Champion Model Selected: **Logistic Regression**
- **Selection Criterion:** Highest 5-Fold Stratified Cross-Validation Balanced Accuracy on training data (addressing class imbalance).
- **Optimal Decision Threshold:** `0.69` (Tuned on CV out-of-fold probability distribution)
- **Test Accuracy:** `71.2%`
- **Test Balanced Accuracy:** `67.7%`
- **Test Macro F1:** `0.6755`
- **At-Risk Student Recall:** `57.7%`
- **Placed Student Recall:** `77.8%`
- **ROC-AUC:** `0.7450`
- **5-Fold CV Balanced Accuracy:** `0.7356`

## 3. Top Influencing Factors (Feature Importance)

| Feature        |   Importance |
|:---------------|-------------:|
| degree_p       |       0.808  |
| ssc_p          |       0.7224 |
| hsc_p          |       0.5778 |
| workex_Yes     |       0.3003 |
| workex_No      |       0.3003 |
| hsc_b_Others   |       0.2404 |
| hsc_b_Central  |       0.2403 |
| mba_p          |       0.2368 |
| hsc_s_Commerce |       0.1728 |
| etest_p        |       0.1674 |

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