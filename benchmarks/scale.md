# The full workflow at scale

Five-fold cross-validated scores of the model the framework selected. Files above the size threshold are read by Apache Spark; in-memory stages use a bounded sample.

| Dataset | Rows | Engine | Target (correct?) | Best model | Accuracy | Macro F1 | ROC-AUC | Baseline | Full data, Spark MLlib | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| adult_income | 32,561 (32,537 modelled) | pandas | income (yes) | XGBoost | 0.8729 | 0.8158 | 0.9285 | 0.5047 | – | 335.2 |
| bank_marketing | 45,211 | pandas | subscribed (yes) | LightGBM | 0.91 | 0.7587 | 0.9376 | 0.4953 | – | 278.6 |
| telco_churn | 5,000 | pandas | CustomerChurned (yes) | LightGBM | 0.9618 | 0.9148 | 0.9236 | 0.5023 | – | 192.2 |
| miniboone | 130,064 (49,831 modelled) | spark | signal (yes) | LightGBM | 0.9385 | 0.9288 | 0.9828 | 0.5091 | GBTClassifier: acc 0.9275, F1 0.9093 (104,325 train rows) | 832.0 |
| covertype | 581,012 (50,000 modelled) | spark | Cover_Type (yes) | Extra Trees | 0.8927 | 0.8411 | – | 0.1454 | RandomForestClassifier: acc 0.7886, F1 0.6526 (465,120 train rows) | 855.9 |
