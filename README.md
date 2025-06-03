# 🛳️ Titanic Survival Prediction – ML Pipeline (CatBoost + Random Forest)

This project solves a binary classification task: predicting the survival of Titanic passengers based on features such as age, class, sex, embarkation port, etc. 
It was developed in Google Colab and implements a full ML pipeline: from preprocessing to final Kaggle submission.

---

## 📂 Project Structure

- `notebook.ipynb` – main notebook with full ML pipeline
- `RF_submission.csv` – final predictions ready for Kaggle
- `best_Age_model.joblib` – saved model for age imputation

---

## 🚀 Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- CatBoost (`CatBoostRegressor` for imputing missing `Age`)
- Scikit-learn (`Pipeline`, `RandomForestClassifier`, `OneHotEncoder`, `RandomizedSearchCV`)
- Google Colab + Google Drive

---

## 📊 Pipeline Stages

1. **Data Loading and Cleaning**
   - Fill missing `Fare` with median
   - Fill missing `Embarked` with most frequent value ('S')
   - Extract `Title` from `Name`, map rare titles to `Rare`

2. **Imputing Missing Ages**
   - Use `CatBoostRegressor` to predict missing `Age`
   - Tune parameters via `RandomizedSearchCV`
   - Save and reuse the model to fill missing `Age` in both train and test datasets

3. **Baseline Model**
   - Train `LogisticRegression` on raw numeric features
   - Evaluate via `accuracy`, `f1`, and `roc_auc`

4. **Main Model**
   - Use `RandomForestClassifier` in a pipeline with `StandardScaler` + `OneHotEncoder`
   - Perform hyperparameter tuning (e.g., `n_estimators`, `max_depth`, etc.)
   - Evaluate with `model_eval()` function (cross-validation + ROC curve)

5. **Submission**
   - Save final predictions to `RF_submission.csv`
   - Final Kaggle score: **0.79186**

---

## 📈 Metrics (Example)

| Model               | Accuracy | F1 Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | ~0.66    | —        | —       |
| Random Forest       | ~0.79    | —        | —       |

---

## 🧪 How to Run (Colab Instructions)

1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Ensure `train.csv` and `test.csv` are located at:
   `/content/drive/MyDrive/Colab Notebooks/`
4. Run the notebook cells step by step
5. Download `RF_submission.csv` and upload to Kaggle

---

## 📊 Ideas for Improvement

- Try ensembling (`Voting`, `Stacking`, etc.)
- Engineer more features (`FamilySize`, `Cabin`, `Ticket`, etc.)
- Try other models (e.g., XGBoost, LightGBM)
- Add SHAP/feature importance visualizations
