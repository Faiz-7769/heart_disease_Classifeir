# 🫀 Heart Disease Prediction with Decision Tree and Random Forest

This project uses supervised machine learning models to predict the presence of heart disease using various medical attributes. The goal is to explore the effectiveness of **Decision Tree** and **Random Forest** classifiers, along with data visualization and performance evaluation techniques.

---

## 📁 Dataset

- Source: UCI Heart Disease Dataset
- Target: `target` (1 = disease, 0 = no disease)
- Features include: age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, etc.

---

## 📊 1. Exploratory Data Analysis (EDA)

### 📌 Visualizations Performed:
- **Seaborn Pairplot**:
  - Visualizes feature relationships using `hue='thal'` to detect class separability.
- **Heatmap**:
  - Correlation heatmap of numerical features using `seaborn.heatmap()`.
- **Matplotlib Bar Charts**:
  - Feature importance visualization from the Random Forest model.

---

## 🧹 2. Preprocessing

- Used **`StandardScaler`** from `sklearn.preprocessing` to standardize features like:
  - `trestbps`, `chol`, `thalach`
- Split dataset into training and testing sets using `train_test_split`.

---

## 🌳 3. Modeling

### ✅ Decision Tree Classifier
- Trained using `sklearn.tree.DecisionTreeClassifier`
- Visualized using `plot_tree()` from `sklearn.tree`
- Used `random_state=42` for reproducibility

### 🌲 Random Forest Classifier
- Trained using `sklearn.ensemble.RandomForestClassifier`
- Feature importances extracted and visualized using **bar charts**

---

## 📈 4. Evaluation Metrics

### 🧾 Confusion Matrix
- Plotted using `sklearn.metrics.ConfusionMatrixDisplay`
- Compared predictions of both models

### 📃 Classification Report
- Included precision, recall, F1-score for both models

### 📉 Accuracy vs. Depth Plot
- Plotted accuracy percentages for varying `max_depth` values in Decision Tree
- Helped visualize underfitting vs overfitting

---

## 📌 Key Libraries Used

- `matplotlib.pyplot`
- `seaborn`
- `pandas`, `numpy`
- `sklearn`:
  - `StandardScaler`
  - `DecisionTreeClassifier`, `RandomForestClassifier`
  - `ConfusionMatrixDisplay`, `classification_report`
  - `plot_tree`

---

## 🧠 Insights

- Random Forest provided better generalization and performance compared to Decision Tree.
- Important features identified included `cp`, `thal`, `oldpeak`, and `thalach`.
- Depth tuning significantly affected decision tree performance.


