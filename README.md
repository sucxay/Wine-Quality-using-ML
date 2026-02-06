# Wine Quality Machine Learning Analysis

## 📊 Dataset Overview

- **Total Samples**: 1,699 wines
- **Features**: 11 physicochemical properties
- **Target Variable**: Wine quality (scale 3-8)
- **Classification Task**: Predicting good quality (≥6) vs bad quality (<6)

## 🔍 Key Findings

### Data Distribution
- **Good Quality Wines**: 45.6% (≥6 rating)
- **Bad Quality Wines**: 54.4% (<6 rating)
- **Mean Quality Score**: 5.64
- **Quality Range**: 3 to 8

### Most Important Features for Quality Prediction

1. **Alcohol-Density Ratio** (13.0% importance)
   - Engineered feature combining alcohol and density
   - Strong indicator of wine quality

2. **Alcohol Content** (11.2% importance)
   - Higher alcohol → Better quality
   - Average: 10.4%

3. **Sulphates** (10.3% importance)
   - Positive correlation with quality
   - Acts as antioxidant and antimicrobial

4. **Volatile Acidity** (8.6% importance)
   - Negative correlation with quality
   - High levels indicate vinegar taste

5. **Total Sulfur Dioxide** (7.0% importance)
   - Moderate levels preferred

### Correlation Insights

**Positive Correlations with Quality:**
- Alcohol: +0.45
- Sulphates: +0.24
- Citric Acid: +0.22

**Negative Correlations with Quality:**
- Volatile Acidity: -0.38
- Density: -0.18
- Total Sulfur Dioxide: -0.17

## 🤖 Machine Learning Models Performance

### Model Comparison

| Model | Accuracy | F1-Score | CV Mean | CV Std |
|-------|----------|----------|---------|--------|
| **Random Forest** | 78.2% | 79.1% | 77.6% | 3.7% |
| Gradient Boosting | 74.1% | 73.7% | 75.9% | 1.2% |
| Decision Tree | 73.2% | 74.8% | 70.5% | 2.6% |
| SVM | 71.8% | 71.3% | 75.6% | 1.5% |
| Logistic Regression | 70.6% | 70.8% | 73.1% | 0.9% |

### Best Model: Tuned Random Forest

**Hyperparameters:**
- n_estimators: 300
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 1

**Performance:**
- Test Accuracy: **77.1%**
- Test F1-Score: **78.0%**
- Cross-validation Score: **79.1%**

**Classification Metrics:**
```
              Precision  Recall  F1-Score  Support
Bad Quality      0.74     0.77     0.75      162
Good Quality     0.78     0.75     0.77      178
```

## 💡 Feature Engineering

Created 4 new features to improve model performance:

1. **total_acidity**: fixed acidity + volatile acidity
2. **free_sulfur_ratio**: free SO₂ / total SO₂
3. **alcohol_density_ratio**: alcohol / density (most important!)
4. **acidity_sugar_ratio**: total acidity / residual sugar

These engineered features improved model accuracy by ~3-5%.

## 📈 What Makes Good Wine?

Based on our analysis, good quality wines typically have:

✅ **Higher alcohol content** (10-13%)
✅ **Lower volatile acidity** (<0.5)
✅ **Higher sulphates** (>0.6)
✅ **Moderate citric acid** (0.3-0.5)
✅ **Lower density** (~0.995-0.997)
✅ **Optimal pH** (3.0-3.4)

❌ Avoid:
- High volatile acidity (>0.7)
- Very low alcohol (<9%)
- Excessive sulfur dioxide (>200)

## 🚀 Usage

### Running the Analysis

```bash
python wine_quality_analysis.py
```

This will:
- Load and explore the dataset
- Perform correlation analysis
- Train 5 different ML models
- Tune the best model (Random Forest)
- Generate visualizations
- Save results

### Making Predictions

```bash
python wine_predictor.py
```

Or use in your own code:

```python
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open('wine_quality_model.pkl', 'rb'))
scaler = pickle.load(open('wine_quality_scaler.pkl', 'rb'))

# Prepare your wine data
wine = {
    'fixed acidity': 7.5,
    'volatile acidity': 0.5,
    'citric acid': 0.36,
    'residual sugar': 6.1,
    'chlorides': 0.071,
    'free sulfur dioxide': 17.0,
    'total sulfur dioxide': 102.0,
    'density': 0.9978,
    'pH': 3.35,
    'sulphates': 0.80,
    'alcohol': 10.5
}

# Add engineered features
wine['total_acidity'] = wine['fixed acidity'] + wine['volatile acidity']
wine['free_sulfur_ratio'] = wine['free sulfur dioxide'] / (wine['total sulfur dioxide'] + 1)
wine['alcohol_density_ratio'] = wine['alcohol'] / wine['density']
wine['acidity_sugar_ratio'] = wine['total_acidity'] / (wine['residual sugar'] + 1)

# Predict
wine_df = pd.DataFrame([wine])
wine_scaled = scaler.transform(wine_df)
prediction = model.predict(wine_scaled)
probability = model.predict_proba(wine_scaled)

print(f"Quality: {'Good' if prediction[0] == 1 else 'Bad'}")
print(f"Confidence: {probability[0][prediction[0]] * 100:.1f}%")
```

## 📁 Output Files

1. **eda_analysis.png** - Exploratory data visualizations
   - Correlation heatmap
   - Quality distribution
   - Feature distributions
   - Alcohol vs quality

2. **model_results.png** - Model performance visualizations
   - Model comparison bar chart
   - Feature importance
   - Confusion matrix
   - ROC curves

3. **wine_quality_model.pkl** - Trained Random Forest model
4. **wine_quality_scaler.pkl** - StandardScaler for preprocessing

## 🔧 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 Model Interpretability

### Feature Importance Rankings

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | alcohol_density_ratio | 13.01% |
| 2 | alcohol | 11.25% |
| 3 | sulphates | 10.28% |
| 4 | volatile acidity | 8.62% |
| 5 | total sulfur dioxide | 7.04% |
| 6 | free_sulfur_ratio | 6.51% |
| 7 | density | 5.79% |
| 8 | chlorides | 5.41% |
| 9 | acidity_sugar_ratio | 5.09% |
| 10 | total_acidity | 5.05% |

### Model Insights

- **Alcohol content** is the single most important chemical property
- **Volatile acidity** is the strongest negative indicator
- **Engineered features** (ratios) capture complex relationships
- **Tree-based models** outperform linear models significantly
- **Ensemble methods** provide best generalization

## 🎯 Recommendations

### For Wine Producers:
1. Focus on increasing alcohol content (within reasonable limits)
2. Minimize volatile acidity during fermentation
3. Optimize sulphate levels
4. Maintain proper alcohol-to-density balance

### For Model Improvement:
1. Collect more data (especially for extreme quality scores)
2. Add temporal features (aging time, vintage)
3. Include grape variety information
4. Consider price as a feature
5. Explore deep learning approaches

## 📝 Notes

- This is a red wine dataset
- Quality is subjective and based on sensory data
- Chemical measurements are objective
- Model achieves ~77% accuracy (better than random 54%)
- Feature engineering significantly improves performance
- Cross-validation ensures model robustness

## 🔬 Future Work

1. Multi-class classification (predict exact quality scores)
2. Regression approach for continuous quality prediction
3. White wine dataset comparison
4. Regional/varietal analysis
5. Neural network architectures
6. SHAP values for better interpretability
7. Time series analysis if vintage data available

---

**Created**: February 2026
**Model Type**: Random Forest Classifier
**Best Accuracy**: 77.1%
**Dataset**: Wine Quality Dataset (Red Wine)
