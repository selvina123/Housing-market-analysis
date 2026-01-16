🏠 Housing Market Analysis & Prediction

An end-to-end data analytics and machine learning project focused on analyzing housing market trends and predicting housing-related outcomes using multiple statistical, machine learning, and deep learning models.
This project covers data preprocessing, exploratory data analysis (EDA), model training, evaluation, explainability, and interactive visualization.

📌 Project Objectives

Analyze housing market trends using historical data
Perform feature engineering and exploratory analysis
Train and compare multiple predictive models
Interpret model behavior using explainability techniques
Deploy results through an interactive dashboard

📊 Data Sources

The project uses multiple housing-related datasets, including:
City-level housing starts
Median sale prices over time
Housing inventory and sales counts
New construction and census-based indicators
Formats include CSV and Excel, covering both raw and processed data.

🤖 Models Used for Training

The following models were trained, evaluated, and saved as artifacts:
📈 Regression & Machine Learning Models
Linear Regression (linear_regression.pkl)
Lasso Regression (lasso_regression.pkl)
Decision Tree Regressor (decision_tree.pkl)
Gradient Boosting Regressor (gradient_boosting.pkl)
XGBoost Regressor (xgboost.pkl)
LightGBM Regressor (lightgbm.pkl)

These models were compared using performance metrics stored in:

model_comparison.csv
model_metrics.json

🧠 Deep Learning Model

LSTM (Long Short-Term Memory) Neural Network
(lstm_model.h5)
Used for time-series modeling of housing market trends.

⚙️ Supporting Artifacts

Feature Scaler (scaler.pkl)
Model Documentation (models.md)

🔍 Model Explainability

To ensure transparency and interpretability:

SHAP (SHapley Additive exPlanations) was used
Feature importance visualized in:
shap_summary_plot.png
Analysis implemented in:
shap_analysis.py

This helps understand which features most influence predictions.

📊 Key Visualizations
Generated visual outputs include:
Median sale price trends over time
Housing heat index distribution
Percentage of homes sold above list price
Correlation heatmaps
Interactive dashboard visuals

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/selvina123/Housing-market-analysis.git
cd Housing-market-analysis

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run analysis scripts
python eda.py
python data_preprocessing.py
python model_training.py

4️⃣ Launch the dashboard
python app.py

🛠️ Technologies Used

Python
Pandas, NumPy
Scikit-learn
XGBoost, LightGBM
TensorFlow / Keras (LSTM)
Matplotlib, Seaborn
SHAP
Git & GitHub

📌 Key Takeaways

Compared classical regression, tree-based, ensemble, and deep learning models
Demonstrated model performance trade-offs
Applied explainable AI (XAI) techniques
Delivered results via an interactive application

👩‍💻 Author
Selvina Swarna
Graduate Student – Information Systems
Syracuse University
