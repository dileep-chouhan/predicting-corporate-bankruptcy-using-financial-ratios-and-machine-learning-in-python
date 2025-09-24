import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic financial data
num_companies = 1000
data = {
    'CurrentRatio': np.random.uniform(0.5, 5, num_companies),
    'DebtToEquity': np.random.uniform(0, 2, num_companies),
    'ProfitMargin': np.random.uniform(-0.5, 0.5, num_companies),
    'ReturnOnAssets': np.random.uniform(-0.2, 0.2, num_companies),
    'Bankrupt': np.random.choice([0, 1], size=num_companies, p=[0.8, 0.2]) # 20% bankruptcy rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this example, we'll assume the data is clean.
X = df.drop('Bankrupt', axis=1)
y = df['Bankrupt']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
# Train a Logistic Regression model (a simple model for demonstration)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 4. Visualization ---
#Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Solvent', 'Bankrupt'], yticklabels=['Solvent', 'Bankrupt'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")
#Feature Importance (Illustrative - Logistic Regression doesn't directly provide feature importance like tree-based models)
#We can use coefficients as a proxy for feature importance in this case.
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")