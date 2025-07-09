import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1. Load dataset
df = pd.read_csv('../dataset/cost_of_living.csv')  # Dataset containing features and Absolute_Poverty

# 2. Prepare features (exclude 'State' and 'District') and target ('Absolute_Poverty')
X = df.drop(columns=['State', 'District', 'Absolute_Poverty'])
y = df['Absolute_Poverty']

# 3. Encode categorical feature 'Strata' (Urban/Rural) using LabelEncoder
le_strata = LabelEncoder()
X['Strata'] = le_strata.fit_transform(X['Strata'])  # e.g., 'Rural'->0, 'Urban'->1

# 4. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train a Gradient Boosting Regressor on the training set
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. Save the trained model and label encoder to .pkl files
joblib.dump(model, '../model/poverty_model.pkl')
joblib.dump(le_strata, '../model/strata_encoder.pkl')
