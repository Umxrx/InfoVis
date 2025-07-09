from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and encoder
model = joblib.load('model/poverty_model.pkl')
le_strata = joblib.load('model/strata_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Format input
        input_df = pd.DataFrame([{
            'Strata': data['strata'],
            'PLI': float(data['pli']),
            'PAKW': float(data['pakw']),
            'Median_Income': float(data['median_income']),
            'Mean_Income': float(data['mean_income']),
            'Median_Expenditure': float(data['median_expenditure']),
            'Mean_Expenditure': float(data['mean_expenditure']),
            'Gini_Coefficient': float(data['gini_coefficient'])
        }])

        # Encode Strata
        input_df['Strata'] = le_strata.transform(input_df['Strata'])

        # Predict
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
