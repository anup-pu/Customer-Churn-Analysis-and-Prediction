# predict_churn_from_mysql.py

import pandas as pd
import pymysql
import joblib

# Load trained model and encoders
rf_model = joblib.load('churn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Connect to MySQL
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='Anup@123',
    database='db_churn'
)

# Load new data to predict
query = "SELECT * FROM vw_joindata"
new_data = pd.read_sql(query, conn)

original_data = new_data.copy()

# Drop unused columns
new_data = new_data.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)

# Apply label encoders to new data
for column in new_data.select_dtypes(include='object').columns:
    if column in label_encoders:
        new_data[column] = label_encoders[column].transform(new_data[column])
    else:
        raise ValueError(f"Unexpected column '{column}' in new data.")

# Predict
new_predictions = rf_model.predict(new_data)
original_data['Customer_Status_Predicted'] = new_predictions

# Filter churned customers
churned_customers = original_data[original_data['Customer_Status_Predicted'] == 1]

# Save to CSV
churned_customers.to_csv(r"C:\Users\Anup Kumar\OneDrive\Desktop\churn analysis\churn model\Predicted_Churn.csv", index=False)

print("✅ Predicted churn customers saved successfully.")
