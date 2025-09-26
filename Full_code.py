import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------
# 1. Preprocess diet_recommendations_dataset
# --------------------------

dataset_path = r"C:\Users\harsh\Downloads\diet_recommendations_dataset.csv"
df = pd.read_csv(dataset_path)

# Drop rows with missing key values
df.dropna(subset=['Disease_Type', 'Dietary_Restrictions', 'Allergies'], inplace=True)

# Separate categorical & numerical features
categorical_features = []
numerical_features = []
for column in df.columns:
    if df[column].dtype == 'object':
        categorical_features.append(column)
    else:
        numerical_features.append(column)

# Encode categorical features except ID & target
categorical_features_to_encode = [col for col in categorical_features if col not in ['Patient_ID', 'Diet_Recommendation']]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features_to_encode])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features_to_encode), index=df.index)

# Combine encoded + numerical + ID + target
df = pd.concat([df[numerical_features], encoded_df, df[['Patient_ID', 'Diet_Recommendation']]], axis=1)

# Scale numerical features
numerical_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Daily_Caloric_Intake',
                      'Cholesterol_mg/dL', 'Blood_Pressure_mmHg', 'Glucose_mg/dL',
                      'Weekly_Exercise_Hours', 'Adherence_to_Diet_Plan',
                      'Dietary_Nutrient_Imbalance_Score']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save preprocessed dataset
output_path = r"C:\Users\harsh\Downloads\preprocessed_diet_recommendations_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Preprocessed dataset saved to: {output_path}")


# --------------------------
# 2. Preprocess healthy_meal_plans
# --------------------------

dataset_path_new = r"C:\Users\harsh\Downloads\healthy_meal_plans.csv"
df_new = pd.read_csv(dataset_path_new)

# Separate categorical & numerical
categorical_features_new = []
numerical_features_new = []
for column in df_new.columns:
    if df_new[column].dtype == 'object':
        categorical_features_new.append(column)
    else:
        numerical_features_new.append(column)

# Encode categorical features
categorical_features_to_encode_new = [col for col in categorical_features_new if col not in []]
encoder_new = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features_new = encoder_new.fit_transform(df_new[categorical_features_to_encode_new])
encoded_df_new = pd.DataFrame(encoded_features_new, columns=encoder_new.get_feature_names_out(categorical_features_to_encode_new), index=df_new.index)

# Combine encoded + numerical
df_new = pd.concat([df_new[numerical_features_new], encoded_df_new], axis=1)

# Scale numerical features
numerical_features_new = ['num_ingredients', 'calories', 'prep_time', 'protein', 'fat', 'carbs']
scaler_new = StandardScaler()
df_new[numerical_features_new] = scaler_new.fit_transform(df_new[numerical_features_new])

# Save preprocessed dataset
output_path_new = r"C:\Users\harsh\Downloads\preprocessed_healthy_meal_plans.csv"
df_new.to_csv(output_path_new, index=False)
print(f"Preprocessed dataset saved to: {output_path_new}")


# --------------------------
# 3. Preprocess hospital data analysis
# --------------------------

dataset_path_hospital = r"C:\Users\harsh\Downloads\hospital data analysis.csv"
df_hospital = pd.read_csv(dataset_path_hospital)

# Separate categorical & numerical
categorical_features_hospital = []
numerical_features_hospital = []
for column in df_hospital.columns:
    if df_hospital[column].dtype == 'object':
        categorical_features_hospital.append(column)
    else:
        numerical_features_hospital.append(column)

# Encode categorical features except Patient_ID
categorical_features_to_encode_hospital = [col for col in categorical_features_hospital if col not in ['Patient_ID']]
encoder_hospital = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features_hospital = encoder_hospital.fit_transform(df_hospital[categorical_features_to_encode_hospital])
encoded_df_hospital = pd.DataFrame(encoded_features_hospital, columns=encoder_hospital.get_feature_names_out(categorical_features_to_encode_hospital), index=df_hospital.index)

# Combine encoded + numerical
df_hospital = pd.concat([df_hospital[numerical_features_hospital], encoded_df_hospital], axis=1)

# Scale numerical features (exclude Patient_ID from scaling)
numerical_features_to_scale_hospital = [col for col in numerical_features_hospital if col not in ['Patient_ID']]
scaler_hospital = StandardScaler()
df_hospital[numerical_features_to_scale_hospital] = scaler_hospital.fit_transform(df_hospital[numerical_features_to_scale_hospital])

# Save preprocessed dataset
output_path_hospital = r"C:\Users\harsh\Downloads\preprocessed_hospital_data_analysis.csv"
df_hospital.to_csv(output_path_hospital, index=False)
print(f"Preprocessed dataset saved to: {output_path_hospital}")
