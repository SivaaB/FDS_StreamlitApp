import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

# Page config
st.set_page_config(
    page_title="Health Risk Assessment App",
    page_icon="👨‍⚕️",
    layout="wide"
)

# Model training function
@st.cache_resource
def train_models(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
                continue
        
        return models, scaler
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None

# Cache data loading
@st.cache_data
def load_datasets():
    try:
        breast_cancer = pd.read_csv("StreamlitApp/CSV/breast-cancer.csv")
        diabetes = pd.read_csv("StreamlitApp/CSV/diabetes_prediction_dataset.csv")
        heart = pd.read_csv("StreamlitApp/CSV/heart2.csv")
        lung = pd.read_csv("StreamlitApp/CSV/survey lung cancer.csv")
        
        # Print column names for debugging
        print("Lung Cancer Dataset Columns:", lung.columns.tolist())
        
        # Preprocess datasets
        le = LabelEncoder()
        
        # Breast Cancer preprocessing
        breast_cancer['diagnosis'] = (breast_cancer['diagnosis'] == 'M').astype(int)
        
        # Diabetes preprocessing
        diabetes['gender'] = le.fit_transform(diabetes['gender'])
        diabetes['smoking_history'] = le.fit_transform(diabetes['smoking_history'])
        
        # Heart Disease preprocessing
        heart['Sex'] = le.fit_transform(heart['Sex'])
        heart['ChestPainType'] = le.fit_transform(heart['ChestPainType'])
        heart['ExerciseAngina'] = le.fit_transform(heart['ExerciseAngina'])
        heart['ST_Slope'] = le.fit_transform(heart['ST_Slope'])
        
        # Lung Cancer preprocessing
        lung['GENDER'] = le.fit_transform(lung['GENDER'])
        lung['LUNG_CANCER'] = (lung['LUNG_CANCER'] == 'YES').astype(int)
        
        # Convert all Yes/No columns with exact column names from CSV
        yes_no_columns = [
            'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
            'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',  # Note the spaces
            'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for col in yes_no_columns:
            if col in lung.columns:
                # Convert numeric 1,2 to keep existing encoding
                if lung[col].dtype in ['int64', 'float64']:
                    continue
                lung[col] = lung[col].map({'YES': 2, 'NO': 1})
            else:
                st.error(f"Column {col} not found in lung cancer dataset")
                print(f"Missing column: {col}")
        
        return breast_cancer, diabetes, heart, lung
    except FileNotFoundError as e:
        st.error(f"Error loading datasets: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error processing datasets: {str(e)}")
        print(f"Error details: {str(e)}")
        st.stop()

# Title and description
st.title("👨‍⚕️ Comprehensive Health Risk Assessment")
st.markdown("""
This application helps assess your risk for various health conditions using machine learning models.
Upload your medical parameters and get instant risk assessments for:
- 🎗️ Breast Cancer
- 🫁 Lung Cancer
- ❤️ Heart Disease
- 💉 Diabetes
""")

# Load all datasets
breast_cancer, diabetes, heart, lung = load_datasets()

# Sidebar navigation
st.sidebar.header("Navigation")
condition = st.sidebar.selectbox(
    "Select Health Condition", 
    ["Data - Visualization Overview", "Breast Cancer", "Lung Cancer", "Heart Disease", "Diabetes"]
)

# Overview page
if condition == "Overview":
    st.header("Health Risk Statistics")
    
    col2 = st.columns(1)

    condition2 = st.sidebar.selectbox(
    "Select Health Condition", 
    ["Breast Cancer", "Lung Cancer", "Heart Disease", "Diabetes"]
    )

    condition3 = st.sidebar.selectbox(
    "Select Classification Model", 
    ["Logisitic Regression", "Random Forest", "XGBoost"]
    )
    
    with col2:
        st.subheader("Analysis")
        
        if condition2 == "Breast Cancer" & condition3 == "Logistic Regression":
            fig, ax = plt.subplots(figsize=(10, 8))
            selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']
            sns.heatmap(breast_cancer[selected_features].corr(), annot=True, cmap='coolwarm')
            plt.title("Breast Cancer Features Correlation")
            st.pyplot(fig)

        elif condition2 == "Lung Cancer" & condition3 == "Logistic Regression":
            fig, ax = plt.subplots(figsize=(10, 8))
            selected_features = ['age', 'bmi', 'blood_glucose_level', 'diabetes']
            sns.heatmap(diabetes[selected_features].corr(), annot=True, cmap='coolwarm')
            plt.title("Diabetes Features Correlation")
            st.pyplot(fig)
        
        elif condition2 == "Heart Disease" & condition3 == "Logistic Regression":
            fig, ax = plt.subplots(figsize=(10, 8))
            selected_features = ['Age', 'RestingBP', 'Cholesterol', 'HeartDisease']
            sns.heatmap(heart[selected_features].corr(), annot=True, cmap='coolwarm')
            plt.title("Heart Disease Features Correlation")
            st.pyplot(fig)
        
        elif condition2 == "Diabetes" & condition3 == "Logistic Regression":
            fig, ax = plt.subplots(figsize=(10, 8))
            selected_features = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'LUNG_CANCER']
            sns.heatmap(lung[selected_features].corr(), annot=True, cmap='coolwarm')
            plt.title("Lung Cancer Features Correlation")
            st.pyplot(fig)

elif condition == "Breast Cancer":
    st.header("🎗️ Breast Cancer Risk Assessment")
    
    # Input form for breast cancer parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius = st.number_input("Mean Radius", min_value=0.0, max_value=40.0, value=15.0)
        texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=20.0)
        perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, value=90.0)
        
    with col2:
        area = st.number_input("Mean Area", min_value=0.0, max_value=2500.0, value=600.0)
        smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.2, value=0.1)
        compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=0.3, value=0.1)
        
    with col3:
        concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=0.4, value=0.1)
        symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=0.3, value=0.2)
        fractal_dim = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.1, value=0.06)

    if st.button("Predict Breast Cancer Risk"):
        # Prepare input data
        input_data = np.array([[radius, texture, perimeter, area, smoothness, 
                               compactness, concavity, symmetry, fractal_dim]])
        
        # Select features for training
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                   'smoothness_mean', 'compactness_mean', 'concavity_mean',
                   'symmetry_mean', 'fractal_dimension_mean']
        
        X = breast_cancer[features]
        y = breast_cancer['diagnosis']
        
        # Train models and get predictions
        models, scaler = train_models(X, y)
        input_scaled = scaler.transform(input_data)
        
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        for name, model in models.items():
            prediction = model.predict_proba(input_scaled)[0]
            
            with col1:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie([prediction[1], 1-prediction[1]], colors=['salmon', 'lightblue'],
                      labels=['Malignant', 'Benign'])
                plt.title(f"{name}")
                st.pyplot(fig)
            
            with col2:
                st.write(f"**{name} Results:**")
                st.write(f"Probability of Malignant: {prediction[1]:.2%}")
                st.write(f"Probability of Benign: {prediction[0]:.2%}")
                st.write("---")

elif condition == "Diabetes":
    st.header("💉 Diabetes Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        hypertension = st.checkbox("Hypertension")
        
    with col2:
        heart_disease = st.checkbox("Heart Disease")
        smoking = st.selectbox("Smoking History", ["never", "former", "current", "not current"])
        blood_glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=300.0, value=100.0)

    if st.button("Predict Diabetes Risk"):
        # Convert categorical inputs
        gender_map = {"Male": 1, "Female": 0}
        smoking_map = {"never": 0, "former": 1, "current": 2, "not current": 3}
        
        input_data = np.array([[
            age, gender_map[gender], bmi, hypertension, heart_disease,
            smoking_map[smoking], blood_glucose
        ]])
        
        features = ['age', 'gender', 'bmi', 'hypertension', 'heart_disease',
                   'smoking_history', 'blood_glucose_level']
        
        X = diabetes[features]
        y = diabetes['diabetes']
        
        models, scaler = train_models(X, y)
        input_scaled = scaler.transform(input_data)
        
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        for name, model in models.items():
            prediction = model.predict_proba(input_scaled)[0]
            
            with col1:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie([prediction[1], 1-prediction[1]], colors=['salmon', 'lightblue'],
                      labels=['Diabetic', 'Non-Diabetic'])
                plt.title(f"{name}")
                st.pyplot(fig)
            
            with col2:
                st.write(f"**{name} Results:**")
                st.write(f"Probability of Diabetic: {prediction[1]:.2%}")
                st.write(f"Probability of Non-Diabetic: {prediction[0]:.2%}")
                st.write("---")

elif condition == "Heart Disease":
    st.header("❤️ Heart Disease Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        
    with col2:
        resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
        fasting_bs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
        
    with col3:
        max_hr = st.number_input("Maximum Heart Rate", min_value=0, max_value=220, value=150)
        exercise_angina = st.checkbox("Exercise-Induced Angina")
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    if st.button("Predict Heart Disease Risk"):
        try:
            # Convert categorical inputs
            sex_map = {"Male": 1, "Female": 0}
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, 
                     "Non-anginal Pain": 2, "Asymptomatic": 3}
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            
            input_data = np.array([[
                age, sex_map[sex], cp_map[chest_pain], resting_bp, cholesterol,
                fasting_bs, max_hr, exercise_angina, slope_map[st_slope]
            ]])
            
            features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                       'FastingBS', 'MaxHR', 'ExerciseAngina', 'ST_Slope']
            
            X = heart[features]
            y = heart['HeartDisease']
            
            models, scaler = train_models(X, y)
            if models is None or scaler is None:
                st.error("Error in model training. Please try again.")
                st.stop()
                
            input_scaled = scaler.transform(input_data)
            
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            for name, model in models.items():
                prediction = model.predict_proba(input_scaled)[0]
                
                with col1:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie([prediction[1], 1-prediction[1]], colors=['salmon', 'lightblue'],
                          labels=['Heart Disease', 'No Heart Disease'])
                    plt.title(f"{name}")
                    st.pyplot(fig)
                
                with col2:
                    st.write(f"**{name} Results:**")
                    st.write(f"Probability of Heart Disease: {prediction[1]:.2%}")
                    st.write(f"Probability of No Heart Disease: {prediction[0]:.2%}")
                    st.write("---")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

else:  # Lung Cancer
    st.header("🫁 Lung Cancer Risk Assessment")
    
    # Debug info - can be removed in production
    st.write("Debug Info:")
    st.write("Available columns:", lung.columns.tolist())
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
        anxiety = st.selectbox("Anxiety", ["Yes", "No"])
        
    with col2:
        peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
        chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        allergy = st.selectbox("Allergy", ["Yes", "No"])
        wheezing = st.selectbox("Wheezing", ["Yes", "No"])

    if st.button("Predict Lung Cancer Risk"):
        try:
            # Convert categorical inputs
            gender_map = {"Male": 1, "Female": 0}
            yes_no_map = {"Yes": 2, "No": 1}
            
            input_data = np.array([[
                age, gender_map[gender], yes_no_map[smoking],
                yes_no_map[yellow_fingers], yes_no_map[anxiety],
                yes_no_map[peer_pressure], yes_no_map[chronic_disease],
                yes_no_map[fatigue], yes_no_map[allergy],
                yes_no_map[wheezing]
            ]])
            
            # Updated feature names to match exactly with dataset columns
            features = ['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
                       'WHEEZING']
            
            # Verify all features exist in the dataset
            missing_features = [f for f in features if f not in lung.columns]
            if missing_features:
                st.error(f"Missing features in dataset: {missing_features}")
                st.warning("""
                    There seems to be a mismatch in the dataset columns. 
                    Please check the following:
                    1. Column names are case-sensitive
                    2. Spaces in column names match exactly
                    3. No extra spaces at the end of column names
                """)
                st.stop()
            
            X = lung[features]
            y = lung['LUNG_CANCER']
            
            models, scaler = train_models(X, y)
            if models is None or scaler is None:
                st.error("Error in model training. Please try again.")
                st.stop()
            
            input_scaled = scaler.transform(input_data)
            
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            for name, model in models.items():
                prediction = model.predict_proba(input_scaled)[0]
                
                with col1:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie([prediction[1], 1-prediction[1]], colors=['salmon', 'lightblue'],
                          labels=['Lung Cancer', 'No Lung Cancer'])
                    plt.title(f"{name}")
                    st.pyplot(fig)
                
                with col2:
                    st.write(f"**{name} Results:**")
                    st.write(f"Probability of Lung Cancer: {prediction[1]:.2%}")
                    st.write(f"Probability of No Lung Cancer: {prediction[0]:.2%}")
                    st.write("---")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.warning("""
                Troubleshooting steps:
                1. Check if all required features are present in the dataset
                2. Verify the data types of input values
                3. Ensure all categorical variables are properly encoded
            """)
