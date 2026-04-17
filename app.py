import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load your saved model
@st.cache_resource
def load_model():
    model = joblib.load('diabetes_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

# Load the model
model, feature_names = load_model()

# Page setup
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🏥")

# Title
st.title("🏥 Diabetes Risk Predictor")
st.markdown("### Enter your health information below to check your diabetes risk")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Basic Information")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    st.subheader("📊 Health Metrics")
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

# Predict button
st.markdown("---")
if st.button("🔍 Predict My Diabetes Risk", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_percentage = probability * 100
    
    # Show results
    st.markdown("## 📊 Your Results")
    
    # Show risk meter
    if risk_percentage < 30:
        st.success(f"### ✅ LOW RISK: {risk_percentage:.1f}%")
        st.markdown("Your diabetes risk appears low. Keep up your healthy habits!")
    elif risk_percentage < 70:
        st.warning(f"### ⚠️ MODERATE RISK: {risk_percentage:.1f}%")
        st.markdown("Your risk is moderate. Consider making some lifestyle improvements.")
    else:
        st.error(f"### 🔴 HIGH RISK: {risk_percentage:.1f}%")
        st.markdown("Your risk is high. Please consult a healthcare provider soon.")
    
    # Visual progress bar
    st.progress(int(risk_percentage))
    
    # Personalized recommendations
    st.subheader("💪 Personalized Health Tips")
    
    recommendations = []
    if glucose > 140:
        recommendations.append("• Monitor your blood sugar levels regularly")
    if bmi > 25:
        recommendations.append("• Try to maintain a healthy weight through diet and exercise")
    if blood_pressure > 80:
        recommendations.append("• Reduce salt intake and stay physically active")
    if age > 45:
        recommendations.append("• Schedule regular health check-ups")
    if pregnancies > 5:
        recommendations.append("• Discuss gestational diabetes history with your doctor")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.write("• You're doing great! Keep maintaining a healthy lifestyle!")
        st.write("• Regular exercise and balanced diet are key to prevention")

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")

# Sidebar with info
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    **How it works:**
    - Uses Machine Learning (Random Forest)
    - Trained on 768 patient records
    - Accuracy: ~78%
    
    **Normal ranges:**
    - Glucose: 70-140 mg/dL
    - BP: < 120/80 mm Hg
    - BMI: 18.5-25
    
    *Higher values generally increase risk*
    """
)
