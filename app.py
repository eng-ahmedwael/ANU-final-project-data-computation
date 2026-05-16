import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")

# Load model and metadata
model_exists = os.path.exists("svm_model.pkl")

if not model_exists:
    st.error("❌ Model not found!")
    st.info("""
    **اتبع هذه الخطوات:**
    
    1. افتح ملف `project.ipynb` الـ notebook
    2. شغّل جميع الخلايا من الأعلى للأسفل (Cell → Run All)
    3. تأكد من طباعة رسالة "✓ Model saved with 21 features" في الخلية الأخيرة
    4. بعدها ارجع إلى هنا
    
    """)
    st.stop()

try:
    model = joblib.load("svm_model.pkl")
    
    # Try to load metadata if available
    if os.path.exists("model_metadata.pkl"):
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        expected_features = metadata.get('feature_names', [])
        n_features = metadata.get('n_features', 0)
    else:
        expected_features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                           'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                           'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
                           'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
        n_features = 21
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown("### 📝 أدخل بيانات المريض")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### الحالات الطبية")
    HighBP = st.selectbox("🩸 ضغط دم مرتفع", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    HighChol = st.selectbox("🧬 كوليسترول مرتفع", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    CholCheck = st.selectbox("✓ فحص الكوليسترول (آخر 5 سنوات)", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Smoker = st.selectbox("🚬 مدخن", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Stroke = st.selectbox("🧠 جلطة دماغية سابقة", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Diabetes = st.selectbox("🩺 سكري", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    
with col2:
    st.markdown("#### نمط الحياة")
    PhysActivity = st.selectbox("🏃 نشاط بدني منتظم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Fruits = st.selectbox("🍎 تناول فواكه منتظم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Veggies = st.selectbox("🥬 تناول خضروات منتظم", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    HvyAlcoholConsump = st.selectbox("🍷 كحول ثقيل", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    AnyHealthcare = st.selectbox("🏥 تأمين صحي", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    NoDocbcCost = st.selectbox("💰 تجنب الطبيب بسبب التكلفة", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

# Another row for measurements
st.markdown("#### القياسات والمؤشرات الصحية")
col3, col4 = st.columns(2)

with col3:
    BMI = st.number_input("⚖️ مؤشر كتلة الجسم (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    GenHlth = st.select_slider("📊 الحالة الصحية العامة", options=[1, 2, 3, 4, 5], value=3, 
                               format_func=lambda x: {1: "ممتازة", 2: "جيدة جداً", 3: "جيدة", 4: "عادلة", 5: "سيئة"}[x])
    MentHlth = st.slider("🧠 أيام الصحة النفسية السيئة (آخر 30 يوم)", min_value=0, max_value=30, value=0)

with col4:
    DiffWalk = st.selectbox("🚶 صعوبة في المشي أو الصعود", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    PhysHlth = st.slider("💪 أيام الصحة البدنية السيئة (آخر 30 يوم)", min_value=0, max_value=30, value=0)

# Demographics
st.markdown("#### معلومات ديموغرافية")
col5, col6, col7 = st.columns(3)

with col5:
    Age = st.slider("👤 العمر", min_value=18, max_value=80, value=40)

with col6:
    Sex = st.selectbox("⚤ الجنس", [0, 1], format_func=lambda x: "ذكر" if x == 1 else "أنثى")

with col7:
    Education = st.select_slider("🎓 مستوى التعليم", options=[1, 2, 3, 4, 5, 6], value=3,
                                format_func=lambda x: {1: "لم أكمل", 2: "ثانوي", 3: "دبلوم", 4: "بكالوريوس", 5: "ماجستير", 6: "دكتوراه"}[x])

# Income with better range
st.markdown("#### الدخل السنوي (بآلاف الدولارات)")
Income = st.slider("💵 الدخل السنوي", min_value=0, max_value=150, value=50, step=5,
                   help="0-25K, 25-35K, 35-50K, 50-75K, 75-100K, 100-150K, 150K+")

# Convert income to category (1-8 scale based on ranges)
if Income < 25:
    income_cat = 1
elif Income < 35:
    income_cat = 2
elif Income < 50:
    income_cat = 3
elif Income < 75:
    income_cat = 4
elif Income < 100:
    income_cat = 5
elif Income < 150:
    income_cat = 6
else:
    income_cat = 7

st.divider()

# Predict button
if st.button("🔍 تنبأ بالنتيجة", use_container_width=True, type="primary"):
    try:
        # Create DataFrame with exact column names and order from training data
        input_data = pd.DataFrame({
            'HighBP': [float(HighBP)],
            'HighChol': [float(HighChol)],
            'CholCheck': [float(CholCheck)],
            'BMI': [float(BMI)],
            'Smoker': [float(Smoker)],
            'Stroke': [float(Stroke)],
            'Diabetes': [float(Diabetes)],
            'PhysActivity': [float(PhysActivity)],
            'Fruits': [float(Fruits)],
            'Veggies': [float(Veggies)],
            'HvyAlcoholConsump': [float(HvyAlcoholConsump)],
            'AnyHealthcare': [float(AnyHealthcare)],
            'NoDocbcCost': [float(NoDocbcCost)],
            'GenHlth': [float(GenHlth)],
            'MentHlth': [float(MentHlth)],
            'PhysHlth': [float(PhysHlth)],
            'DiffWalk': [float(DiffWalk)],
            'Sex': [float(Sex)],
            'Age': [float(Age)],
            'Education': [float(Education)],
            'Income': [float(income_cat)]
        })
        
        # Ensure columns are in correct order if metadata is available
        if expected_features:
            input_data = input_data[expected_features]

        prediction = model.predict(input_data)

        st.divider()
        if prediction[0] == 1:
            st.error("⚠️ **خطر مرتفع من أمراض القلب**", icon="❤️‍🩹")
            st.markdown("### ⚠️ التوصيات:")
            st.warning("""
            - استشر طبيب القلب فوراً
            - أجرِ فحوصات دورية منتظمة
            - حافظ على نمط حياة صحي
            - قلل من الضغوط والتوتر
            """)
        else:
            st.success("✓ **خطر منخفض من أمراض القلب**", icon="💚")
            st.markdown("### 💚 التوصيات:")
            st.info("""
            - استمر في نمط الحياة الصحي
            - مارس الرياضة بانتظام
            - راقب صحتك بشكل دوري
            - حافظ على وزن صحي
            """)
            
    except Exception as e:
        st.error(f"❌ حدث خطأ في التنبؤ: {str(e)}")
        st.warning(f"Expected {n_features} features, got {input_data.shape[1] if 'input_data' in locals() else 'unknown'}")

# Footer
st.divider()
st.markdown("""
---
**ملاحظة مهمة:** هذا التطبيق هو أداة تعليمية فقط ولا يجب أن يحل محل استشارة الطبيب المتخصص.
للحصول على تشخيص دقيق، يرجى استشارة مقدم الرعاية الصحية المؤهل.
""")

