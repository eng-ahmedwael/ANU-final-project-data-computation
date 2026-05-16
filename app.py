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
    
    # Define expected features
    expected_features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                       'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                       'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
                       'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    n_features = 21
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown("### 📝 أدخل بيانات المريض")

# Create three columns for more compact layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### الحالات الطبية")
    HighBP = st.selectbox("🩸 ضغط دم مرتفع", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Smoker = st.selectbox("🚬 مدخن", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Stroke = st.selectbox("🧠 جلطة دماغية", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Diabetes = st.selectbox("🩺 سكري", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

with col2:
    st.markdown("#### الفحوصات والتأمين")
    HighChol = st.selectbox("🧬 كوليسترول مرتفع", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    CholCheck = st.selectbox("✓ فحص كوليسترول", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    AnyHealthcare = st.selectbox("🏥 تأمين صحي", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    NoDocbcCost = st.selectbox("💰 تجنب الطبيب", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

with col3:
    st.markdown("#### نمط الحياة")
    PhysActivity = st.selectbox("🏃 نشاط بدني", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Fruits = st.selectbox("🍎 فواكه", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    Veggies = st.selectbox("🥬 خضروات", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    HvyAlcoholConsump = st.selectbox("🍷 كحول ثقيل", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

# Health measurements in columns
st.markdown("#### المقاييس الصحية والديموغرافية")
col4, col5, col6, col7 = st.columns(4)

with col4:
    BMI = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    GenHlth = st.select_slider("📊 الحالة الصحية", options=[1, 2, 3, 4, 5], value=3)

with col5:
    MentHlth = st.slider("🧠 أيام سيئة (نفسي)", min_value=0, max_value=30, value=0)
    PhysHlth = st.slider("💪 أيام سيئة (بدني)", min_value=0, max_value=30, value=0)

with col6:
    DiffWalk = st.selectbox("🚶 صعوبة مشي", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    st.markdown("---")
    st.markdown("""
    **فئات التعليم:**
    - **1** = لم يحضر المدرسة أو روضة فقط
    - **2** = الصفوف 1-8 (الابتدائي)
    - **3** = الصفوف 9-11 (الإعدادي)
    - **4** = صف 12 أو معادلة (ثانوي)
    - **5** = كلية 1-3 سنوات (دبلوم)
    - **6** = كلية 4+ سنوات (بكالوريوس وأعلى)
    """)

with col7:
    Age = st.slider("👤 العمر", min_value=18, max_value=80, value=40)
    Sex = st.selectbox("⚤ الجنس", [0, 1], format_func=lambda x: "ذكر" if x == 1 else "أنثى")

# Income with proper categorization (1-8)
st.markdown("#### 🎓 التعليم (التعليم)")
Education = st.select_slider("🎓 اختر مستوى التعليم (1-6)", options=[1, 2, 3, 4, 5, 6], value=3)

st.markdown("#### 💵 الدخل السنوي (الدخل السنوي)")
st.markdown("""
**فئات الدخل:**
- **1** = أقل من 15,000$
- **2** = 15,000 - 25,000$
- **3** = 25,000 - 35,000$
- **4** = 35,000 - 50,000$
- **5** = 50,000 - 75,000$
- **6** = 75,000 - 100,000$
- **7** = 100,000 - 150,000$
- **8** = أكثر من 150,000$
""")
income_cat = st.slider("💵 اختر فئة الدخل (1-8)", min_value=1, max_value=8, value=4)

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

