import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
import shap  
import matplotlib.pyplot as plt  
from lime.lime_tabular import LimeTabularExplainer  
  
# 加载模型和数据  
model = joblib.load('model_rf.pkl')  
X_test = pd.read_csv('X_test.csv')  
  
# 特征名称  
feature_names = [  
    "Radscore",  
    "Clinical_T_stage",  
    "Clinical_N_stage",  
    "Location",  
    "Age",  
    "Drinking_history",  
    "AFP",  
    "PLT",  
    "ALB",  
    "MONO",  
    "Weight",  
    "PAB",  
    "CA153",  
    "CEA"  
]  
  
# Streamlit 用户界面  
st.title("Differentiating AMC and MC")  
  
# 用户输入  
# 导入Streamlit库（如果尚未导入）  
import streamlit as st  
  
# 用户输入  
Radscore = st.number_input("Radscore:", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
  
# Clinical T stage 选项修正，格式化函数中的显示逻辑调整  
Clinical_T_stage = st.selectbox("Clinical T stage:", options=[0, 1, 2, 3, 4], format_func=lambda x: ["cT2", "cT3", "cT4a", "cT4b", "Unknown"][x])  
  
# Clinical N stage 选项和格式化函数正常  
Clinical_N_stage = st.selectbox("Clinical N stage:", options=[0, 1], format_func=lambda x: "cN0" if x == 0 else "cN+")  
  
# Location 选项修正，格式化函数中的逻辑正确  
Location = st.selectbox("Location:", options=[0, 1, 2], format_func=lambda x: "rectum" if x == 0 else ("left" if x == 1 else "right"))  
  
# Age 输入修正，初始值在有效范围内  
Age = st.number_input("Age:", min_value=10, max_value=80, value=50, step=1)
  
# Drinking_history 选项和格式化函数正常  
Drinking_history = st.selectbox("Drinking_history:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")  
  
# AFP 输入修正，初始值设置为一个更合理的低值  
AFP = st.number_input("AFP:", min_value=0.7, max_value=780.0, value=10.0, step=1.0)  # 将 max_value 改为 780.0  
# 以下数值输入的修正主要是确保初始值在有效范围内，并且步骤值合理  
PLT = st.number_input("PLT:", min_value=104.0, max_value=780.0, value=281.0, step=1.0)  
ALB = st.number_input("ALB:", min_value=15.0, max_value=49.8, value=42.0, step=0.1)  
MONO = st.number_input("MONO:", min_value=0.1, max_value=1.17, value=0.42, step=0.01)  
Weight = st.number_input("Weight:", min_value=37.0, max_value=95.0, value=60.0, step=1.0)  
PAB = st.number_input("PAB:", min_value=36.0, max_value=415.0, value=71.0, step=1.0)  
CA153 = st.number_input("CA153:", min_value=1.9, max_value=49.0, value=10.0, step=0.1)  
CEA = st.number_input("CEA:", min_value=0.78, max_value=1500.0, value=10.0, step=1.0)  # 确保 max_value 为浮点数
 
  
# 这里可以添加代码来处理这些输入，比如存储到变量、进行数据分析或显示结果等。  
  
# 处理输入并进行预测  
feature_values = [  
    Radscore, Clinical_T_stage, Clinical_N_stage, Location, Age, Drinking_history,  
    AFP, PLT, ALB, MONO, Weight, PAB, CA153, CEA  
]  
  
features = np.array([feature_values])  
  
if st.button("Predict"):  
    predicted_class = model.predict(features)[0]  
    predicted_proba = model.predict_proba(features)[0]  
  
    st.write(f"**Predicted Label**: {predicted_class} (1: Others, 0: LPA)")  
    st.write(f"**Predicted Probability**: {predicted_proba}")  
  
    # probability = predicted_proba[predicted_class] * 100  
    # if predicted_class == 1:  
    #     advice = (  
    #         f"Our model indicates a low probability of pGGNs being pathologically identified as LPA."  
    #         f" The model estimates the probability of LPA as {probability:.1f}%."  
    #         " Operative intervention is advised, specifically an anatomic lobectomy in conjunction with systematic lymph node dissection."  
    #     )  
    # else:  
    #     advice = (  
    #         f"Our model indicates a high probability of pGGNs being pathologically identified as LPA."  
    #         f" The model estimates the probability of other histological subtypes as {probability:.1f}%."  
    #         " It's important to maintain a healthy lifestyle and keep having regular check-ups."  
    #     )  
    # st.write(advice)  
  
      # SHAP 解释  
      st.subheader("SHAP Force Plot Explanation")  
      explainer_shap = shap.TreeExplainer(model)  
      shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))  
      shap.initjs()  
      shap.force_plot(explainer_shap.expected_value[predicted_class], shap_values[predicted_class], pd.DataFrame([feature_values], columns=feature_names))  
    
      # LIME 解释  
      st.subheader("LIME Explanation")  
      lime_explainer = LimeTabularExplainer(X_test.values, feature_names=feature_names, class_names=['LPA', 'Others'], mode='classification')  
      lime_exp = lime_explainer.explain_instance(features.flatten(), predict_fn=model.predict_proba)  
      lime_html = lime_exp.as_html(show_table=False)  
      st.write(lime_html, unsafe_allow_html=True)  # 使用 st.write 显示 HTML，并设置 unsafe_allow_html=True（如果 HTML 是安全的）
