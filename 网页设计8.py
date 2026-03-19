import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
model = joblib.load('XG.pkl')
feature_names = [
    "感染时程大于3个月", "骨暴露", "病变部位", "钙", "碱性磷酸酶",
    "血浆纤维蛋白原", "C反应蛋白"
]
st.set_page_config(page_title="骨髓炎复发风险预测器", layout="wide")
st.title("骨髓炎复发风险预测器")
st.markdown("### 请填写以下信息，点击预测获取骨髓炎复发风险评估结果")
感染时程大于3个月 = st.selectbox(
   "感染时程大于3个月:",
    options=[0, 1],
    format_func=lambda x: 'False (0)' if x == 0 else 'True (1)'
)
骨暴露 = st.selectbox(
   "骨暴露:",
    options=[0, 1],
    format_func=lambda x: 'False (0)' if x == 0 else 'True (1)'
)
病变部位 = st.selectbox(
   "病变部位:",
    options=[0, 1],
    format_func=lambda x: "手足指（趾）" if x == 0 else "四肢骨"
)
钙 = st.number_input("钙:", min_value=0, max_value=1000, value=0)
碱性磷酸酶 = st.number_input("碱性磷酸酶:", min_value=0, max_value=1000, value=0)
血浆纤维蛋白原 = st.number_input("血浆纤维蛋白原:", min_value=0, max_value=1000, value=0)
C反应蛋白 = st.number_input("C反应蛋白:", min_value=0, max_value=1000, value=0)
feature_values = [
    Infection_duration, Bone_exposure, Lesion_site, Ca, ALP,
    Plasma_fibrinogen, CRP
]
features = np.array([feature_values])
if st.button("预测"):
    
    predicted_class = model.predict(features)[0] 
    predicted_proba = model.predict_proba(features)[0] 
    st.subheader("预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险"
    st.write(f"**风险等级：{predicted_class} ({risk_label})**")
    st.write(f"**风险概率：** 低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}")
    st.subheader("💡 健康建议")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"模型预测您的骨髓炎复发风险为高风险（概率{probability:.1f}%）。"
            "建议优化临床决策，帮助改善患者预后"
        )
    explainer = shap.TreeExplainer(model)  
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names)) 
    shap.force_plot(explainer.expected_value, shap_values, pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")