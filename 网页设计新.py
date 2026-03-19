{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c1cc36d-4df9-4314-9d4b-5acbe539702f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "from lime.lime_tabular import LimeTabularExplainer\n",
    "import warnings\n",
    "\n",
    "# 忽略警告信息\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ====================== 1. 基础配置 ======================\n",
    "# 加载训练好的随机森林模型（确保RF.pkl与脚本同目录）\n",
    "model = joblib.load('XG.pkl')\n",
    "\n",
    "# 加载测试数据（用于LIME解释器，确保X_test.csv与脚本同目录）\n",
    "X_test = pd.read_csv((\"C:/Users/lenovo/Desktop/骨5/B62操作版（无复发列）.csv\"))\n",
    "\n",
    "# 定义特征名称（替换为业务相关名称，与编码规则对应）\n",
    "feature_names = [\n",
    "    \"感染时程大于3个月\", \"骨暴露\", \"病变部位\", \"钙\", \"碱性磷酸酶\",\n",
    "    \"血浆纤维蛋白原\", \"C反应蛋白\"\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78a5f528-fac9-47e7-b411-8a473c7dd11e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ======================= 2. StreamLit页面配置 =======================\n",
    "st.set_page_config(page_title=\"骨髓炎复发风险预测器\", layout=\"wide\")\n",
    "st.title(\"骨髓炎复发风险预测器\")\n",
    "st.markdown(\"### 请填写以下信息，点击预测获取骨髓炎复发风险评估结果\")\n",
    "\n",
    "# ======================= 3. 特征输入组件（按编码规则设计） =======================\n",
    "# 1. 硬的食物（0：完全没问题，1：有问题）\n",
    "感染时程大于3个月 = st.selectbox(\n",
    "   \"感染时程大于3个月:\",\n",
    "    options=[0, 1],\n",
    "    format_func=lambda x: 'False (0)' if x == 0 else 'True (1)'\n",
    ")\n",
    "\n",
    "# 2. 睡眠时长（0：正常，1：异常）\n",
    "骨暴露 = st.selectbox(\n",
    "   \"骨暴露:\",\n",
    "    options=[0, 1],\n",
    "    format_func=lambda x: 'False (0)' if x == 0 else 'True (1)'\n",
    ")\n",
    "病变部位 = st.selectbox(\n",
    "   \"病变部位:\",\n",
    "    options=[0, 1],\n",
    "    format_func=lambda x: \"手足指（趾）\" if x == 0 else \"四肢骨\"\n",
    ")\n",
    "钙 = st.number_input(\"钙:\", min_value=0, max_value=1000, value=0)\n",
    "碱性磷酸酶 = st.number_input(\"碱性磷酸酶:\", min_value=0, max_value=1000, value=0)\n",
    "血浆纤维蛋白原 = st.number_input(\"血浆纤维蛋白原:\", min_value=0, max_value=1000, value=0)\n",
    "C反应蛋白 = st.number_input(\"C反应蛋白:\", min_value=0, max_value=1000, value=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f7fdf1f-b944-45ab-bd89-3f35c1ec6de5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ======================== 4. 数据处理与预测 ========================\n",
    "# 整合用户输入特征\n",
    "feature_values = [\n",
    "    Infection duration, Bone exposure, Lesion site, Ca, ALP,\n",
    "    Plasma fibrinogen, CRP\n",
    "]\n",
    "# 转换为模型输入格式\n",
    "features = np.array([feature_values])\n",
    "\n",
    "# 预测按钮逻辑\n",
    "if st.button(\"预测\"):\n",
    "    # 模型预测\n",
    "    predicted_class = model.predict(features)[0]  # 0: 低风险, 1: 高风险\n",
    "    predicted_proba = model.predict_proba(features)[0]  # 概率值\n",
    "\n",
    "    # 显示预测结果（中文适配）\n",
    "    st.subheader(\"预测结果\")\n",
    "    risk_label = \"高风险\" if predicted_class == 1 else \"低风险\"\n",
    "    st.write(f\"**风险等级：{predicted_class} ({risk_label})**\")\n",
    "    st.write(f\"**风险概率：** 低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}\")\n",
    "\n",
    "    # 生成个性化建议（中文）\n",
    "    st.subheader(\"💡 健康建议\")\n",
    "    probability = predicted_proba[predicted_class] * 100\n",
    "    if predicted_class == 1:\n",
    "        advice = (\n",
    "            f\"模型预测您的骨髓炎复发风险为高风险（概率{probability:.1f}%）。\"\n",
    "            \"建议优化临床决策，帮助改善患者预后\"\n",
    "        )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5b4eaec-0b26-44d8-9f9b-293d8bff62dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = shap.TreeExplainer(model)  \n",
    "shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names)) \n",
    "shap.force_plot(explainer.expected_value, shap_values, pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    \n",
    "plt.savefig(\"shap_force_plot.png\", bbox_inches='tight', dpi=1200)\n",
    "    st.image(\"shap_force_plot.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6da64bdc-6452-47a7-b763-40cfcb906a0e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20f5da3f-03d7-4138-847d-4a4777b19224",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
