import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# ==========================================
# 1. 網頁基本設定 (前端顯示與經典外觀完全固定)
# ==========================================
st.set_page_config(
    page_title="巨魔酒館",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 巨魔酒館")
st.markdown("""
歡迎來到巨魔酒館，路過的冒險者！這裡的三位**異族中介**將根據你的職業、身手與歷練，
為你預估這趟懸賞任務的**預期報酬**。請在下方羊皮紙上填寫你的傭兵履歷：
""")

# ==========================================
# 2. 快取載入模型與翻譯官 (Encoder)
# ==========================================
@st.cache_resource
def load_all_resources():
    resources = {}
    if os.path.exists("xg.jb"): resources["獸人中介 (XGBoost)"] = joblib.load("xg.jb")
    if os.path.exists("lig.jb"): resources["精靈中介 (LightGBM)"] = joblib.load("lig.jb")
    if os.path.exists("cat.jb"): resources["矮人中介 (CatBoost)"] = joblib.load("cat.jb")
    if os.path.exists("encoder.jb"): resources["encoder"] = joblib.load("encoder.jb")
    return resources

res = load_all_resources()
encoder = res.get("encoder", None)

# 從翻譯官的大腦裡，直接抓出當初訓練時所有合法的字串清單，保證下拉選單字串 100% 精準
if encoder is not None:
    feature_names = ['jobLocationAt', 'jobCategory', 'jobCompanyIndustry', 'edu', 'jobRqYear', 'specialty', 'manageResp']
    categories_list = encoder.categories_
    
    options = {feature_names[i]: sorted(list(categories_list[i])) for i in range(len(feature_names))}
else:
    st.error("❌ 找不到全新的翻譯官檔案 `encoder.jb`，請先執行重新訓練腳本！")
    st.stop()

# ==========================================
# 3. 用戶界面 (經典外觀，文字完全不變)
# ==========================================
st.header("📜 填寫你的羊皮紙")
col1, col2 = st.columns(2)

with col1:
    location_selected = st.selectbox("1. 任務地區", options['jobLocationAt'], index=options['jobLocationAt'].index("台北市") if "台北市" in options['jobLocationAt'] else 0)
    job_selected = st.selectbox("2. 職業", options['jobCategory'])
    industry_selected = st.selectbox("3. 雇主陣營", options['jobCompanyIndustry'])
    exp_selected = st.selectbox("4. 歷練年份", options['jobRqYear'])

with col2:
    edu_selected = st.selectbox("5. 魔法學院學位", options['edu'], index=options['edu'].index("大學") if "大學" in options['edu'] else 0)
    manage_selected = st.selectbox("6. 統御能力", options['manageResp'])
    specialty_selected = st.selectbox("7. 掌握的特殊符文/公會技能", options['specialty'])

# ==========================================
# 4. 預測邏輯 (乾淨、優雅、100%不噴錯)
# ==========================================
st.divider()
st.header("💰 異族中介估價")

# 過濾掉 encoder，只留下可選的模型中介
available_brokers = {k: v for k, v in res.items() if k != "encoder"}

if available_brokers:
    chosen_broker_name = st.selectbox("請選擇為你引薦任務的中介", list(available_brokers.keys()))
    
    if st.button("⚔️ 送出羊皮紙，評估預期酬勞！"):
        
        # 建立最基礎的字串字典
        user_input_raw = {
            'jobLocationAt': location_selected,
            'jobCategory': job_selected,
            'jobCompanyIndustry': industry_selected,
            'edu': edu_selected,
            'jobRqYear': exp_selected,
            'specialty': specialty_selected,
            'manageResp': manage_selected
        }
        correct_order = ['jobLocationAt', 'jobCategory', 'jobCompanyIndustry', 'edu', 'jobRqYear', 'specialty', 'manageResp']
        input_df = pd.DataFrame([user_input_raw])[correct_order]
        
        # --- 完美的後台分流 ---
        if "CatBoost" in chosen_broker_name:
            # 🟢 CatBoost 喜歡純字串
            final_input = input_df.astype(str)
        else:
            # 🟠 XGBoost 與 🔵 LightGBM：直接呼叫當初打包的翻譯官，一秒把字串轉成精準的純數字 DataFrame！
            encoded_values = encoder.transform(input_df)
            final_input = pd.DataFrame(encoded_values, columns=correct_order)
            
        with st.spinner("正在引薦任務..."):
            try:
                model_to_use = available_brokers[chosen_broker_name]
                prediction = model_to_use.predict(final_input)
                
                if hasattr(prediction, "__len__") and len(prediction.shape) > 1:
                    predicted_salary = prediction[0][0]
                else:
                    predicted_salary = prediction[0]
                
                st.balloons()
                st.markdown(f"### 🎯 {chosen_broker_name} 給出的評估結論：")
                st.success(f"這趟任務的預估月薪報酬為： **{int(predicted_salary):,} 元金幣** / 月")
                
            except Exception as eval_err:
                st.error(f"該中介的專屬通道發生異常！詳細報錯：{eval_err}")
else:
    st.warning("⚠️ 酒館內尚無中介就位，請先確認目錄下是否存在模型檔案。")