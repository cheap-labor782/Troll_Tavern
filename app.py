import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.font_manager as fm

# ==========================================
# 0. 修正雲端 Linux 伺服器中文亂碼設定
# ==========================================
# 確認字型檔案是否存在（請確保 NotoSansTC-Regular.ttf 已上傳至 GitHub 專案根目錄）
font_path = "NotoSansTC-Regular.ttf"

if os.path.exists(font_path):
    # 載入自訂字型
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    # 註冊字型到 font manager 中
    fm.fontManager.addfont(font_path)
else:
    # 備用方案：如果找不到字型檔，依據系統設定
    import platform
    system_platform = platform.system()
    if system_platform == "Windows":
        plt.rcParams['font.family'] = ['Microsoft JhengHei']
    elif system_platform == "Darwin":
        plt.rcParams['font.family'] = ['Arial Unicode MS']

# 修正負號顯示問題
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(page_title="巨魔酒館", page_icon="🍺", layout="wide")
st.title("🍺 巨魔酒館")
st.markdown("""
歡迎來到巨魔酒館，路過的冒險者！這裡的三位**異族中介**將根據你的職業、身手與歷練，
為你預估這趟懸賞任務的**預期報酬**。請在下方羊皮紙上填寫你的傭兵履歷：
""")

# ==========================================
# 2. 載入模型與 Encoder
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

if encoder is not None:
    feature_names = ['jobLocationAt', 'jobCategory', 'jobCompanyIndustry', 'edu', 'jobRqYear', 'specialty', 'manageResp']
    categories_list = encoder.categories_
    
    # 強制轉成 list + sorted，防止 set 問題
    options = {}
    for i, name in enumerate(feature_names):
        cat = categories_list[i]
        if isinstance(cat, (set, np.ndarray)):
            cat = list(cat)
        options[name] = sorted(cat)
else:
    st.error("❌ 找不到 encoder.jb，請先執行訓練腳本！")
    st.stop()

# ==========================================
# 3. 用戶輸入
# ==========================================
st.header("📜 填寫你的羊皮紙")
col1, col2 = st.columns(2)

with col1:
    loc_options = options['jobLocationAt']
    default_idx = loc_options.index("台北市") if "台北市" in loc_options else 0
    location_selected = st.selectbox("1. 任務地區", loc_options, index=default_idx)
    
    job_selected = st.selectbox("2. 職業", options['jobCategory'])
    industry_selected = st.selectbox("3. 雇主陣營", options['jobCompanyIndustry'])
    exp_selected = st.selectbox("4. 歷練年份", options['jobRqYear'])

with col2:
    edu_selected = st.selectbox("5. 魔法學院學位", options['edu'], 
                                index=options['edu'].index("大學") if "大學" in options['edu'] else 0)
    manage_selected = st.selectbox("6. 統御能力", options['manageResp'])
    specialty_selected = st.selectbox("7. 掌握的特殊符文/公會技能", options['specialty'])

# ==========================================
# 4. 預測 + 評語 + 小遊戲
# ==========================================
st.divider()
st.header("💰 異族中介估價")

def show_analysis_charts():
    st.subheader("📊 市場分析報告")
    
    # 1. 薪資區間圖
    st.markdown("**🔹 品管／品保工程師 市場行情**")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    categories = ['最低薪資下限', '市場平均行情', '最高薪資上限']
    values = [40000, 50000, 60000]
    colors = ['#8c8c8c', '#ff7f0e', '#2c3e50']
    bars = ax1.bar(categories, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1000, f'NT${height:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('新台幣 (元)')
    st.pyplot(fig1)
    
    # 2. 地區薪資分布
    st.markdown("**🔹 地理區域薪資行情分布**")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    regions = ['南投縣', '台北市', '嘉義市', '彰化縣', '新竹市', '苗栗縣', '高雄市']
    salaries = [48000, 72000, 51000, 52000, 55000, 41000, 53000]
    ax2.plot(regions, salaries, marker='o', linewidth=3, color='#ff7f0e')
    ax2.set_ylabel('平均月薪 (NT$)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, v in enumerate(salaries):
        ax2.text(i, v + 1000, f'{v:,}', ha='center')
    st.pyplot(fig2)
    
    # 3. 特徵重要性
    st.markdown("**🔹 影響薪資的核心因素 (CatBoost)**")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    features = ['jobRqYear', 'jobCategory', 'clean_location', 'job_cluster_label']
    importance = [31.4, 24.7, 23.8, 20.1]
    ax3.barh(features[::-1], importance[::-1], color='#ff7f0e')
    for i, v in enumerate(importance[::-1]):
        ax3.text(v + 0.5, i, f'{v}%', va='center', fontweight='bold')
    ax3.set_xlabel('權重影響力 (%)')
    st.pyplot(fig3)

available_brokers = {k: v for k, v in res.items() if k != "encoder"}

if available_brokers:
    chosen_broker_name = st.selectbox("請選擇為你引薦任務的中介", list(available_brokers.keys()))
    
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'predicted_salary' not in st.session_state:
        st.session_state.predicted_salary = 0.0
    if 'used_broker' not in st.session_state:
        st.session_state.used_broker = ""
    if 'gold' not in st.session_state:
        st.session_state.gold = 1000 
    if 'bargain_result' not in st.session_state:
        st.session_state.bargain_result = None
   
    if st.button("⚔️ 送出羊皮紙，評估預期酬勞！"):
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
       
        if "CatBoost" in chosen_broker_name:
            final_input = input_df.astype(str)
        else:
            encoded_values = encoder.transform(input_df)
            final_input = pd.DataFrame(encoded_values, columns=correct_order)
           
        with st.spinner("正在引薦任務..."):
            try:
                model_to_use = available_brokers[chosen_broker_name]
                prediction = model_to_use.predict(final_input)
                
                st.session_state.predicted_salary = float(prediction[0] if hasattr(prediction, "__len__") else prediction)
                st.session_state.used_broker = chosen_broker_name
                st.session_state.prediction_done = True
                st.session_state.bargain_result = None  
                st.balloons()
            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")

    if st.session_state.prediction_done:
        salary = st.session_state.predicted_salary
        broker_key = st.session_state.used_broker

        st.markdown(f"### 🎯 {broker_key} 給出的評估結論：")
        st.success(f"這趟任務的預估月薪報酬為： **{int(salary):,} 元金幣** / 月")

        comments = {
            "獸人中介 (XGBoost)": {"low": "吼！！這點金幣連喝麥酒都不夠！去多殺幾隻哥布林再來吧！", "mid": "還算能打！夠你買把好斧頭和幾桶烈酒了。", "high": "不錯！這趟懸賞夠猛！獸人戰士就該拿這種酬勞！", "top": "哇哈哈哈！！這是王者級的懸賞！整個部落都會羨慕你！"},
            "精靈中介 (LightGBM)": {"low": "……這報酬實在太微薄了，連森林裡的松鼠都看不上。", "mid": "優雅而適中，符合一位有品位的冒險者。", "high": "相當不錯的報酬，古樹也會為你感到欣慰。", "top": "星辰與月光都為你閃耀！這是傳奇般的優渥懸賞！"},
            "矮人中介 (CatBoost)": {"low": "哼，這點金幣連買一桶好麥酒都不夠！去重鑄你的斧頭吧小子！", "mid": "還算厚道，能買幾件好盔甲和喝個痛快。", "high": "這才像話！夠你打造一把傳世武器了！", "top": "以我祖先的鬍子發誓！這是金山般的酬勞！！"}
        }
        
        if salary < 45000:
            comment = comments[broker_key]["low"]
            color = "🔴"
        elif salary < 70000:
            comment = comments[broker_key]["mid"]
            color = "🟡"
        elif salary < 100000:
            comment = comments[broker_key]["high"]
            color = "🟢"
        else:
            comment = comments[broker_key]["top"]
            color = "🌟"
        
        st.markdown(f"**{color} {broker_key}的酒館評語：** {comment}")

        # ====================== 酒館小遊戲 ======================
        st.divider()
        st.subheader("🪙 酒館黑市交易")
        
        st.info(f"目前你身上有 **{st.session_state.gold} 金幣**")
        st.write("**異族中介願意提供更深入的市場分析報告（含三張圖表），但要價 500 金幣。**")
        
        col_pay, col_bargain = st.columns(2)
        
        with col_pay:
            if st.button("💰 支付 500 金幣 取得完整報告", type="primary", key="pay_btn"):
                if st.session_state.gold >= 500:
                    st.session_state.gold -= 500
                    st.session_state.bargain_result = "paid"
                else:
                    st.session_state.bargain_result = "no_money"
        
        with col_bargain:
            if st.button("🗣️ 討價還價（有風險）", key="bargain_btn"):
                if random.random() < 0.30:
                    st.session_state.bargain_result = "success"
                else:
                    st.session_state.bargain_result = "fail"
        
        if st.session_state.bargain_result == "paid":
            st.success("交易成功！以下是詳細市場分析報告：")
            show_analysis_charts()
            
        elif st.session_state.bargain_result == "no_money":
            st.error("😵 你身上沒有足夠的金幣！中介嘲笑你：「窮鬼冒險者也敢來談生意？」")
            
        elif st.session_state.bargain_result == "success":
            st.success("🎉 討價還價成功！中介被你說服，免費提供報告！")
            show_analysis_charts()
            
        elif st.session_state.bargain_result == "fail":
            refusals = {
                "獸人中介 (XGBoost)": "「吼！！想白嫖？信不信我一斧頭砍死你！」",
                "精靈中介 (LightGBM)": "「凡人，你太貪婪了……森林不會庇護這種行為。」",
                "矮人中介 (CatBoost)": "「小子！想空手套白狼？先去礦坑挖夠 500 金幣再來！」"
            }
            st.warning(f"❌ {broker_key} 怒道：{refusals.get(broker_key, '中介拒絕了你的要求！')}")

else:
    st.warning("⚠️ 酒館內尚無中介就位，請先確認模型檔案是否存在。")
