# 🍺 巨魔酒館 (Troll Tavern) — 機器學習薪資估價預測系統

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://trolltavern-3wggekgwpkrkj44w3rk5rh.streamlit.app/)

歡迎來到巨魔酒館！這是一個基於 Python 與 Streamlit 開發的機器學習 Web 應用程式。
酒館內駐紮了三位脾氣迥異的**異族中介**（XGBoost、LightGBM、CatBoost），他們將根據冒險者（求職者）填寫的傭兵履歷，即時評估並預估該趟懸賞任務的**預期月薪報酬**。

🎯 **線上展示網址**：[點此進入巨魔酒館](https://trolltavern-3wggekgwpkrkj44w3rk5rh.streamlit.app/)

---

## 👥 異族中介（模型）特性介紹

在酒館中，你可以選擇不同陣營的中介為你估價，每位中介背後都代表了一種主流的梯度提升樹（Gradient Boosting Tree）演算法：

1. **獸人中介 (XGBoost)** 🟠
   * **內部特性**：採用母體共鳴機制處理類別型態。
   * **酒館八卦**：底層屬於極度老實的連續數值數學公式預測器。因為太過老實且缺乏人類社會的常識，當遇到極端或矛盾的履歷組合時，偶爾會陷入邏輯混亂而給出「負數」金幣（月薪），被酒館冒險者們戲稱為「喜歡胡說八道」。
2. **精靈中介 (LightGBM)** 🔵
   * **內部特性**：同樣基於母體結構進行特徵對齊，預測速度極快。
   * **酒館八卦**：心思細膩，對特徵結構（Categorical Features）的匹配要求極度嚴苛，不容許任何一絲型態錯位。
3. **矮人中介 (CatBoost)** 🟢
   * **內部特性**：天生具備強大的純字串（String）雜湊與編碼機制。
   * **酒館八卦**：脾氣最穩健，不看 Pandas 的臉色，只認文字內容。在幾次型態地獄的大戰中，是唯一自始至終都能完美通關、給出漂亮數字的硬漢。

---

## 🛠️ 預測特徵說明 (傭兵履歷)

系統依據以下 7 大核心特徵進行綜合薪資估算：
* `jobLocationAt`：任務地區（如：台北市、新北市等縣市）
* `jobCategory`：具體專業職位
* `jobCompanyIndustry`：雇主陣營（公司產業）
* `edu`：魔法學院學位（教育程度）
* `jobRqYear`：歷練年份（工作經驗要求）
* `specialty`：掌握的特殊符文/公會技能（專業專長）
* `manageResp`：統御能力（是否管理職）

---

## 📁 專案架構與檔案說明

```text
├── app.py               # Streamlit 網頁主程式 (包含前端 UI 與後台多模型分流預測邏輯)
├── requirements.txt     # 線上雲端伺服器所需的環境套件清單
├── study.csv            # 原始訓練數據集 (用於網頁下拉選單生成與類別型態共鳴對齊)
├── xg.jb                # 重新訓練/匯出的 XGBoost 模型檔案
├── lig.jb               # 重新訓練/匯出的 LightGBM 模型檔案
└── cat.jb               # 重新訓練/匯出的 CatBoost 模型檔案
