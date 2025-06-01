# 📊 Stock AI Recommendation Dashboard

本專案旨在建立一個股票 AI 推薦系統的 Dashboard，整合 Graph Neural Network 模型的推論結果、推薦理由與視覺化圖表，並具備互動性與可擴充性，供履歷展示與投資參考使用。

---

## ✅ 專案進度摘要

* [x] 專案頁面結構規劃完成
* [x] 路由與導覽列架構完成
* [x] 主色調設定為 `#00bcc2`（Ant Design v5 + Vite）
* [x] Recommendation Mini Graph 模組完成，支援 Modal 與 inline 顯示，並自適應寬高
* [x] 推薦頁 (`/recommend`) 基礎畫面與資料結構設計完成
* [x] Watchlist 頁面 (`/watchlist`) 設計確認
* [x] Graph 可視化頁 (`/graph`) 確認使用 force graph
* [x] 導入 RWD 設計慣例，統一寬度以適應容器大小
* [ ] `/history/:stockId` 頁面待開發（個股推薦歷史、MiniGraph 顯示）
* [ ] `/watchlist` 收藏功能尚未實作（需資料儲存設計）
* [ ] `/backtest` 回測分析頁面尚未開發（含策略回測與報酬視覺化）
* [ ] `/model` 頁面部分功能（Log 顯示與圖表視覺化）尚未整合
* [ ] `/graph` 圖形頁面初版完成，但互動功能需優化（zoom/pan/tooltip）
* [ ] 模型部署推論 API 串接中（資料格式與模型結果結構尚未最終確認）
* [ ] 用戶自訂偏好（產業類別、自選股）功能待討論
* [ ] 後端 API 結構與推論結果格式待確認

---

## 📁 頁面功能列表與狀態

| 頁面                  | 說明                   | 討論完成 | 已實作     |
| ------------------- | -------------------- | ---- | ------- |
| `/recommend`        | 顯示每日推薦股票，包含信心指數與推薦圖  | ✅    | 🔄 部分完成 |
| `/watchlist`        | 顯示使用者收藏股票與最近推薦紀錄     | ✅    | ⏳ 未實作   |
| `/history/:stockId` | 顯示個股歷史走勢、推薦紀錄、推薦來源圖  | ✅    | ⏳ 待開發   |
| `/backtest`         | 未來加入模型回測分析介面         | ❌    | ⏳ 尚未開始  |
| `/model`            | 顯示模型訓練資訊、訓練 log、效能圖表 | ✅    | 🔄 部分完成 |
| `/graph`            | 顯示整體股票圖結構與推薦邊緣權重     | ✅    | ✅ 初步完成  |

---

## 🔧 技術棧與套件

* 前端：React 18 + Vite
* UI 套件：Ant Design v5（採用 CSS-in-JS 與 Token 主題系統）
* 圖表：Recharts（歷史價格）、react-force-graph（推薦來源圖）
* 資料來源：每日更新 TWSE 股價與模型推薦結果
* 預期後端：FastAPI + GNN 模型服務（尚未接入）

---

## ✅ 待辦事項（按優先順序）

1. [ ] 完成 `/history/:stockId` 頁面，顯示推薦歷史與影響來源圖
2. [ ] 實作「加入收藏」功能並同步於 watchlist
3. [ ] 串接推論 API 並設計對應資料格式與快取策略
4. [ ] 將 force-graph 圖做進一步互動優化（縮放、hover 細節）
5. [ ] 若時間充裕，新增登入系統與使用者偏好設定（例如產業偏好）

---

> 本 README 將持續隨開發進度更新，若有設計與結構變更，請同步修改本說明。
