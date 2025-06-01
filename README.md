# 股票 AI 推薦系統 Dashboard

本專案為一個結合 Graph Neural Network 與推薦系統的股票 AI Dashboard，目的是提供每日與未來區間的推薦股票，並以圖像化與資料視覺化方式呈現。

---

## 📌 專案目標

* 對每一支股票建立 Graph Node 表示（GNN）
* 利用鄰接股票的價格變動進行推理推薦
* 提供投資人或一般使用者每日可參考的股票清單
* 提供推薦解釋與歷史績效查詢

---

## ✅ 已完成內容

### 1. 網站結構

* [V] SPA 應用架構（React + Vite + Ant Design v5）
* [V] 設計全站路由與側邊導覽列
* [x] 使用 Tailwind + AntD v5 token 實現主色 `#00bcc2`

### 2. 頁面結構

* [V] `/recommend`：今日推薦清單，含推薦理由、信心指數與可視化
* [x] `/model`：展示模型訓練紀錄、Log、Hyperparam 與版本
* [x] `/watchlist`：使用者收藏追蹤清單
* [x] `/graph`：完整股票圖（GNN）視覺化呈現
* [x] `MiniGraph` 模組化：可內嵌或以 Modal 互動展示推薦來源節點

### 3. 可視化模組

* [x] 使用 `react-force-graph` 動態繪製 GNN 節點關係圖
* [x] 可縮放、拖曳與 Tooltip（label）互動
* [x] 導入 `ResizeObserver` 支援自動寬高調適

---

## 🚧 開發中

### `/history/:stockId` 頁面（個股歷史推薦與績效分析）

* [ ] 顯示基本資訊（名稱、代碼、產業）
* [ ] 折線圖顯示歷史股價與 AI 推薦時間點
* [ ] 過去推薦紀錄列表（含信心值）
* [ ] 每筆推薦紀錄可展開 MiniGraph 顯示來源節點

---

## 🔜 接下來的規劃

* [ ] 收藏機制（使用者可加入 watchlist）
* [ ] 搜尋功能（依名稱或代碼快速篩選）
* [ ] 後端 API 串接與資料快取
* [ ] 支援 Responsive Design

---

## 📦 技術棧

* Frontend: `React`, `Vite`, `TypeScript`
* UI: `Ant Design v5`, `TailwindCSS`
* Chart: `Recharts`, `react-force-graph`
* Style: CSS-in-JS + ConfigProvider token

---

如需貢獻、測試或部署協助，歡迎隨時聯繫開發者 🙌
