// src/App.tsx
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { ConfigProvider } from "antd";
import zhTW from "antd/locale/zh_TW"; 

import Sidebar from "./components/layout/Sidebar";

import RecommendPage from "./routes/RecommendPage";
import WatchlistPage from "./routes/WatchlistPage";
import StockHistoryPage from "./routes/StockHistoryPage";
import BacktestPage from "./routes/BacktestPage";
import ModelInfoPage from "./routes/ModelInfoPage";
import GraphPage from "./routes/GraphPage";

function App() {
  return (
    <ConfigProvider
      locale={zhTW}
      theme={{
        token: {
          colorPrimary: "#00bcc2", 
          borderRadius: 6,
          fontSize: 14,
        },
      }}
    >
    <Router>
      <div style={{ display: "flex", width: "100vw" }}>
        <Sidebar/>
        <div style={{ flex: 1, padding: 12, overflow:"auto"}}>
          <Routes>
            <Route path="/" element={<Navigate to="/recommend" replace />} />
            <Route path="/recommend" element={<RecommendPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/history/:stockId" element={<StockHistoryPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/model" element={<ModelInfoPage />} />
            <Route path="/graph" element={<GraphPage />} />
          </Routes>
        </div>
      </div>
    </Router>
    </ConfigProvider>
  );
}

export default App;
