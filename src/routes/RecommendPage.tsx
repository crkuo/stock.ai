import { useState } from "react";
import { Card, Table, Segmented, theme} from "antd";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import RecommendationMiniGraph from "../components/RecommendationMiniGraph";

const mockData = [
  {
    symbol: "2330",
    name: "台積電",
    score: 92,
    confidence: 0.87,
    reason_neighbors: [
      { node: { id: "2454", name: "聯發科" }, link: { source: "2454", target: "2330", weight: 0.6 } },
      { node: { id: "2317", name: "鴻海" }, link: { source: "2317", target: "2330", weight: 0.4 } },
    ],
  },
  {
    symbol: "2317",
    name: "鴻海",
    score: 88,
    confidence: 0.84,
    reason_neighbors: [
      { node: { id: "2303", name: "聯電" }, link: { source: "2303", target: "2317", weight: 0.5 } },
    ],
  },
];

export default function RecommendPage() {
  const [period, setPeriod] = useState<"day" | "week">("day");
  const { token } = theme.useToken(); 

  const columns = [
    { title: "股票代號", dataIndex: "symbol" },
    { title: "股票名稱", dataIndex: "name" },
    { title: "推薦分數", dataIndex: "score" },
    {
      title: "信心指數",
      dataIndex: "confidence",
      render: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
    {
      title: "推薦來源",
      dataIndex: "reason_neighbors",
      render: (_: any, record: any) => (
        <RecommendationMiniGraph
          stock={{ id: record.symbol, name: record.name, central: true }}
          neighbors={record.reason_neighbors}
        />
      ),
    },
  ];

  return (
    <div style={{ width: "100%", height: "100%", padding: 24, boxSizing: "border-box" }}>
      <h2>AI 股票推薦</h2>

      <Segmented
        options={[
          { label: "今日推薦", value: "day" },
          { label: "本週推薦", value: "week" },
        ]}
        value={period}
        onChange={(value) => setPeriod(value as "day" | "week")}
        style={{ marginBottom: 24 }}
      />

      <Card title="Top 股票推薦（依推薦分數）" style={{ marginBottom: 24 }}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart layout="vertical" data={mockData}>
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="name" />
            <Tooltip />
            <Bar dataKey="score" fill= {token.colorPrimary} barSize={40}/>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="推薦詳情表格">
        <Table
          dataSource={mockData}
          rowKey="symbol"
          columns={columns}
          pagination={false}
        />
      </Card>
    </div>
  );
}
