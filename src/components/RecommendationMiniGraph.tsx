import React, { useState, useRef, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import {Button, Modal } from "antd";
import ResizeObserver from "resize-observer-polyfill";

// 節點與邊資料格式
interface StockNode {
  id: string;
  name: string;
  central?: boolean; // 是否為被推薦股票
}

interface StockLink {
  source: string;
  target: string;
  weight: number; // 可用來決定線的粗細
}

interface RecommendationMiniGraphProps {
  stock: StockNode; // 被推薦股票
  neighbors: { node: StockNode; link: StockLink }[]; // 鄰居影響來源
  triggerText?: string; // Modal 觸發按鈕文字
}

const GraphContainer = ({ children }: { children: (width: number, height: number) => React.ReactNode }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      for (let entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    if (containerRef.current) {
      ro.observe(containerRef.current);
    }
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative" }}
    >
      {dimensions.width > 0 && dimensions.height > 0 && children(dimensions.width, dimensions.height)}
    </div>
  );
};

const RecommendationMiniGraph: React.FC<RecommendationMiniGraphProps> = ({ stock, neighbors, triggerText = "查看影響圖" }) => {
  const [open, setOpen] = useState(false);
  const nodes: StockNode[] = [
    stock,
    ...neighbors.map((n) => n.node),
  ];

  const links: StockLink[] = neighbors.map((n) => n.link);
  const GraphCanvas = (
    <div style={{ width: "100%", height: "100%"}}>
      <GraphContainer>
        {(width, height) => (
          <ForceGraph2D
            width={width}
            height={height}
            graphData={{ nodes, links }}
            nodeLabel={(node: any) => node.name}
            nodeAutoColorBy="id"
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.name;
              const fontSize = 12 / globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.fillStyle = node.central ? "#f5222d" : "#555";
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, node.central ? 6 : 4, 0, 2 * Math.PI, false);
              ctx.fill();
              ctx.fillText(label, node.x! + 8, node.y! + 4);
            }}
            linkWidth={(link) => (link as StockLink).weight * 4}
            linkDirectionalArrowLength={6}
            linkDirectionalArrowRelPos={1}
            linkDirectionalParticles={1}
            linkDirectionalParticleSpeed={0.005}
          />
        )}
      </GraphContainer>
    </div>
  );
  return (
    <>
      <Button type="link" onClick={() => setOpen(true)}>
        {triggerText}
      </Button>

      <Modal
        open={open}
        onCancel={() => setOpen(false)}
        footer={null}
        title={`推薦來源：${stock.name}`}
        width={600}
      >
        <div style={{ width: "100%", height: 400 }}>
          {GraphCanvas}
        </div>
      </Modal>
    </>
  );
};

export default RecommendationMiniGraph;

// ✅ 使用範例：
// <RecommendationMiniGraph
//   stock={{ id: "2330", name: "台積電", central: true }}
//   neighbors={[
//     { node: { id: "2454", name: "聯發科" }, link: { source: "2454", target: "2330", weight: 0.6 } },
//     { node: { id: "2317", name: "鴻海" }, link: { source: "2317", target: "2330", weight: 0.4 } },
//   ]}
// />