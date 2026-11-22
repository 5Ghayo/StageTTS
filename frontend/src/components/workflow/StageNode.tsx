import React, { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { StageFlowNode } from "../../store/useWorkflowStore";
import type { EnvBadge, NodeCategory, NodeRunStatus } from "../../types/workflow";

interface CategoryTheme {
  headerBg: string;
  bodyBg: string;
  footerBg: string;
  border: string;
  selectedBorder: string;
  accent: string;
  badgeBg: string;
  badgeText: string;
}

/**
 * ComfyUI 风格低饱和哑光色卡（铺满整张节点背景，与 #161616 画布及 #222222 面板自然契合）。
 */
const CATEGORY_THEME: Record<NodeCategory, CategoryTheme> = {
  input: {
    headerBg: "#312624",
    bodyBg: "#28201E",
    footerBg: "#211B19",
    border: "rgba(182, 134, 122, 0.22)",
    selectedBorder: "#B08276",
    accent: "#B08276",
    badgeBg: "rgba(182, 134, 122, 0.14)",
    badgeText: "#CCB0A8",
  },
  parse: {
    headerBg: "#2E2632",
    bodyBg: "#251F28",
    footerBg: "#1F1A22",
    border: "rgba(156, 130, 168, 0.22)",
    selectedBorder: "#9A80A6",
    accent: "#9A80A6",
    badgeBg: "rgba(156, 130, 168, 0.14)",
    badgeText: "#C2B3C9",
  },
  hub: {
    headerBg: "#312C21",
    bodyBg: "#28241C",
    footerBg: "#211E17",
    border: "rgba(182, 154, 100, 0.24)",
    selectedBorder: "#B59A64",
    accent: "#B59A64",
    badgeBg: "rgba(182, 154, 100, 0.14)",
    badgeText: "#CCBEA0",
  },
  synthesis: {
    headerBg: "#223332",
    bodyBg: "#1D2A29",
    footerBg: "#182322",
    border: "rgba(102, 158, 148, 0.24)",
    selectedBorder: "#649C91",
    accent: "#649C91",
    badgeBg: "rgba(102, 158, 148, 0.14)",
    badgeText: "#A6C7C0",
  },
  output: {
    headerBg: "#232E38",
    bodyBg: "#1D262E",
    footerBg: "#182027",
    border: "rgba(112, 152, 188, 0.24)",
    selectedBorder: "#6D94B6",
    accent: "#6D94B6",
    badgeBg: "rgba(112, 152, 188, 0.14)",
    badgeText: "#AEC0D1",
  },
};

const ENV_BADGE_MAP: Record<EnvBadge, string> = {
  env_parse: "解析",
  env_tts: "合成",
  env_backend: "后端",
  frontend: "舞台",
};

const STATUS_DOT: Record<NodeRunStatus, string> = {
  idle: "bg-muted",
  queued: "bg-status-warning",
  running: "bg-status-ready animate-pulse",
  done: "bg-status-ready",
  error: "bg-status-error",
};

const PORT_COLOR: Record<string, string> = {
  MEDIA: "!bg-[#B08276] !border-[#1C1C1C]",
  AUDIO: "!bg-[#9A80A6] !border-[#1C1C1C]",
  "AUDIO[]": "!bg-[#649C91] !border-[#1C1C1C]",
  SEGMENTS: "!bg-[#A58BB0] !border-[#1C1C1C]",
  JSON: "!bg-[#B59A64] !border-[#1C1C1C]",
  SCRIPT: "!bg-[#B59A64] !border-[#1C1C1C]",
  MP4: "!bg-[#6D94B6] !border-[#1C1C1C]",
  CANVAS: "!bg-[#6D94B6] !border-[#1C1C1C]",
};

/**
 * ComfyUI 风格低饱和着色管线节点（铺色卡片底 + 柔和微对比顶底栏）。
 */
export const StageNode: React.FC<NodeProps<StageFlowNode>> = memo(
  ({ data, selected }) => {
    const theme = CATEGORY_THEME[data.category];
    const envLabel = ENV_BADGE_MAP[data.env];
    const rowCount = Math.max(data.inputs.length, data.outputs.length, 1);

    return (
      <div
        style={{
          backgroundColor: theme.bodyBg,
          borderColor:
            data.status === "running"
              ? "#4FA882"
              : selected
                ? theme.selectedBorder
                : theme.border,
        }}
        className="w-[200px] overflow-hidden rounded-[6px] border transition-colors"
      >
        {/* 顶栏（低饱和同色系稍亮头栏 + 状态点） */}
        <div
          style={{ backgroundColor: theme.headerBg }}
          className="flex items-center justify-between border-b border-white/[0.06] px-2.5 py-1.5"
        >
          <div className="flex min-w-0 items-center gap-1.5">
            <span
              className={`h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[data.status]}`}
            />
            <span className="truncate text-[12px] font-medium text-primary">
              {data.title}
            </span>
          </div>
          <span className="ml-2 shrink-0 rounded-[4px] bg-black/25 px-1.5 py-0.5 font-mono tabular-nums text-[10px] text-secondary">
            {data.vramGb === null ? "—" : data.vramGb > 0 ? `${data.vramGb.toFixed(1)}G` : "CPU"}
          </span>
        </div>

        {/* 左右端口区（承接整卡低饱和铺色底） */}
        <div className="space-y-1.5 py-2 text-[11px]">
          {Array.from({ length: rowCount }).map((_, idx) => {
            const inputPort = data.inputs[idx];
            const outputPort = data.outputs[idx];
            return (
              <div
                key={idx}
                className="relative flex min-h-[19px] items-center justify-between px-3"
              >
                {inputPort ? (
                  <div className="flex items-center gap-1.5">
                    <Handle
                      id={inputPort.id}
                      type="target"
                      position={Position.Left}
                      className={`!h-2 !w-2 !border ${PORT_COLOR[inputPort.type] ?? "!bg-[#888888] !border-[#1C1C1C]"}`}
                    />
                    <span className="text-secondary">{inputPort.label}</span>
                  </div>
                ) : (
                  <span />
                )}

                {outputPort ? (
                  <div className="flex items-center gap-1.5 text-right">
                    <span className="text-primary">{outputPort.label}</span>
                    <Handle
                      id={outputPort.id}
                      type="source"
                      position={Position.Right}
                      className={`!h-2 !w-2 !border ${PORT_COLOR[outputPort.type] ?? "!bg-[#888888] !border-[#1C1C1C]"}`}
                    />
                  </div>
                ) : (
                  <span />
                )}
              </div>
            );
          })}
        </div>

        {/* 底部摘要与同色系低饱和环境标识 */}
        <div
          style={{ backgroundColor: theme.footerBg }}
          className="flex items-center justify-between gap-2 border-t border-white/[0.05] px-2.5 py-1.5 text-[10px]"
        >
          <span className="truncate text-secondary" title={data.summary}>
            {data.summary}
          </span>
          <span
            style={{
              backgroundColor: theme.badgeBg,
              color: theme.badgeText,
            }}
            className="shrink-0 rounded-[4px] px-1.5 py-0.5 font-medium"
          >
            {envLabel}
          </span>
        </div>
      </div>
    );
  }
);

StageNode.displayName = "StageNode";
