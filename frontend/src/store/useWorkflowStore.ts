import {
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  applyEdgeChanges,
  applyNodeChanges,
} from "@xyflow/react";
import { create } from "zustand";
import {
  type ParseStatusResponse,
  fetchParseStatus,
  parseProjectMedia,
  renderVideo,
  saveProjectScript,
  subscribeParseStatus,
  synthesizeSegments,
} from "../services/api";
import type {
  CenterViewTab,
  LeftDrawerTab,
  NodeRunStatus,
  PipelineConfig,
  PipelineLogEntry,
  RightPanelTab,
  StageNodeData,
  WorkspacePreset,
} from "../types/workflow";
import { synthesisIssues } from "../utils/synthesisIssues";
import { useProjectStore } from "./useProjectStore";

export type StageFlowNode = Node<StageNodeData, "stageNode">;

function createInitialNodes(config: PipelineConfig): StageFlowNode[] {
  const parseVram = config.computeType === "int8" ? 3.4 : 5.2;
  return [
    {
      id: "media_input",
      type: "stageNode",
      position: { x: 20, y: 120 },
      data: {
        title: "媒体输入",
        category: "input",
        env: "env_backend",
        vramGb: 0,
        status: "done",
        inputs: [],
        outputs: [{ id: "out-video", label: "原音视频", type: "MEDIA" }],
        summary: "导入视频或音频",
        params: { format: "mp4 / wav" },
      },
    },
    {
      id: "banquet",
      type: "stageNode",
      position: { x: 260, y: 28 },
      data: {
        title: "音轨分离",
        category: "parse",
        env: "env_parse",
        vramGb: 1.8,
        status: "done",
        inputs: [{ id: "in-media", label: "媒体流", type: "MEDIA" }],
        outputs: [
          { id: "out-vocals", label: "对白轨", type: "AUDIO" },
          { id: "out-bgm", label: "背景轨", type: "AUDIO" },
        ],
        summary: "Banquet 分离对白、歌声、伴奏和音效",
        params: { model: "banquet" },
      },
    },
    {
      id: "whisperx",
      type: "stageNode",
      position: { x: 500, y: 18 },
      data: {
        title: "语音识别与聚类",
        category: "parse",
        env: "env_parse",
        vramGb: parseVram,
        status: "done",
        inputs: [{ id: "in-vocals", label: "对白轨", type: "AUDIO" }],
        outputs: [{ id: "out-turns", label: "角色时间段", type: "SEGMENTS" }],
        summary: `${config.computeType === "int8" ? "省显存 INT8" : "标准 FP16"} · ${parseVram} GB`,
        params: { compute_type: config.computeType },
      },
    },
    {
      id: "llm_tagger",
      type: "stageNode",
      position: { x: 740, y: 18 },
      data: {
        title: "翻译与情绪标注",
        category: "parse",
        env: "env_parse",
        vramGb: 1.1,
        status: "done",
        inputs: [{ id: "in-turns", label: "角色时间段", type: "SEGMENTS" }],
        outputs: [{ id: "out-script-draft", label: "剧本初稿", type: "JSON" }],
        summary: `目标语言: ${config.targetLang === "zh" ? "中文" : config.targetLang === "en" ? "英文" : "日语"}`,
        params: { target_lang: config.targetLang },
      },
    },
    {
      id: "script_hub",
      type: "stageNode",
      position: { x: 500, y: 205 },
      data: {
        title: "剧本枢纽",
        category: "hub",
        env: "env_backend",
        vramGb: 0,
        status: "done",
        inputs: [
          { id: "in-draft", label: "剧本初稿", type: "JSON" },
          { id: "in-bgm", label: "背景轨", type: "AUDIO" },
        ],
        outputs: [
          { id: "out-script", label: "标准剧本", type: "SCRIPT" },
          { id: "out-bgm-pass", label: "背景轨", type: "AUDIO" },
        ],
        summary: "多端共享单一数据源",
        params: { schema: "v2" },
      },
    },
    {
      id: "agent_tts",
      type: "stageNode",
      position: { x: 740, y: 205 },
      data: {
        title: "语音合成与反思",
        category: "synthesis",
        env: "env_tts",
        vramGb: 4.2,
        status: "idle",
        inputs: [{ id: "in-script", label: "标准剧本", type: "SCRIPT" }],
        outputs: [{ id: "out-wavs", label: "分句配音", type: "AUDIO[]" }],
        summary: "超 15% 时最多改写两轮",
        params: { overflow_ratio: 1.15, max_rounds: 2 },
      },
    },
    {
      id: "ffmpeg_mixer",
      type: "stageNode",
      position: { x: 980, y: 112 },
      data: {
        title: "视频混音导出",
        category: "output",
        env: "env_backend",
        vramGb: 0,
        status: "idle",
        inputs: [
          { id: "in-wavs", label: "分句配音", type: "AUDIO[]" },
          { id: "in-bgm", label: "背景轨", type: "AUDIO" },
        ],
        outputs: [{ id: "out-mp4", label: "成片视频", type: "MP4" }],
        summary:
          config.burnSubtitles !== "none"
            ? config.burnWatermark
              ? "单次重编码烧录字幕与水印"
              : "重编码烧录 ASS 硬字幕"
            : config.burnWatermark
              ? "重编码烧录合规水印"
              : "无损流快速封装",
        params: {
          burn_watermark: config.burnWatermark,
          burn_subtitles: config.burnSubtitles,
        },
      },
    },
    {
      id: "virtual_stage_out",
      type: "stageNode",
      position: { x: 980, y: 278 },
      data: {
        title: "虚拟舞台演播",
        category: "output",
        env: "frontend",
        vramGb: 0.2,
        status: "idle",
        inputs: [
          { id: "in-script-stage", label: "标准剧本", type: "SCRIPT" },
          { id: "in-wavs-stage", label: "分句配音", type: "AUDIO[]" },
        ],
        outputs: [{ id: "out-canvas", label: "舞台画面", type: "CANVAS" }],
        summary: "实时能量嘴型与动作驱动",
        params: { renderer: "live2d" },
      },
    },
  ];
}

const INITIAL_EDGES: Edge[] = [
  {
    id: "e-input-banquet",
    source: "media_input",
    sourceHandle: "out-video",
    target: "banquet",
    targetHandle: "in-media",
    style: { stroke: "#A87B70", strokeWidth: 1.5 },
  },
  {
    id: "e-banquet-whisper",
    source: "banquet",
    sourceHandle: "out-vocals",
    target: "whisperx",
    targetHandle: "in-vocals",
    style: { stroke: "#927A9E", strokeWidth: 1.5 },
  },
  {
    id: "e-whisper-llm",
    source: "whisperx",
    sourceHandle: "out-turns",
    target: "llm_tagger",
    targetHandle: "in-turns",
    style: { stroke: "#927A9E", strokeWidth: 1.5 },
  },
  {
    id: "e-llm-hub",
    source: "llm_tagger",
    sourceHandle: "out-script-draft",
    target: "script_hub",
    targetHandle: "in-draft",
    style: { stroke: "#AD925F", strokeWidth: 1.5 },
  },
  {
    id: "e-banquet-hub-bgm",
    source: "banquet",
    sourceHandle: "out-bgm",
    target: "script_hub",
    targetHandle: "in-bgm",
    style: { stroke: "#5C5C5C", strokeWidth: 1.25, strokeDasharray: "4 4" },
  },
  {
    id: "e-hub-tts",
    source: "script_hub",
    sourceHandle: "out-script",
    target: "agent_tts",
    targetHandle: "in-script",
    style: { stroke: "#5C9186", strokeWidth: 1.5 },
  },
  {
    id: "e-tts-mixer",
    source: "agent_tts",
    sourceHandle: "out-wavs",
    target: "ffmpeg_mixer",
    targetHandle: "in-wavs",
    style: { stroke: "#698EAD", strokeWidth: 1.5 },
  },
  {
    id: "e-hub-mixer-bgm",
    source: "script_hub",
    sourceHandle: "out-bgm-pass",
    target: "ffmpeg_mixer",
    targetHandle: "in-bgm",
    style: { stroke: "#5C5C5C", strokeWidth: 1.25, strokeDasharray: "4 4" },
  },
  {
    id: "e-hub-stage",
    source: "script_hub",
    sourceHandle: "out-script",
    target: "virtual_stage_out",
    targetHandle: "in-script-stage",
    style: { stroke: "#AD925F", strokeWidth: 1.25, strokeDasharray: "4 4" },
  },
  {
    id: "e-tts-stage",
    source: "agent_tts",
    sourceHandle: "out-wavs",
    target: "virtual_stage_out",
    targetHandle: "in-wavs-stage",
    style: { stroke: "#698EAD", strokeWidth: 1.5 },
  },
];

function nowTimeStr(): string {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function makeLog(
  env: PipelineLogEntry["env"],
  message: string,
  level: PipelineLogEntry["level"] = "info",
): PipelineLogEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    timestamp: nowTimeStr(),
    env,
    message,
    level,
  };
}

const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  computeType: "int8",
  targetLang: "zh",
  burnWatermark: false,
  burnSubtitles: "none",
};

interface WorkflowState {
  workspacePreset: WorkspacePreset;
  centerTab: CenterViewTab;
  rightTab: RightPanelTab;
  leftDrawer: LeftDrawerTab;
  selectedNodeId: string | null;
  nodes: StageFlowNode[];
  edges: Edge[];
  pipelineConfig: PipelineConfig;
  isPipelineRunning: boolean;
  pipelineLogs: PipelineLogEntry[];

  setWorkspacePreset: (preset: WorkspacePreset) => void;
  setCenterTab: (tab: CenterViewTab) => void;
  setRightTab: (tab: RightPanelTab) => void;
  toggleLeftDrawer: (tab: Exclude<LeftDrawerTab, "none">) => void;
  selectNode: (nodeId: string | null) => void;
  onNodesChange: (changes: NodeChange<StageFlowNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<Edge>[]) => void;
  updatePipelineConfig: (patch: Partial<PipelineConfig>) => void;
  resetGraphLayout: () => void;
  appendLog: (
    env: PipelineLogEntry["env"],
    message: string,
    level?: PipelineLogEntry["level"],
  ) => void;
  runPipeline: (options?: { parseOnly?: boolean }) => Promise<void>;
  runPipelineSimulation: () => Promise<void>;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  workspacePreset: "editing",
  centerTab: "video",
  rightTab: "inspector",
  leftDrawer: "none",
  selectedNodeId: "agent_tts",
  nodes: createInitialNodes(DEFAULT_PIPELINE_CONFIG),
  edges: INITIAL_EDGES,
  pipelineConfig: DEFAULT_PIPELINE_CONFIG,
  isPipelineRunning: false,
  pipelineLogs: [makeLog("system", "RTX 5060 8GB 串行显存接力策略已就绪", "info")],

  setWorkspacePreset: (preset) => {
    if (preset === "workflow") {
      set({
        workspacePreset: preset,
        centerTab: "graph",
        rightTab: "node",
      });
      return;
    }
    if (preset === "stage") {
      set({
        workspacePreset: preset,
        centerTab: "stage",
        rightTab: "inspector",
      });
      return;
    }
    set((state) => ({
      workspacePreset: preset,
      centerTab: "video",
      rightTab:
        state.rightTab === "node" || state.rightTab === "logs" ? "inspector" : state.rightTab,
    }));
  },

  setCenterTab: (centerTab) => set({ centerTab }),

  setRightTab: (rightTab) => set({ rightTab }),

  toggleLeftDrawer: (tab) =>
    set((state) => ({
      leftDrawer: state.leftDrawer === tab ? "none" : tab,
    })),

  selectNode: (nodeId) =>
    set((state) => ({
      selectedNodeId: nodeId,
      rightTab: nodeId ? "node" : state.rightTab,
    })),

  onNodesChange: (changes) =>
    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes),
    })),

  onEdgesChange: (changes) =>
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
    })),

  updatePipelineConfig: (patch) =>
    set((state) => {
      const nextConfig = { ...state.pipelineConfig, ...patch };
      const parseVram = nextConfig.computeType === "int8" ? 3.4 : 5.2;
      const nextNodes = state.nodes.map((node) => {
        if (node.id === "whisperx") {
          return {
            ...node,
            data: {
              ...node.data,
              vramGb: parseVram,
              summary: `${nextConfig.computeType === "int8" ? "省显存 INT8" : "标准 FP16"} · ${parseVram} GB`,
              params: {
                ...node.data.params,
                compute_type: nextConfig.computeType,
              },
            },
          };
        }
        if (node.id === "llm_tagger") {
          const langLabel =
            nextConfig.targetLang === "zh"
              ? "中文"
              : nextConfig.targetLang === "en"
                ? "英文"
                : "日语";
          return {
            ...node,
            data: {
              ...node.data,
              summary: `目标语言: ${langLabel}`,
              params: {
                ...node.data.params,
                target_lang: nextConfig.targetLang,
              },
            },
          };
        }
        if (node.id === "ffmpeg_mixer") {
          return {
            ...node,
            data: {
              ...node.data,
              summary:
                nextConfig.burnSubtitles !== "none"
                  ? nextConfig.burnWatermark
                    ? "单次重编码烧录字幕与水印"
                    : "重编码烧录 ASS 硬字幕"
                  : nextConfig.burnWatermark
                    ? "重编码烧录合规水印"
                    : "无损流快速封装",
              params: {
                ...node.data.params,
                burn_watermark: nextConfig.burnWatermark,
                burn_subtitles: nextConfig.burnSubtitles,
              },
            },
          };
        }
        return node;
      });
      return {
        pipelineConfig: nextConfig,
        nodes: nextNodes,
      };
    }),

  resetGraphLayout: () =>
    set((state) => ({
      nodes: createInitialNodes(state.pipelineConfig),
      edges: INITIAL_EDGES,
    })),

  appendLog: (env, message, level = "info") =>
    set((state) => ({
      pipelineLogs: [...state.pipelineLogs.slice(-39), makeLog(env, message, level)],
    })),

  runPipeline: async (options) => {
    const parseOnly = Boolean(options?.parseOnly);
    if (get().isPipelineRunning) {
      return;
    }

    const projectState = useProjectStore.getState();
    const currentScript = projectState.script;
    if (!currentScript) {
      set({ rightTab: "logs" });
      get().appendLog("system", "请先导入音视频或打开一个项目后再运行管线", "warn");
      return;
    }

    if (projectState.backendStatus === "offline") {
      set({ rightTab: "logs" });
      get().appendLog("system", "当前处于离线模式，无法调用本地算力，改走前端演示", "warn");
      await get().runPipelineSimulation();
      return;
    }

    const setNodeStatus = (nodeId: string, status: NodeRunStatus) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, status } } : node,
        ),
      }));
    };

    const setActiveEdges = (activeEdgeIds: string[]) => {
      const activeSet = new Set(activeEdgeIds);
      set((state) => ({
        edges: state.edges.map((edge) => ({
          ...edge,
          animated: activeSet.has(edge.id),
        })),
      }));
    };

    const targetProjectId = currentScript.project_id;
    const isSameProject = () => useProjectStore.getState().script?.project_id === targetProjectId;
    const cfg = get().pipelineConfig;
    const needsParse =
      parseOnly || currentScript.segments.length === 0 || !currentScript.media.vocals_wav;

    set((state) => ({
      isPipelineRunning: true,
      rightTab: "logs",
      nodes: state.nodes.map((n) => {
        if (
          !needsParse &&
          ["media_input", "banquet", "whisperx", "llm_tagger", "script_hub"].includes(n.id)
        ) {
          return { ...n, data: { ...n.data, status: "done" } };
        }
        return { ...n, data: { ...n.data, status: "queued" } };
      }),
    }));

    let activeScript = currentScript;
    let activeNode = needsParse ? "media_input" : "agent_tts";

    try {
      // -------------------------------------------------------------------
      // 阶段 A：调用真实后端 POST /api/projects/{id}/parse（env_parse 子进程）
      // -------------------------------------------------------------------
      if (needsParse) {
        activeNode = "media_input";
        setNodeStatus("media_input", "running");
        setActiveEdges(["e-input-banquet"]);
        get().appendLog(
          "env_parse",
          `启动真实 4 阶段解析：项目 ${activeScript.project_id}（目标语种=${cfg.targetLang}，精度=${cfg.computeType}）`,
          "info",
        );

        let seenLogCount = 0;
        const applyStageFromWorker = (stage: string) => {
          if (stage === "Stage0" || stage === "Init") {
            activeNode = "media_input";
            setNodeStatus("media_input", "running");
            setActiveEdges(["e-input-banquet"]);
          } else if (stage === "Stage1") {
            activeNode = "banquet";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "running");
            setActiveEdges(["e-banquet-whisper", "e-banquet-hub-bgm"]);
          } else if (stage === "Stage2" || stage === "Stage3") {
            activeNode = "whisperx";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "running");
            setActiveEdges(["e-whisper-llm"]);
          } else if (stage === "Stage4") {
            activeNode = "llm_tagger";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "done");
            setNodeStatus("llm_tagger", "running");
            setActiveEdges(["e-llm-hub"]);
          } else if (stage === "Done") {
            activeNode = "script_hub";
            setNodeStatus("media_input", "done");
            setNodeStatus("banquet", "done");
            setNodeStatus("whisperx", "done");
            setNodeStatus("llm_tagger", "done");
            setNodeStatus("script_hub", "done");
          }
        };

        const applyParseStatus = (status: ParseStatusResponse, final = false) => {
          if (!final && !status.active) {
            return;
          }
          const count = status.log_count ?? status.logs.length;
          if (count < seenLogCount) {
            return;
          }
          if (status.stage && status.stage !== "idle") {
            applyStageFromWorker(status.stage);
          }
          const firstNew = Math.max(0, status.logs.length - (count - seenLogCount));
          for (const line of status.logs.slice(firstNew)) {
            const level = /使用未分离音频|失败|显存不足/.test(line) ? "warn" : "info";
            get().appendLog("env_parse", line, level);
          }
          seenLogCount = count;
        };

        const refreshParseStatus = async (final = false) => {
          try {
            const status = await fetchParseStatus(targetProjectId);
            applyParseStatus(status, final);
          } catch {
            // 解析仍由 POST 请求跟踪。
          }
        };

        let fallbackTimer: number | undefined;
        const unsubscribeSse = subscribeParseStatus(
          targetProjectId,
          (status) => applyParseStatus(status),
          () => {
            if (fallbackTimer === undefined) {
              fallbackTimer = window.setInterval(() => void refreshParseStatus(), 1200);
            }
          },
        );

        try {
          if (projectState.isDirty) {
            activeScript = await saveProjectScript(activeScript);
            if (isSameProject()) {
              useProjectStore.setState({ isDirty: false });
            }
          }
          activeScript = await parseProjectMedia(activeScript.project_id, {
            target_lang: cfg.targetLang,
            compute_type: cfg.computeType,
          });
        } finally {
          unsubscribeSse();
          if (fallbackTimer !== undefined) {
            window.clearInterval(fallbackTimer);
          }
          await refreshParseStatus(true);
        }

        // 将解析出的真实剧本写回全局项目状态（仅当当前未切换项目时更新视图）
        if (isSameProject()) {
          useProjectStore.getState().setScript(activeScript);
        }
        setNodeStatus("media_input", "done");
        setNodeStatus("banquet", "done");
        setNodeStatus("whisperx", "done");
        setNodeStatus("llm_tagger", "done");
        setNodeStatus("script_hub", "done");

        const charCount = Object.keys(activeScript.characters).length;
        get().appendLog(
          "env_parse",
          `解析完成并已释放显存：生成 ${activeScript.segments.length} 条片段、${charCount} 个说话人`,
          "success",
        );

        if (parseOnly) {
          setActiveEdges([]);
          set({ isPipelineRunning: false });
          get().appendLog(
            "system",
            "仅解析模式已完成，可切回「剪辑」视图查看时间轴并按需合成配音",
            "success",
          );
          return;
        }
      }

      if (activeScript.segments.length === 0) {
        setActiveEdges([]);
        set({ isPipelineRunning: false });
        get().appendLog("system", "未识别到可配音的语音片段，管线已停止", "warn");
        return;
      }

      // -------------------------------------------------------------------
      // 阶段 B：调用真实后端 POST /api/projects/{id}/synthesize（GPT-SoVITS）
      // -------------------------------------------------------------------
      if (!needsParse && projectState.isDirty) {
        activeScript = await saveProjectScript(activeScript);
        if (isSameProject()) {
          useProjectStore.setState({ script: activeScript, isDirty: false });
        }
      }
      const pendingIds = activeScript.segments
        .filter((s) => !s.keep_original && s.status !== "ready")
        .map((s) => s.id);

      activeNode = "agent_tts";
      setNodeStatus("agent_tts", "running");
      setActiveEdges(["e-hub-tts", "e-tts-mixer", "e-tts-stage"]);

      if (pendingIds.length > 0) {
        get().appendLog(
          "env_tts",
          `开始调用 GPT-SoVITS 合成 ${pendingIds.length} 条待合成片段...`,
          "info",
        );
        const synthRes = await synthesizeSegments(activeScript.project_id, pendingIds);
        activeScript = synthRes.script;
        if (isSameProject()) {
          useProjectStore.getState().setScript(activeScript);
          useProjectStore.getState().bumpAudioVersion(synthRes.synthesized_ids);
        }

        for (const issue of synthesisIssues(synthRes)) {
          get().appendLog("env_tts", issue, "warn");
        }
        get().appendLog(
          "env_tts",
          `语音合成完成：成功 ${synthRes.synthesized_ids.length} 条`,
          "success",
        );
      } else {
        get().appendLog("env_tts", "所有片段均为已就绪状态，跳过重复合成", "info");
      }
      setNodeStatus("agent_tts", "done");

      // -------------------------------------------------------------------
      // 阶段 C：调用真实后端 POST /api/projects/{id}/render（FFmpeg 混音封装）
      // -------------------------------------------------------------------
      const hasReadyOrOrig = activeScript.segments.some(
        (s) => s.status === "ready" || s.keep_original,
      );
      if (hasReadyOrOrig) {
        activeNode = "ffmpeg_mixer";
        setNodeStatus("ffmpeg_mixer", "running");
        setNodeStatus("virtual_stage_out", "running");
        get().appendLog("env_backend", "正在执行 pydub 毫秒拼轨与 FFmpeg 混音封装...", "info");

        const renderRes = await renderVideo(activeScript.project_id, {
          burn_watermark: cfg.burnWatermark,
          burn_subtitles: cfg.burnSubtitles,
          auto_ducking: activeScript.media.auto_ducking ?? true,
        });
        activeScript = {
          ...activeScript,
          media: {
            ...activeScript.media,
            output_video: renderRes.output_video,
          },
        };
        if (isSameProject()) {
          useProjectStore.getState().setScript(activeScript);
        }
        setNodeStatus("ffmpeg_mixer", "done");
        setNodeStatus("virtual_stage_out", "done");
        get().appendLog(
          "env_backend",
          `全管线执行完毕，成片已输出：${renderRes.output_video}`,
          "success",
        );
      } else {
        setNodeStatus("ffmpeg_mixer", "idle");
        setNodeStatus("virtual_stage_out", "idle");
        get().appendLog("env_backend", "暂无已就绪配音轨，已跳过成片封装", "warn");
      }

      setActiveEdges([]);
      set({ isPipelineRunning: false });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "管线执行异常";
      setNodeStatus(activeNode, "error");
      setActiveEdges([]);
      set({ isPipelineRunning: false });
      get().appendLog("system", `管线执行失败：${msg}`, "error");
    }
  },

  runPipelineSimulation: async () => {
    if (get().isPipelineRunning) {
      return;
    }

    const setNodeStatus = (nodeId: string, status: NodeRunStatus) => {
      set((state) => ({
        nodes: state.nodes.map((node) =>
          node.id === nodeId ? { ...node, data: { ...node.data, status } } : node,
        ),
      }));
    };

    const setActiveEdges = (activeEdgeIds: string[]) => {
      const activeSet = new Set(activeEdgeIds);
      set((state) => ({
        edges: state.edges.map((edge) => ({
          ...edge,
          animated: activeSet.has(edge.id),
        })),
      }));
    };

    const cfg = get().pipelineConfig;
    const whisperVram = cfg.computeType === "int8" ? 3.4 : 5.2;

    set((state) => ({
      isPipelineRunning: true,
      nodes: state.nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: "queued" },
      })),
    }));
    get().appendLog("system", "开始执行全链路管线调度", "info");

    setNodeStatus("media_input", "running");
    setActiveEdges(["e-input-banquet"]);
    await sleep(350);
    setNodeStatus("media_input", "done");

    setNodeStatus("banquet", "running");
    setActiveEdges(["e-banquet-whisper", "e-banquet-hub-bgm"]);
    get().appendLog("env_parse", "Banquet 分离对白、歌声、伴奏和音效", "info");
    await sleep(500);
    setNodeStatus("banquet", "done");

    setNodeStatus("whisperx", "running");
    setActiveEdges(["e-whisper-llm"]);
    get().appendLog("env_parse", `语音识别与说话人聚类（显存 ${whisperVram} GB）`, "info");
    await sleep(600);
    setNodeStatus("whisperx", "done");

    setNodeStatus("llm_tagger", "running");
    setActiveEdges(["e-llm-hub"]);
    get().appendLog("env_parse", "口语化翻译与情绪动作标注", "info");
    await sleep(450);
    setNodeStatus("llm_tagger", "done");

    get().appendLog("env_parse", "解析子进程退出，显存已释放归零", "success");
    await sleep(350);

    setNodeStatus("script_hub", "running");
    setActiveEdges(["e-hub-tts", "e-hub-mixer-bgm", "e-hub-stage"]);
    await sleep(350);
    setNodeStatus("script_hub", "done");

    setNodeStatus("agent_tts", "running");
    setActiveEdges(["e-tts-mixer", "e-tts-stage"]);
    get().appendLog("env_tts", "按角色分组合成并检查时长约束 (4.2 GB)", "info");
    await sleep(700);
    setNodeStatus("agent_tts", "done");

    setNodeStatus("ffmpeg_mixer", "running");
    setNodeStatus("virtual_stage_out", "running");
    get().appendLog("env_backend", "混音封装与舞台同步完成", "success");
    await sleep(450);
    setNodeStatus("ffmpeg_mixer", "done");
    setNodeStatus("virtual_stage_out", "done");
    setActiveEdges([]);
    set({ isPipelineRunning: false });
  },
}));
