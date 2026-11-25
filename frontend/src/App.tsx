import { Bot, Sliders, Terminal, Users, Wrench } from "lucide-react";
import type React from "react";
import { useEffect } from "react";
import { AgentChat } from "./components/AgentChat";
import { Inspector } from "./components/Inspector";
import { TimelinePR } from "./components/TimelinePR";
import { VideoPreview } from "./components/VideoPreview";
import { VirtualStage } from "./components/VirtualStage";
import { LeftIconRail } from "./components/layout/LeftIconRail";
import { ResizableSplit } from "./components/layout/ResizableSplit";
import { TopMenuBar } from "./components/layout/TopMenuBar";
import { WelcomeCanvas } from "./components/layout/WelcomeCanvas";
import { AssetBinPanel } from "./components/panels/AssetBinPanel";
import { NodeInspectorPanel, PipelineLogsPanel } from "./components/panels/NodeInspectorPanel";
import { SpeakerReviewPanel } from "./components/panels/SpeakerReviewPanel";
import { PipelineGraph } from "./components/workflow/PipelineGraph";
import { useProjectStore } from "./store/useProjectStore";
import { useWorkflowStore } from "./store/useWorkflowStore";

export const App: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const isPlaying = useProjectStore((s) => s.isPlaying);
  const initWorkspace = useProjectStore((s) => s.initWorkspace);
  const stepSegment = useProjectStore((s) => s.stepSegment);
  const setIsPlaying = useProjectStore((s) => s.setIsPlaying);

  const workspacePreset = useWorkflowStore((s) => s.workspacePreset);
  const rightTab = useWorkflowStore((s) => s.rightTab);
  const setRightTab = useWorkflowStore((s) => s.setRightTab);

  // 启动时仅探活后端与拉取项目列表，绝不自动载入样例
  useEffect(() => {
    void initWorkspace();
  }, [initWorkspace]);

  // 播放驱动循环
  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    let lastTs = performance.now();
    let rafId = 0;

    const tick = (now: number) => {
      const dt = (now - lastTs) / 1000;
      lastTs = now;
      const state = useProjectStore.getState();
      const duration = state.script?.media.duration ?? 18.5;
      const nextTime = state.currentTime + dt;
      if (nextTime >= duration) {
        state.setCurrentTime(0);
        state.setIsPlaying(false);
        return;
      }
      state.setCurrentTime(nextTime);
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  // 全局快捷键：Ctrl/Cmd+S 保存，Ctrl/Cmd+Z 撤销，Ctrl/Cmd+Shift+Z / Y 重做，Space 播放/暂停，← / → 切换片段
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const state = useProjectStore.getState();
      if (!state.script) {
        return;
      }
      const busy =
        state.isLoading || state.isSynthesizing || useWorkflowStore.getState().isPipelineRunning;
      const mod = e.ctrlKey || e.metaKey;
      const lowerKey = e.key.toLowerCase();
      if (mod && lowerKey === "s") {
        e.preventDefault();
        if (!busy) {
          void state.saveProjectScript();
        }
        return;
      }
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target?.isContentEditable) {
        return;
      }
      if (mod && lowerKey === "z") {
        e.preventDefault();
        if (busy) {
          return;
        }
        if (e.shiftKey) {
          state.redo();
        } else {
          state.undo();
        }
        return;
      }
      if (mod && !e.shiftKey && lowerKey === "y") {
        e.preventDefault();
        if (!busy) {
          state.redo();
        }
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        const playing = useProjectStore.getState().isPlaying;
        setIsPlaying(!playing);
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        stepSegment(-1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        stepSegment(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setIsPlaying, stepSegment]);

  const renderWorkflowRightPanel = () => {
    const activeTab = rightTab === "logs" ? "logs" : "node";
    return (
      <div className="flex h-full flex-col bg-[#222222]">
        <div className="shrink-0 border-b border-white/[0.06] bg-[#222222] px-3 py-1.5">
          <div className="grid h-7 grid-cols-2 gap-0.5 rounded-[4px] bg-[#181818] p-0.5 text-[12px]">
            <button
              type="button"
              onClick={() => setRightTab("node")}
              className={`flex items-center justify-center gap-1.5 rounded-[3px] font-medium transition-colors ${
                activeTab === "node"
                  ? "bg-[#333333] text-[#FFFFFF]"
                  : "text-secondary hover:text-primary"
              }`}
            >
              <Wrench className="h-3.5 w-3.5" strokeWidth={1.5} />
              <span>节点参数</span>
            </button>
            <button
              type="button"
              onClick={() => setRightTab("logs")}
              className={`flex items-center justify-center gap-1.5 rounded-[3px] font-medium transition-colors ${
                activeTab === "logs"
                  ? "bg-[#333333] text-[#FFFFFF]"
                  : "text-secondary hover:text-primary"
              }`}
            >
              <Terminal className="h-3.5 w-3.5" strokeWidth={1.5} />
              <span>运行日志</span>
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1">
          {activeTab === "logs" ? <PipelineLogsPanel /> : <NodeInspectorPanel />}
        </div>
      </div>
    );
  };

  const renderEditingRightPanel = () => {
    const activeTab = rightTab === "agent" || rightTab === "speakers" ? rightTab : "inspector";
    return (
      <div className="flex h-full flex-col bg-[#222222]">
        <div className="shrink-0 border-b border-white/[0.06] bg-[#222222] px-3 py-1.5">
          <div className="grid h-7 grid-cols-3 gap-0.5 rounded-[4px] bg-[#181818] p-0.5 text-[12px]">
            <button
              type="button"
              onClick={() => setRightTab("inspector")}
              className={`flex items-center justify-center gap-1.5 rounded-[3px] font-medium transition-colors ${
                activeTab === "inspector"
                  ? "bg-[#333333] text-[#FFFFFF]"
                  : "text-secondary hover:text-primary"
              }`}
            >
              <Sliders className="h-3.5 w-3.5" strokeWidth={1.5} />
              <span>精修</span>
            </button>
            <button type="button" onClick={() => setRightTab("speakers")}
              className={`flex items-center justify-center gap-1.5 rounded-[3px] font-medium ${activeTab === "speakers" ? "bg-[#333333] text-white" : "text-secondary hover:text-primary"}`}>
              <Users className="h-3.5 w-3.5" strokeWidth={1.5} />
              <span>角色复核</span>
            </button>
            <button
              type="button"
              onClick={() => setRightTab("agent")}
              className={`flex items-center justify-center gap-1.5 rounded-[3px] font-medium transition-colors ${
                activeTab === "agent"
                  ? "bg-[#333333] text-[#FFFFFF]"
                  : "text-secondary hover:text-primary"
              }`}
            >
              <Bot className="h-3.5 w-3.5" strokeWidth={1.5} />
              <span>AI 助手</span>
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1">
          {activeTab === "agent" && <AgentChat />}
          {activeTab === "speakers" && script && <SpeakerReviewPanel key={script.project_id} script={script} />}
          {activeTab === "inspector" && <Inspector />}
        </div>
      </div>
    );
  };

  const renderMainBody = () => {
    if (workspacePreset === "workflow") {
      return (
        <ResizableSplit
          direction="horizontal"
          initialSize={70}
          minSize={55}
          maxSize={84}
          className="h-full flex-1"
          first={<PipelineGraph />}
          second={renderWorkflowRightPanel()}
        />
      );
    }

    if (!script) {
      return <WelcomeCanvas />;
    }

    return (
      <div className="flex min-h-0 flex-1 gap-[2px] overflow-hidden bg-[#121212]">
        <LeftIconRail />
        <AssetBinPanel />

        <div className="min-w-0 flex-1">
          <ResizableSplit
            direction="vertical"
            initialSize={65}
            minSize={46}
            maxSize={80}
            className="h-full"
            first={
              <ResizableSplit
                direction="horizontal"
                initialSize={63}
                minSize={45}
                maxSize={78}
                className="h-full"
                first={workspacePreset === "stage" ? <VirtualStage /> : <VideoPreview />}
                second={renderEditingRightPanel()}
              />
            }
            second={<TimelinePR />}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen flex-col gap-[2px] overflow-hidden bg-[#121212] text-primary">
      <TopMenuBar />
      <main className="flex min-h-0 flex-1 overflow-hidden">{renderMainBody()}</main>
    </div>
  );
};

export default App;
