import { ChevronRight, Clapperboard, FolderOpen, Play, Upload } from "lucide-react";
import type React from "react";
import { useRef, useState } from "react";
import { useProjectStore } from "../../store/useProjectStore";

/**
 * Premiere / Blender 风格紧凑项目启动器（未打开项目时展示，不自动载入样例）。
 */
export const WelcomeCanvas: React.FC = () => {
  const projectList = useProjectStore((s) => s.projectList);
  const backendStatus = useProjectStore((s) => s.backendStatus);
  const isLoading = useProjectStore((s) => s.isLoading);
  const importMediaFile = useProjectStore((s) => s.importMediaFile);
  const loadProjectScript = useProjectStore((s) => s.loadProjectScript);
  const loadDemoProject = useProjectStore((s) => s.loadDemoProject);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) {
      void importMediaFile(file);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      className="flex h-full w-full items-center justify-center bg-[#161616] p-6 text-[12px]"
    >
      <div
        className={`w-full max-w-2xl overflow-hidden rounded-[4px] border bg-[#222222] transition-colors ${
          isDragging ? "border-[#3A6EA5]" : "border-white/[0.06]"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />

        {/* 顶部标题条 */}
        <div className="flex h-9 items-center justify-between border-b border-white/[0.06] bg-[#1C1C1C] px-3.5">
          <div className="flex items-center gap-2">
            <Clapperboard className="h-4 w-4 text-secondary" strokeWidth={1.5} />
            <span className="text-[13px] font-medium text-primary">StageTTS 项目启动器</span>
          </div>

          <div className="flex items-center gap-1.5 rounded-[4px] bg-white/[0.04] px-2 py-0.5 text-[11px] text-secondary">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                backendStatus === "online"
                  ? "bg-status-ready"
                  : backendStatus === "connecting"
                    ? "bg-secondary animate-pulse"
                    : "bg-status-warning"
              }`}
            />
            <span>
              {backendStatus === "online"
                ? "后端已连接"
                : backendStatus === "connecting"
                  ? "连接检测中"
                  : "离线预览模式"}
            </span>
          </div>
        </div>

        {/* 双栏主区：左侧快捷操作列表 + 右侧最近项目表 */}
        <div className="grid grid-cols-1 md:grid-cols-[240px_1fr]">
          {/* 左侧：快捷操作行 */}
          <div className="flex flex-col justify-between border-b border-white/[0.06] p-3 md:border-b-0 md:border-r">
            <div>
              <div className="mb-2 px-2 text-[11px] font-medium uppercase tracking-wider text-muted">
                开始
              </div>

              <div className="space-y-1">
                <button
                  type="button"
                  disabled={isLoading}
                  onClick={() => fileInputRef.current?.click()}
                  className="group flex w-full items-center justify-between rounded-[4px] px-2.5 py-2 text-left transition-colors hover:bg-[#2C2C2C] disabled:opacity-50"
                >
                  <div className="flex items-start gap-2.5">
                    <Upload
                      className="mt-0.5 h-4 w-4 shrink-0 text-secondary group-hover:text-primary"
                      strokeWidth={1.5}
                    />
                    <div>
                      <div className="font-medium text-primary">新建 / 导入音视频</div>
                      <div className="mt-0.5 text-[11px] text-secondary">选择或拖拽 MP4 / WAV</div>
                    </div>
                  </div>
                  <ChevronRight
                    className="h-3.5 w-3.5 shrink-0 text-muted group-hover:text-primary"
                    strokeWidth={1.5}
                  />
                </button>

                <button
                  type="button"
                  disabled={isLoading}
                  onClick={() => void loadDemoProject()}
                  className="group flex w-full items-center justify-between rounded-[4px] px-2.5 py-2 text-left transition-colors hover:bg-[#2C2C2C] disabled:opacity-50"
                >
                  <div className="flex items-start gap-2.5">
                    <Play
                      className="mt-0.5 h-4 w-4 shrink-0 text-secondary group-hover:text-primary"
                      strokeWidth={1.5}
                    />
                    <div>
                      <div className="font-medium text-primary">载入演示样例</div>
                      <div className="mt-0.5 text-[11px] text-secondary">双角色 4 句示例剧本</div>
                    </div>
                  </div>
                  <ChevronRight
                    className="h-3.5 w-3.5 shrink-0 text-muted group-hover:text-primary"
                    strokeWidth={1.5}
                  />
                </button>
              </div>
            </div>

            <div className="mt-4 border-t border-dashed border-white/[0.08] px-2 pt-2.5 text-[11px] text-muted">
              多角色情感配音 · 自然语言改轨 · 虚拟演播
            </div>
          </div>

          {/* 右侧：最近项目单色数据表 */}
          <div className="flex min-h-[220px] flex-col">
            <div className="flex h-8 items-center justify-between border-b border-white/[0.06] bg-[#1D1D1D] px-3 text-[11px] text-secondary">
              <span className="flex items-center gap-1.5 font-medium">
                <FolderOpen className="h-3.5 w-3.5 text-secondary" strokeWidth={1.5} />
                <span>最近项目</span>
              </span>
              <span className="font-mono tabular-nums text-muted">{projectList.length} 项</span>
            </div>

            {projectList.length > 0 ? (
              <div className="flex-1 overflow-y-auto">
                <table className="w-full border-collapse text-left text-[12px]">
                  <thead>
                    <tr className="border-b border-white/[0.06] bg-[#222222] text-[11px] text-muted">
                      <th className="px-3 py-1.5 font-normal">项目标识</th>
                      <th className="w-20 px-3 py-1.5 text-right font-normal">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {projectList.map((projId) => (
                      <tr
                        key={projId}
                        className="border-b border-white/[0.05] transition-colors hover:bg-[#2C2C2C]"
                      >
                        <td className="px-3 py-1.5 font-mono tabular-nums text-primary">
                          {projId}
                        </td>
                        <td className="px-3 py-1.5 text-right">
                          <button
                            type="button"
                            disabled={isLoading}
                            onClick={() => void loadProjectScript(projId)}
                            className="h-6 rounded-[4px] border border-white/[0.08] bg-[#2C2C2C] px-2 text-[11px] text-primary transition-colors hover:bg-[#363636] disabled:opacity-50"
                          >
                            打开
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center px-6 py-8 text-center">
                <p className="text-[12px] text-secondary">
                  {backendStatus === "online"
                    ? "工作区暂无历史项目"
                    : "后端服务未连接，暂无法读取工作区列表"}
                </p>
                <p className="mt-1 text-[11px] text-muted">
                  可从左侧导入本地媒体，或载入演示样例快速体验
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
