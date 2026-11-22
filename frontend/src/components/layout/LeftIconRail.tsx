import { Film, Home, Users } from "lucide-react";
import type React from "react";
import { useProjectStore } from "../../store/useProjectStore";
import { useWorkflowStore } from "../../store/useWorkflowStore";

/**
 * 左侧 40px 停靠竖向图标栏（#222222 哑光面板，无边框）。
 */
export const LeftIconRail: React.FC = () => {
  const script = useProjectStore((s) => s.script);
  const closeProject = useProjectStore((s) => s.closeProject);

  const leftDrawer = useWorkflowStore((s) => s.leftDrawer);
  const toggleLeftDrawer = useWorkflowStore((s) => s.toggleLeftDrawer);

  if (!script) {
    return null;
  }

  return (
    <aside className="flex w-10 shrink-0 flex-col items-center justify-between bg-[#222222] py-2">
      <div className="flex flex-col items-center gap-1.5">
        <button
          type="button"
          title="角色与情绪音色"
          onClick={() => toggleLeftDrawer("characters")}
          className={`flex h-7 w-7 items-center justify-center rounded-[4px] transition-colors ${
            leftDrawer === "characters"
              ? "bg-[#333333] text-[#FFFFFF]"
              : "text-secondary hover:bg-[#2C2C2C] hover:text-primary"
          }`}
        >
          <Users className="h-4 w-4" strokeWidth={1.5} />
        </button>

        <button
          type="button"
          title="项目媒体素材"
          onClick={() => toggleLeftDrawer("media")}
          className={`flex h-7 w-7 items-center justify-center rounded-[4px] transition-colors ${
            leftDrawer === "media"
              ? "bg-[#333333] text-[#FFFFFF]"
              : "text-secondary hover:bg-[#2C2C2C] hover:text-primary"
          }`}
        >
          <Film className="h-4 w-4" strokeWidth={1.5} />
        </button>
      </div>

      <button
        type="button"
        title="返回首页（关闭当前项目）"
        onClick={closeProject}
        className="flex h-7 w-7 items-center justify-center rounded-[4px] text-secondary transition-colors hover:bg-[#2C2C2C] hover:text-primary"
      >
        <Home className="h-4 w-4" strokeWidth={1.5} />
      </button>
    </aside>
  );
};
