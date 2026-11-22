import React, { useCallback, useEffect, useRef, useState } from "react";

interface ResizableSplitProps {
  direction: "horizontal" | "vertical";
  initialSize: number;
  minSize?: number;
  maxSize?: number;
  first: React.ReactNode;
  second: React.ReactNode;
  className?: string;
}

/**
 * 2px #121212 暗槽面板分隔容器（若隐若现、无亮线网格感，支持双击复位）。
 */
export const ResizableSplit: React.FC<ResizableSplitProps> = ({
  direction,
  initialSize,
  minSize = 15,
  maxSize = 85,
  first,
  second,
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<number>(initialSize);
  const [isDragging, setIsDragging] = useState(false);

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      const el = containerRef.current;
      if (!el) {
        return;
      }
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      const rawPercent =
        direction === "horizontal"
          ? ((event.clientX - rect.left) / rect.width) * 100
          : ((event.clientY - rect.top) / rect.height) * 100;
      const clamped = Math.max(minSize, Math.min(maxSize, rawPercent));
      setSize(clamped);
    },
    [direction, maxSize, minSize]
  );

  useEffect(() => {
    if (!isDragging) {
      return;
    }
    const handleUp = () => {
      setIsDragging(false);
      document.body.style.userSelect = "";
    };
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, [handlePointerMove, isDragging]);

  const isHorizontal = direction === "horizontal";

  return (
    <div
      ref={containerRef}
      className={`flex min-h-0 min-w-0 bg-[#121212] ${
        isHorizontal ? "flex-row" : "flex-col"
      } ${className}`}
    >
      <div
        style={isHorizontal ? { width: `${size}%` } : { height: `${size}%` }}
        className="min-h-0 min-w-0 shrink-0"
      >
        {first}
      </div>

      <div
        role="separator"
        aria-orientation={isHorizontal ? "vertical" : "horizontal"}
        title="拖拽调整面板大小 · 双击恢复默认"
        onPointerDown={() => setIsDragging(true)}
        onDoubleClick={() => setSize(initialSize)}
        className={`relative shrink-0 transition-colors after:absolute after:z-10 after:content-[''] ${
          isHorizontal
            ? "w-[2px] cursor-col-resize after:inset-y-0 after:-left-[3px] after:-right-[3px]"
            : "h-[2px] cursor-row-resize after:inset-x-0 after:-top-[3px] after:-bottom-[3px]"
        } ${isDragging ? "bg-[#2C2C2C]" : "bg-[#121212] hover:bg-[#242424]"}`}
      />

      <div className="min-h-0 min-w-0 flex-1">{second}</div>
    </div>
  );
};
