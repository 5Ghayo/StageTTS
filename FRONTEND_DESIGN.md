# StageTTS UI Design System (Pro NLE & Node-Tool Aesthetic)

## 1. Design Direction
Redesign StageTTS as a high-density, desktop-grade audio/video production workbench inspired by **Adobe Premiere Pro (Spectrum Dark)**, **ComfyUI**, **Blender**, and **Linear**.
Core philosophy: **Quiet Chrome, Loud Data**. The application shell must stay completely neutral and unobtrusive so that video content, waveforms, speaker tags, and status indicators stand out effortlessly.

## 2. Strict Color Palette (Zero Blue-Tint, Pure Neutral Matte Grays)
NEVER use pure black (`#000000`) or Tailwind `slate-*` (blue-tinted grays). Use ONLY neutral charcoal grays (`zinc`/`neutral` scale or exact hex CSS variables below):

- `--bg-canvas`: `#141414` (Video viewport & waveform track well — deepest layer)
- `--bg-workspace`: `#1A1A1A` (Main app background & panel gaps)
- `--bg-panel`: `#212121` (Docked panels: Inspector, Topbar, Timeline header)
- `--bg-surface`: `#2A2A2A` (Inputs, buttons, clip blocks, recessed cards)
- `--bg-hover`: `#333333` (Interactive hover state)
- `--border-subtle`: `#2E2E2E` (1px crisp docked panel dividers)
- `--border-strong`: `#3F3F3F` (Input borders, active panel outlines)
- `--text-primary`: `#E4E4E4` (Off-white, never `#FFFFFF` to avoid halation)
- `--text-secondary`: `#999999` (Field labels, metadata)
- `--text-muted`: `#666666` (Timecode ticks, disabled states)

## 3. Color Restraint & Semantic Color Budget
- **90% of the UI must be monochrome gray.**
- **Buttons & Tabs**:
  - Top navigation tabs (`剪辑 / 节点流 / 舞台`): NO bright blue filled pill. Use dark recessed segmented control (`#1A1A1A` container, active tab `#303030` with `#E4E4E4` text and subtle `1px` border).
  - Primary CTA buttons (`导出成片`, `重合成试听`): Use a single muted steel-blue accent (`#3B6EA5`, hover `#467DB8`) OR monochrome high-contrast (`#E4E4E4` bg with `#141414` text). Do NOT paint play buttons, sliders, and badges all bright blue.
  - Secondary buttons (`保存`, `当前句试听`): Flat `#2A2A2A` with `1px solid #3F3F3F` border.
- **Where Color IS Allowed (Desaturated, Semantic Only)**:
  - **Speaker Identifiers**: Use a thin `3px` left accent bar or `6px` dot on timeline clips and labels (e.g., Speaker A: muted slate-blue `#5B8DEF`, Speaker B: muted sage `#4FA882`, Speaker C: muted ochre `#C99A4E`). Keep clip fill dark neutral (`#262626`, selected `#2F2F2F` with `1px` speaker-color border).
  - **Status Indicators**: Tiny `6px` dots or subtle muted text only — Ready (`#4FA882`), Pending/Warning (`#C98A3E`), Overflow Error (`#C95252`).
  - **Playhead**: Crisp `1px` muted coral-red (`#D94A4A`).
  - **Sliders & Waveforms**: Monochrome gray track (`#383838`), light-gray fill (`#A3A3A3`), neutral waveform bars (`#6E7681`, active region `#B0B8C1`).

## 4. Layout Density, Geometry & Typography
- **Docked Panels, Not Floating Cards**: Remove outer margins and floating rounded boxes. Panels must dock edge-to-edge separated strictly by `1px solid #2E2E2E` dividers (like Premiere Pro / ComfyUI).
- **Tight Radii**: Panels `0px` (docked); buttons/inputs/clips `3px–4px` (`rounded-[3px]`). BAN `rounded-xl`, `rounded-2xl`, and `rounded-full` pills (except tiny status dots).
- **Compact Pro Controls**:
  - Topbar height: `36px` (compact, single-line).
  - Buttons & inputs height: `28px`, padding `px-2.5 py-1`.
  - Font sizes: `12px` default UI body, `11px` uppercase/muted section labels, `13px` panel titles.
  - All timecodes (`00:00:00`), durations (`2.5s / 2.8s`), VRAM (`0.2/8G`), and percentages (`75%`) MUST use `font-mono tabular-nums` (`JetBrains Mono`, monospace).

## 5. Specific Refactoring Targets for Current Screens
1. **Right Inspector Panel**:
   - Flatten nested bordered boxes! Do NOT wrap sliders or duration bars inside separate cards. Use clean horizontal rows (`label` on left / compact control on right) separated by subtle `1px` hairline dividers or `10px` vertical gaps so ALL controls including `当前句试听` fit cleanly without clipping.
   - Emotion selector: Compact 5-column segmented bar (`26px` height) with subtle `#383838` active state + tiny colored dot, NOT bright blue filled buttons.
2. **Timeline Area**:
   - Replace bright orange `合成待处理 (1)` button with a subtle bordered chip (`#2A2A2A` bg, `#C98A3E` 6px dot + muted text).
   - Keep `AI 合成语音` compliance tag minimal and show it ONLY ONCE in the topbar or canvas corner as a subtle translucent monochrome badge.
3. **Welcome / Empty State**:
   - Replace the generic "centered 3-card SaaS grid" with a compact Premiere/Blender-style **Project Launcher** panel: left side shows quick actions (`新建/导入音视频`, `载入演示样例`) as clean list rows; right side shows recent projects list in a tight monochrome table.

## 6. Instant Rejects (Anti-Slop Negative Constraints)
- NO `bg-slate-900`, `bg-slate-950`, `bg-blue-600`, `bg-blue-500`, `bg-amber-500` solid fills.
- NO glowing shadows, drop-shadows, gradients, or glassmorphism (`backdrop-blur`).
- NO decorative colorful icon backgrounds (blue/purple/green rounded squares). All UI chrome icons must be monochrome `#999999` (hover `#E4E4E4`), `14px–16px`, stroke-width `1.5`.