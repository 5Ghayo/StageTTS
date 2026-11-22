/**
 * 将高饱和度旧色值映射为设计规范中的低饱和语义角色色，
 * 避免改变后端 script.json 契约同时维持工作台整体克制质感。
 */
const LEGACY_SPEAKER_COLOR_MAP: Record<string, string> = {
  "#3b82f6": "#5B8DEF",
  "#2563eb": "#5B8DEF",
  "#10b981": "#4FA882",
  "#059669": "#4FA882",
  "#f59e0b": "#C99A4E",
  "#d97706": "#C99A4E",
};

export function normalizeSpeakerColor(rawColor?: string | null): string {
  if (!rawColor) {
    return "#5B8DEF";
  }
  const key = rawColor.trim().toLowerCase();
  return LEGACY_SPEAKER_COLOR_MAP[key] ?? rawColor;
}
