import { create } from "zustand";
import { createDemoScriptCopy } from "../data/demoScript";
import {
  ApiError,
  checkHealth,
  fetchProjectScript,
  fetchProjects,
  saveProjectScript as apiSaveProjectScript,
  synthesizeSegments as apiSynthesizeSegments,
  translateProject,
  uploadProjectMedia,
} from "../services/api";
import type {
  CharacterConfig,
  ProjectMode,
  ProjectScript,
  Segment,
  SegmentPatch,
} from "../types/script";
import { synthesisIssues } from "../utils/synthesisIssues";

/**
 * 为静态音频/视频资源 URL 拼接防缓存时间戳（ARCHITECTURE.md 铁律 #8）。
 * 防止单句重合成覆盖本地文件后，浏览器仍播放缓存的旧音频。
 */
export function getCacheBustedUrl(
  rawUrl: string | null | undefined,
  versionToken?: number
): string | null {
  if (!rawUrl) {
    return null;
  }
  if (!versionToken) {
    return rawUrl;
  }
  const separator = rawUrl.includes("?") ? "&" : "?";
  return `${rawUrl}${separator}t=${versionToken}`;
}

export type BackendStatus = "connecting" | "online" | "offline";

export interface TrackStates {
  videoMuted: boolean;
  dubMuted: boolean;
  dubSolo: boolean;
  vocalsMuted: boolean;
  bgmMuted: boolean;
}

interface ProjectState {
  script: ProjectScript | null;
  mode: ProjectMode;
  selectedSegmentId: number | null;
  currentTime: number;
  isPlaying: boolean;
  isLoading: boolean;
  isSynthesizing: boolean;
  isDirty: boolean;
  error: string | null;
  lastSyncMessage: string | null;
  backendStatus: BackendStatus;
  gpuBusy: boolean;
  projectList: string[];
  trackStates: TrackStates;
  timelineZoom: number;
  /** 记录每个片段最后一次重合成的时间戳，用于音频防缓存 */
  audioVersionMap: Record<number, number>;

  initWorkspace: () => Promise<void>;
  loadDemoProject: () => Promise<void>;
  closeProject: () => void;
  importMediaFile: (file: File) => Promise<void>;
  setScript: (script: ProjectScript) => void;
  setMode: (mode: ProjectMode) => void;
  selectSegment: (segmentId: number | null) => void;
  stepSegment: (delta: -1 | 1) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setTimelineZoom: (zoom: number) => void;
  toggleTrackState: (key: keyof TrackStates) => void;
  bumpAudioVersion: (segmentIds: number[]) => void;
  updateSegmentLocal: (patch: SegmentPatch) => void;
  batchUpdateSegmentsLocal: (patches: SegmentPatch[]) => number[];
  updateBgmVolume: (volume: number) => void;
  updateCharacter: (
    speakerKey: string,
    patch: Partial<CharacterConfig>
  ) => void;
  addCharacter: (name: string) => string | null;
  mergeCharacters: (source: string, destination: string) => void;
  updateTranslationSettings: (sourceLang: string | null, glossary: Record<string, string>) => void;
  retranslate: () => Promise<string>;
  loadProjectScript: (projectId: string) => Promise<void>;
  saveProjectScript: () => Promise<void>;
  synthesizeSegment: (segmentIds?: number[]) => Promise<{
    simulated: boolean;
    message: string;
    issues?: string[];
  }>;
}

const DEFAULT_TRACK_STATES: TrackStates = {
  videoMuted: false,
  dubMuted: false,
  dubSolo: false,
  vocalsMuted: false,
  bgmMuted: false,
};

function invalidateAudio(seg: Segment): Segment {
  return {
    ...seg,
    status: seg.keep_original ? "ready" : "pending",
    tts_wav: null,
    actual_duration: seg.keep_original ? seg.end - seg.start : null,
    applied_speed_factor: null,
  };
}

function applyPatchToSegments(
  script: ProjectScript,
  patches: SegmentPatch[]
): { nextScript: ProjectScript; markedPendingIds: number[] } {
  const resynthFields: Array<keyof SegmentPatch & keyof Segment> = [
    "speaker",
    "target_text",
    "target_lang",
    "emotion",
    "keep_original",
    "speed_factor",
  ];
  const patchMap = new Map<number, SegmentPatch>();
  for (const p of patches) {
    patchMap.set(p.id, p);
  }
  const markedPendingIds: number[] = [];

  const segments = script.segments.map((seg) => {
    const patch = patchMap.get(seg.id);
    if (!patch) {
      return seg;
    }
    const { id: _ignoreId, ...fields } = patch;
    const cleanFields = Object.fromEntries(
      Object.entries(fields).filter(([, value]) => value !== undefined)
    ) as Partial<Segment>;
    if (cleanFields.speaker !== undefined) {
      if (!Object.prototype.hasOwnProperty.call(script.characters, cleanFields.speaker)) return seg;
      cleanFields.speaker_needs_review ??= false;
    }
    if (cleanFields.orig_text !== undefined && cleanFields.orig_text !== seg.orig_text) {
      cleanFields.alignment_needs_review ??= true;
    }
    const updated = { ...seg, ...cleanFields };
    const affectsAudio = resynthFields.some((key) => updated[key] !== seg[key]);
    if (affectsAudio && !updated.keep_original) {
      markedPendingIds.push(seg.id);
    }
    return affectsAudio ? invalidateAudio(updated) : updated;
  });

  return {
    nextScript: {
      ...script,
      segments,
    },
    markedPendingIds,
  };
}

function probeBrowserMediaDuration(objectUrl: string, isAudioOnly: boolean): Promise<number> {
  return new Promise((resolve) => {
    const el = document.createElement(isAudioOnly ? "audio" : "video");
    el.preload = "metadata";
    el.onloadedmetadata = () => {
      const d = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : 15;
      resolve(Number(d.toFixed(2)));
    };
    el.onerror = () => resolve(15);
    el.src = objectUrl;
  });
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  script: null,
  mode: "video_dubbing",
  selectedSegmentId: null,
  currentTime: 0,
  isPlaying: false,
  isLoading: false,
  isSynthesizing: false,
  isDirty: false,
  error: null,
  lastSyncMessage: null,
  backendStatus: "connecting",
  gpuBusy: false,
  projectList: [],
  trackStates: { ...DEFAULT_TRACK_STATES },
  timelineZoom: 72,
  audioVersionMap: {},

  initWorkspace: async () => {
    set({ backendStatus: "connecting", error: null });
    try {
      const health = await checkHealth();
      const projects = await fetchProjects().catch(() => []);
      set({
        backendStatus: "online",
        gpuBusy: health.busy,
        projectList: projects,
        lastSyncMessage: "后端服务在线，等待选择或创建项目",
      });
    } catch {
      set({
        backendStatus: "offline",
        gpuBusy: false,
        projectList: [],
        error: null,
        lastSyncMessage: "离线模式：可导入本地媒体或载入演示样例",
      });
    }
  },

  loadDemoProject: async () => {
    const { backendStatus } = get();
    set({ isLoading: true, error: null });
    if (backendStatus === "online") {
      try {
        const data = await fetchProjectScript("proj_demo_01");
        set({
          script: data,
          mode: data.mode,
          selectedSegmentId: data.segments[0]?.id ?? null,
          currentTime: 0,
          isPlaying: false,
          isLoading: false,
          isDirty: false,
          lastSyncMessage: "已载入项目 proj_demo_01",
        });
        return;
      } catch {
        // 若后端暂无该目录则回退内置演示剧本
      }
    }
    const fresh = createDemoScriptCopy();
    set({
      script: fresh,
      mode: fresh.mode,
      selectedSegmentId: fresh.segments[0]?.id ?? null,
      currentTime: 0,
      isPlaying: false,
      isLoading: false,
      isDirty: false,
      audioVersionMap: {},
      lastSyncMessage: "已载入内置演示样例",
    });
  },

  closeProject: () =>
    set({
      script: null,
      selectedSegmentId: null,
      currentTime: 0,
      isPlaying: false,
      isDirty: false,
      error: null,
      lastSyncMessage: null,
    }),

  importMediaFile: async (file) => {
    set({ isLoading: true, error: null });
    const isAudio = file.type.startsWith("audio/");
    const targetMode: ProjectMode = isAudio ? "virtual_stage" : "video_dubbing";

    try {
      const uploaded = await uploadProjectMedia(file, targetMode);
      const created = await fetchProjectScript(uploaded.project_id);
      const projects = await fetchProjects().catch(() => [uploaded.project_id]);
      set({
        script: created,
        mode: created.mode,
        selectedSegmentId: created.segments[0]?.id ?? null,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        backendStatus: "online",
        projectList: projects,
        lastSyncMessage: `已上传并初始化项目 ${uploaded.project_id}`,
      });
      return;
    } catch {
      // 后端离线时构建前端临时本地可预览项目
      const objectUrl = URL.createObjectURL(file);
      const duration = await probeBrowserMediaDuration(objectUrl, isAudio);
      const cleanName =
        file.name.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") ||
        `local_${Date.now()}`;
      const localScript: ProjectScript = {
        project_id: cleanName,
        mode: targetMode,
        media: {
          source_video: isAudio ? null : objectUrl,
          vocals_wav: isAudio ? objectUrl : null,
          bgm_wav: null,
          output_video: null,
          duration,
          bgm_volume: 0.75,
        },
        characters: {
          SPEAKER_00: {
            display_name: "主讲人",
            color: "#5B8DEF",
            gpt_weights: null,
            sovits_weights: null,
            emotion_refs: {},
            avatar: { type: "live2d", model_url: "/assets/models/host_a/model3.json" },
          },
        },
        segments: [
          {
            id: 0,
            start: 0,
            end: Math.min(duration, 4.0),
            speaker: "SPEAKER_00",
            orig_text: "",
            target_text: "请点击“精修”编辑本句台词，或连接后端自动识别分轨。",
            target_lang: "zh",
            emotion: "neutral",
            action: "nod",
            keep_original: false,
            speed_factor: 1.0,
            tts_wav: null,
            actual_duration: null,
            status: "pending",
          },
        ],
      };

      set({
        script: localScript,
        mode: targetMode,
        selectedSegmentId: 0,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        lastSyncMessage: `已导入本地媒体 ${file.name}（离线预览）`,
      });
    }
  },

  setScript: (script) =>
    set({
      script,
      mode: script.mode,
      selectedSegmentId: script.segments[0]?.id ?? null,
      error: null,
      isDirty: false,
    }),

  setMode: (mode) =>
    set((state) => ({
      mode,
      isDirty: Boolean(state.script),
      script: state.script ? { ...state.script, mode } : null,
    })),

  selectSegment: (segmentId) => set({ selectedSegmentId: segmentId }),

  stepSegment: (delta) =>
    set((state) => {
      const segments = state.script?.segments ?? [];
      if (segments.length === 0) {
        return state;
      }
      const currentIdx = segments.findIndex(
        (seg) => seg.id === state.selectedSegmentId
      );
      const nextIdx =
        currentIdx === -1
          ? 0
          : Math.max(0, Math.min(segments.length - 1, currentIdx + delta));
      const target = segments[nextIdx];
      return {
        selectedSegmentId: target.id,
        currentTime: target.start,
      };
    }),

  setCurrentTime: (time) =>
    set((state) => {
      const maxDuration = state.script?.media.duration ?? 30;
      const clamped = Math.max(0, Math.min(maxDuration, time));
      return { currentTime: clamped };
    }),

  setIsPlaying: (playing) => set({ isPlaying: playing }),

  setTimelineZoom: (zoom) =>
    set({ timelineZoom: Math.max(30, Math.min(180, Math.round(zoom))) }),

  toggleTrackState: (key) =>
    set((state) => ({
      trackStates: {
        ...state.trackStates,
        [key]: !state.trackStates[key],
      },
    })),

  bumpAudioVersion: (segmentIds) =>
    set((state) => {
      const now = Date.now();
      const next = { ...state.audioVersionMap };
      for (const id of segmentIds) {
        next[id] = now;
      }
      return { audioVersionMap: next };
    }),

  updateSegmentLocal: (patch) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      const { nextScript } = applyPatchToSegments(state.script, [patch]);
      return {
        script: nextScript,
        isDirty: true,
      };
    }),

  batchUpdateSegmentsLocal: (patches) => {
    const state = get();
    if (!state.script || patches.length === 0) {
      return [];
    }
    const { nextScript, markedPendingIds } = applyPatchToSegments(
      state.script,
      patches
    );
    set({
      script: nextScript,
      isDirty: true,
    });
    return markedPendingIds;
  },

  updateBgmVolume: (volume) =>
    set((state) => {
      if (!state.script) {
        return state;
      }
      const clamped = Math.max(0, Math.min(2, Number(volume.toFixed(2))));
      return {
        isDirty: true,
        script: {
          ...state.script,
          media: {
            ...state.script.media,
            bgm_volume: clamped,
          },
        },
      };
    }),

  updateCharacter: (speakerKey, patch) =>
    set((state) => {
      const script = state.script;
      const previous = script?.characters[speakerKey];
      if (!script || !previous) {
        return state;
      }
      const character = { ...previous, ...patch };
      const voiceChanged = character.gpt_weights !== previous.gpt_weights
        || character.sovits_weights !== previous.sovits_weights
        || character.emotion_refs !== previous.emotion_refs;
      return {
        isDirty: true,
        script: {
          ...script,
          characters: {
            ...script.characters,
            [speakerKey]: character,
          },
          segments: voiceChanged
            ? script.segments.map((seg) => seg.speaker === speakerKey && !seg.keep_original ? invalidateAudio(seg) : seg)
            : script.segments,
        },
      };
    }),

  addCharacter: (name) => {
    const script = get().script;
    const displayName = name.trim();
    if (!script || !displayName) return null;
    let index = Object.keys(script.characters).length + 1;
    while (script.characters[`character_${index}`]) index += 1;
    const speaker = `character_${index}`;
    set({
      isDirty: true,
      script: {
        ...script,
        characters: {
          ...script.characters,
          [speaker]: {
            display_name: displayName,
            speaking_style: "",
            color: "#3b82f6",
            gpt_weights: null,
            sovits_weights: null,
            emotion_refs: {},
            avatar: null,
          },
        },
      },
    });
    return speaker;
  },

  mergeCharacters: (source, destination) => set((state) => {
    const script = state.script;
    if (!script?.characters[source] || !script.characters[destination] || source === destination) return state;
    const patches = script.segments
      .filter((seg) => seg.speaker === source)
      .map((seg) => ({ id: seg.id, speaker: destination, speaker_needs_review: seg.speaker_needs_review ?? true }));
    const { nextScript } = applyPatchToSegments(script, patches);
    const characters = { ...nextScript.characters };
    delete characters[source];
    return { isDirty: true, script: { ...nextScript, characters } };
  }),

  updateTranslationSettings: (sourceLang, glossary) => set((state) => state.script ? {
    script: { ...state.script, source_lang: sourceLang, glossary },
    isDirty: true,
  } : state),

  retranslate: async () => {
    const { script, isDirty, backendStatus, isLoading, isSynthesizing } = get();
    if (!script || backendStatus !== "online") return "请连接后端后重译";
    if (isLoading || isSynthesizing) return "请等待当前操作完成";
    set({ isLoading: true, error: null });
    try {
      if (isDirty) await apiSaveProjectScript(script);
      const result = await translateProject(script.project_id);
      if (get().script !== script) return "重译已在后端完成，当前编辑保留在本地";
      const message = `已翻译 ${result.translated_ids.length} 条，保留 ${result.skipped_count} 条`;
      set({ script: result.script, isDirty: false, lastSyncMessage: message });
      return message;
    } catch (err) {
      return err instanceof Error ? err.message : "重译失败";
    } finally {
      set({ isLoading: false });
    }
  },

  loadProjectScript: async (projectId) => {
    set({ isLoading: true, error: null });
    try {
      const data = await fetchProjectScript(projectId);
      set({
        script: data,
        mode: data.mode,
        selectedSegmentId: data.segments[0]?.id ?? null,
        currentTime: 0,
        isPlaying: false,
        isLoading: false,
        isDirty: false,
        backendStatus: "online",
        lastSyncMessage: `已加载项目 ${projectId}`,
      });
    } catch (err) {
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "加载剧本失败",
      });
    }
  },

  saveProjectScript: async () => {
    const { script, backendStatus } = get();
    if (!script) {
      return;
    }
    if (backendStatus === "offline") {
      set({
        isDirty: false,
        lastSyncMessage: "已保存至本地会话",
      });
      return;
    }
    set({ isLoading: true, error: null });
    try {
      const saved = await apiSaveProjectScript(script);
      if (get().script !== script) {
        set({ isLoading: false });
        return;
      }
      set({
        script: saved,
        isLoading: false,
        isDirty: false,
        backendStatus: "online",
        lastSyncMessage: "已保存至后端剧本文件",
      });
    } catch (err) {
      if (err instanceof ApiError && err.isNetworkError) {
        set({
          isLoading: false,
          backendStatus: "offline",
          isDirty: false,
          lastSyncMessage: "后端暂不可达，已暂存本地",
        });
        return;
      }
      set({
        isLoading: false,
        error: err instanceof Error ? err.message : "保存失败",
      });
    }
  },

  synthesizeSegment: async (segmentIds) => {
    const state = get();
    const { script, backendStatus } = state;
    if (!script) {
      return { simulated: true, message: "请先打开项目" };
    }

    const targetIds =
      segmentIds && segmentIds.length > 0
        ? segmentIds
        : script.segments
            .filter((seg) => !seg.keep_original && seg.status !== "ready")
            .map((seg) => seg.id);

    if (targetIds.length === 0) {
      return { simulated: false, message: "没有待合成的片段" };
    }

    const applyLocalSimulatedSynth = (msg: string) => {
      const targetSet = new Set(targetIds);
      const latestScript = get().script;
      if (!latestScript) {
        set({ isSynthesizing: false });
        return { simulated: true, message: "无有效剧本" };
      }

      const nextSegments = latestScript.segments.map((seg) => {
        if (!targetSet.has(seg.id) || seg.keep_original) {
          return seg;
        }
        const slot = Math.max(0.4, seg.end - seg.start);
        const estimated = Math.min(
          slot * 1.12,
          Math.max(0.8, seg.target_text.length / 4.8 / Math.max(0.5, seg.speed_factor))
        );
        const paddedId = String(seg.id).padStart(4, "0");
        return {
          ...seg,
          tts_wav:
            seg.tts_wav ??
            `/workspace/${latestScript.project_id}/tts/seg_${paddedId}.wav`,
          actual_duration: Number(estimated.toFixed(2)),
          status: "ready" as const,
        };
      });

      get().bumpAudioVersion(targetIds);
      set({
        script: { ...latestScript, segments: nextSegments },
        isSynthesizing: false,
        lastSyncMessage: `演示合成完成（#${targetIds.join(", #")}）`,
      });
      return {
        simulated: true,
        message: msg,
      };
    };

    if (backendStatus === "offline") {
      return applyLocalSimulatedSynth("离线模式：已完成演示合成");
    }

    set({ isSynthesizing: true, error: null });
    try {
      if (state.isDirty) {
        await apiSaveProjectScript(script);
      }
      const res = await apiSynthesizeSegments(script.project_id, targetIds);
      const issues = synthesisIssues(res);
      const message = `已合成 ${res.synthesized_ids.length} 条${issues.length ? ` · ${issues[0]}` : ""}`;
      get().bumpAudioVersion(res.synthesized_ids);
      set({
        script: res.script,
        isSynthesizing: false,
        isDirty: false,
        backendStatus: "online",
        lastSyncMessage: message,
      });
      return {
        simulated: false,
        message,
        issues,
      };
    } catch (err) {
      const isUnavailableOrOffline =
        err instanceof ApiError &&
        (err.status === 502 ||
          err.status === 504 ||
          err.isNotImplemented ||
          err.isNetworkError);
      if (!isUnavailableOrOffline) {
        set({
          isSynthesizing: false,
          error: err instanceof Error ? err.message : "合成失败",
        });
        return {
          simulated: false,
          message: err instanceof Error ? err.message : "合成失败",
          issues: [err instanceof Error ? err.message : "合成失败"],
        };
      }
      return applyLocalSimulatedSynth("语音服务未启动，已执行本地演示合成");
    }
  },
}));
