import type {
  AgentChatResponse,
  ProjectMode,
  ProjectScript,
  RenderResponse,
  SynthesizeResponse,
} from "../types/script";

export class ApiError extends Error {
  status: number;
  isNotImplemented: boolean;
  isNetworkError: boolean;

  constructor(
    message: string,
    status: number,
    options?: { isNetworkError?: boolean }
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.isNotImplemented = status === 501;
    this.isNetworkError = Boolean(options?.isNetworkError);
  }
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    throw new ApiError(
      err instanceof Error ? err.message : "无法连接到 StageTTS 后端服务",
      0,
      { isNetworkError: true }
    );
  }

  if (!response.ok) {
    let detail = `请求失败 (${response.status})`;
    try {
      const data = (await response.json()) as { detail?: string };
      if (data?.detail) {
        detail = data.detail;
      }
    } catch {
      // ignore non-json error body
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export interface HealthResponse {
  status: string;
  busy: boolean;
  workspace_dir?: string;
  assets_dir?: string;
}

export interface UploadResponse {
  project_id: string;
  media_url: string;
  duration: number;
}

export interface ParseStatusResponse {
  active: boolean;
  project_id: string | null;
  stage: string;
  logs: string[];
  log_count?: number;
  error: string | null;
  updated_at: number;
}

export function checkHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export interface GpuDevice {
  name: string;
  used_mib: number;
  total_mib: number;
}

export function fetchGpuStatus(signal: AbortSignal): Promise<{ devices: GpuDevice[] }> {
  return requestJson<{ devices: GpuDevice[] }>("/api/gpu", { signal, cache: "no-store" });
}

export async function fetchProjects(): Promise<string[]> {
  const data = await requestJson<{ projects: string[] }>("/api/projects");
  return data.projects;
}

export function uploadProjectMedia(
  file: File,
  mode: ProjectMode = "video_dubbing",
  projectId = ""
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  if (projectId) {
    form.append("project_id", projectId);
  }
  return requestJson<UploadResponse>("/api/projects/upload", {
    method: "POST",
    body: form,
  });
}

export function fetchProjectScript(projectId: string): Promise<ProjectScript> {
  return requestJson<ProjectScript>(
    `/api/projects/${encodeURIComponent(projectId)}/script`
  );
}

export function saveProjectScript(script: ProjectScript): Promise<ProjectScript> {
  return requestJson<ProjectScript>(
    `/api/projects/${encodeURIComponent(script.project_id)}/script`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(script),
    }
  );
}

export function parseProjectMedia(
  projectId: string,
  options: { target_lang: string; compute_type: "float16" | "int8"; allow_demucs_cpu?: boolean }
): Promise<ProjectScript> {
  return requestJson<ProjectScript>(
    `/api/projects/${encodeURIComponent(projectId)}/parse`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    }
  );
}

export function fetchParseStatus(projectId: string): Promise<ParseStatusResponse> {
  return requestJson<ParseStatusResponse>(
    `/api/projects/${encodeURIComponent(projectId)}/parse/status`
  );
}

export function synthesizeSegments(
  projectId: string,
  segmentIds?: number[]
): Promise<SynthesizeResponse> {
  return requestJson<SynthesizeResponse>(
    `/api/projects/${encodeURIComponent(projectId)}/synthesize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segment_ids: segmentIds ?? null }),
    }
  );
}

export function sendAgentChat(
  projectId: string,
  message: string,
  autoSynthesize = true
): Promise<AgentChatResponse> {
  return requestJson<AgentChatResponse>(
    `/api/projects/${encodeURIComponent(projectId)}/agent/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, auto_synthesize: autoSynthesize }),
    }
  );
}

export function renderVideo(
  projectId: string,
  burnWatermark = false
): Promise<RenderResponse> {
  return requestJson<RenderResponse>(
    `/api/projects/${encodeURIComponent(projectId)}/render`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ burn_watermark: burnWatermark }),
    }
  );
}
