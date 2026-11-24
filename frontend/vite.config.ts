import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function resolveBackendTarget(): string {
  const explicit = process.env.VITE_BACKEND_URL?.trim();
  if (explicit) {
    return explicit.replace(/\/+$/, "");
  }
  const rawHost = (process.env.BACKEND_HOST || "127.0.0.1").trim();
  const normalizedHost = rawHost === "0.0.0.0" || rawHost === "::" || !rawHost ? "127.0.0.1" : rawHost;
  const host =
    normalizedHost.includes(":") && !normalizedHost.startsWith("[")
      ? `[${normalizedHost}]`
      : normalizedHost;
  const port = (process.env.BACKEND_PORT || "8000").trim() || "8000";
  return `http://${host}:${port}`;
}

const backendTarget = resolveBackendTarget();

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000,
      },
      "/workspace": {
        target: backendTarget,
        changeOrigin: true,
      },
      "/assets": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
