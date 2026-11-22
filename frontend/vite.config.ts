import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000,
      },
      "/workspace": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/assets": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
