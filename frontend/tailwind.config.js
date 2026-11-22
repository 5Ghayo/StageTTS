/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        gap: "rgb(var(--bg-gap-rgb) / <alpha-value>)",
        canvas: "rgb(var(--bg-canvas-rgb) / <alpha-value>)",
        workspace: "rgb(var(--bg-workspace-rgb) / <alpha-value>)",
        trough: "rgb(var(--bg-trough-rgb) / <alpha-value>)",
        panel: "rgb(var(--bg-panel-rgb) / <alpha-value>)",
        surface: "rgb(var(--bg-surface-rgb) / <alpha-value>)",
        hover: "rgb(var(--bg-hover-rgb) / <alpha-value>)",
        active: "rgb(var(--bg-active-rgb) / <alpha-value>)",
        subtle: "var(--border-subtle)",
        strong: "var(--border-strong)",
        primary: "rgb(var(--text-primary-rgb) / <alpha-value>)",
        secondary: "rgb(var(--text-secondary-rgb) / <alpha-value>)",
        muted: "rgb(var(--text-muted-rgb) / <alpha-value>)",
        accent: {
          DEFAULT: "rgb(var(--accent-primary-rgb) / <alpha-value>)",
          hover: "rgb(var(--accent-primary-hover-rgb) / <alpha-value>)",
        },
        status: {
          ready: "rgb(var(--status-ready-rgb) / <alpha-value>)",
          warning: "rgb(var(--status-warning-rgb) / <alpha-value>)",
          error: "rgb(var(--status-error-rgb) / <alpha-value>)",
        },
        playhead: "rgb(var(--playhead-rgb) / <alpha-value>)",
        speaker: {
          a: "rgb(var(--speaker-a-rgb) / <alpha-value>)",
          b: "rgb(var(--speaker-b-rgb) / <alpha-value>)",
          c: "rgb(var(--speaker-c-rgb) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          '"PingFang SC"',
          '"Microsoft YaHei"',
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      borderRadius: {
        control: "4px",
      },
    },
  },
  plugins: [],
};
