import type { Config } from "tailwindcss";

// Dark mode is the default in this research tool, but we keep it as a
// `class` strategy so users can toggle to light without rebuilding.
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // A small, deliberate accent palette. Names map to *meaning* in
        // the GUI (status colors, surfaces) rather than to literal hues,
        // so dark/light mode swap can target them centrally.
        surface: {
          0: "rgb(var(--surface-0) / <alpha-value>)",
          1: "rgb(var(--surface-1) / <alpha-value>)",
          2: "rgb(var(--surface-2) / <alpha-value>)",
          border: "rgb(var(--surface-border) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--ink) / <alpha-value>)",
          muted: "rgb(var(--ink-muted) / <alpha-value>)",
          subtle: "rgb(var(--ink-subtle) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          muted: "rgb(var(--accent-muted) / <alpha-value>)",
        },
        ok: "rgb(var(--status-ok) / <alpha-value>)",
        warn: "rgb(var(--status-warn) / <alpha-value>)",
        err: "rgb(var(--status-err) / <alpha-value>)",
        info: "rgb(var(--status-info) / <alpha-value>)",
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "Inter",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
