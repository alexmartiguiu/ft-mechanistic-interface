import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base '/' so absolute /api/* paths resolve to the origin and hit the proxy.
// API port overridable with FTMI_API_PORT (default 8000) — lets the dev server point at
// a backend on a non-default port without editing this file.
const apiPort = process.env.FTMI_API_PORT || "8000";
const proxy = {
  "/api": { target: `http://localhost:${apiPort}`, changeOrigin: true },
};

export default defineConfig({
  base: "/",
  plugins: [react()],
  // allowedHosts: true so an ngrok/proxy Host header isn't rejected.
  // proxy /api → uvicorn (ui_backend); streams SSE through for the agent.
  server: { port: 5173, host: true, allowedHosts: true, proxy },
  preview: { port: 4173, host: true, allowedHosts: true, proxy },
});
