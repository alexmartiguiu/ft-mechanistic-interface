import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built to dist/ and served at "/" by webui/server.py. In dev, proxy the API to the
// FastAPI process so the React app and matplotlib SVGs share one origin.
export default defineConfig({
  plugins: [react()],
  base: "/",
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/report": "http://127.0.0.1:8000",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
