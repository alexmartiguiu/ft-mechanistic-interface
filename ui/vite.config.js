import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base './' so a built bundle also works when opened from a static path / SSH-forwarded port.
export default defineConfig({
  base: "./",
  plugins: [react()],
  // allowedHosts: true so an ngrok/proxy Host header isn't rejected.
  server: { port: 5173, host: true, allowedHosts: true },
  preview: { port: 4173, host: true, allowedHosts: true },
});
