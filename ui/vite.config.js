import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base './' so a built bundle also works when opened from a static path / SSH-forwarded port.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, host: true },
});
