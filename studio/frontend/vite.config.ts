import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` serves on :5173 and proxies API/file routes to the BFF.
// Build: emits into the Python package's web/ dir (kept alongside the logos).
const BFF = "http://localhost:8730";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../konfai_studio/web",
    emptyOutDir: false, // keep konfai-logo.png next to the built index.html
  },
  server: {
    port: 5173,
    proxy: {
      "/api": BFF,
      "/files": BFF,
      "/konfai-logo.png": BFF,
    },
  },
});
