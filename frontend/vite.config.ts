import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In production the app calls the relative path /api (nginx proxies it to the backend container).
// For `npm run dev`, set VITE_PROXY_TARGET if your backend is not on localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": process.env.VITE_PROXY_TARGET ?? "http://localhost:8000" } },
});
