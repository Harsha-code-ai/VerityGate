/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#EEF2F0", panel: "#FFFFFF", ink: "#17252A", muted: "#4F6168", line: "#CBD5D1",
        brand: "#25456B", ok: "#1F7A5A", okbg: "#E3F2EC", bad: "#B3261E", badbg: "#FBE9E7", warn: "#80540A", warnbg: "#F8EFD9",
      },
      fontFamily: { sans: ['"Public Sans Variable"', "system-ui", "sans-serif"], serif: ['"Source Serif 4"', "Georgia", "serif"] },
    },
  },
  plugins: [],
};
