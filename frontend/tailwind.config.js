/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        abyss: '#040b17',   // page background (deep navy)
        deep: '#071426',    // header / wells
        panel: '#0b1c33',   // cards (dark blue)
        raised: '#10294a',  // hover / raised surfaces
        rule: '#173456',    // borders
        ink: '#e6f1ff',     // primary text (light blue-white)
        muted: '#8ba6c6',   // secondary text (blue-grey)
        aqua: '#22d3ee',    // primary accent (cyan)
        teal: '#2dd4bf',    // secondary accent
        sky: '#7dd3fc',     // light blue
        indigoish: '#818cf8',
        warn: '#f59e0b',    // red/orange are reserved for anomalies + warnings
        high: '#f97316',
        crit: '#ef4444',
      },
      fontFamily: {
        sans: ['"Inter"', 'system-ui', 'Segoe UI', 'sans-serif'],
        display: ['"Space Grotesk"', '"Inter"', 'system-ui', 'sans-serif'],
      },
      boxShadow: { glow: '0 0 0 1px rgba(34,211,238,0.18), 0 8px 30px rgba(2,10,25,0.55)' },
    },
  },
  plugins: [],
}
