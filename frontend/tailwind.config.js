/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Indian National Palette & Micro-Accents
        saffron: {
          50: '#fff9f2',
          100: '#fff1e0',
          200: '#ffe2c2',
          300: '#ffc994',
          400: '#ffa857',
          500: '#ff671f', // Primary Saffron / Kesari
          600: '#e65512',
          700: '#bf3f09',
          800: '#992f07',
          900: '#7c2607',
          950: '#441103',
          light: '#ff9933', // Secondary warm Kesari
          DEFAULT: '#ff671f',
        },
        // Ashoka Chakra Navy (National emblem blue & deep state foundation)
        chakra: {
          50: '#eef2ff',
          100: '#e0e7ff',
          200: '#c7d2fe',
          300: '#a5b4fc',
          400: '#818cf8',
          500: '#4338ca',
          600: '#1e3a8a', // Deep Navy
          700: '#06038d', // Ashoka Navy
          800: '#000080', // Royal Ashoka Chakra Navy
          900: '#040254',
          950: '#02012c',
          DEFAULT: '#06038d',
        },
        // Tiranga Emerald (Indian Green foundation)
        emerald: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#138808', // Indian Green
          600: '#046a38', // Deep Tiranga Emerald
          700: '#03522b',
          800: '#033d20',
          900: '#022b17',
          950: '#01180d',
          DEFAULT: '#046a38',
        },
        // Warm Parchment & Manuscript Neutrals
        parchment: {
          50: '#fdfbf7',
          100: '#f8f4ec',
          200: '#efe6d5',
          300: '#e2d4bb',
          400: '#ceb896',
          500: '#ba9b72',
          600: '#a48259',
          700: '#856645',
          800: '#695138',
          900: '#4d3c2a',
          950: '#2c2116',
          DEFAULT: '#fdfbf7',
        },
        // Warm Slate neutrals
        warmSlate: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#0b0f19',
        },
        // Brand Foundation (backward-compatible with existing components)
        brand: {
          50: '#f0f5ff',
          100: '#e0ebff',
          200: '#c7dbfe',
          300: '#a4c4fd',
          400: '#7aa3fb',
          500: '#4f7ef7',
          600: '#335eea',
          700: '#2547d0',
          800: '#1e38a3',
          900: '#1a2e7c',
          950: '#0f173b',
        },
        // Indic Named Tokens
        indic: {
          saffron: '#ff671f',
          kesari: '#ff9933',
          navy: '#06038d',
          chakra: '#000080',
          emerald: '#046a38',
          green: '#138808',
          gold: '#d97706',
          teal: '#0d9488',
          purple: '#7c3aed',
          parchment: '#fdfbf7',
        }
      },
      fontFamily: {
        sans: [
          '"Outfit"',
          '"Noto Sans"',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          '"Noto Sans Devanagari"',
          '"Noto Sans Bengali"',
          '"Noto Sans Tamil"',
          '"Noto Sans Telugu"',
          '"Noto Sans Kannada"',
          '"Noto Sans Malayalam"',
          '"Noto Sans Gujarati"',
          '"Noto Sans Gurmukhi"',
          '"Noto Sans Oriya"',
          'sans-serif'
        ],
        indic: [
          '"Noto Sans Devanagari"',
          '"Kohinoor Devanagari"',
          '"Mangal"',
          '"Noto Sans Bengali"',
          '"Kohinoor Bangla"',
          '"Vrinda"',
          '"Noto Sans Gurmukhi"',
          '"Raavi"',
          '"Noto Sans Gujarati"',
          '"Shruti"',
          '"Noto Sans Oriya"',
          '"Kalinga"',
          '"Noto Sans Tamil"',
          '"Kohinoor Tamil"',
          '"Latha"',
          '"Noto Sans Telugu"',
          '"Kohinoor Telugu"',
          '"Gautami"',
          '"Noto Sans Kannada"',
          '"Tunga"',
          '"Noto Sans Malayalam"',
          '"Kartika"',
          '"Noto Nastaliq Urdu"',
          '"Urdu Typesetting"',
          '"Noto Sans Ol Chiki"',
          '"Nirmala UI"',
          'system-ui',
          'sans-serif'
        ],
        devanagari: [
          '"Noto Sans Devanagari"',
          '"Kohinoor Devanagari"',
          '"Mangal"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        tamil: [
          '"Noto Sans Tamil"',
          '"Kohinoor Tamil"',
          '"Latha"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        telugu: [
          '"Noto Sans Telugu"',
          '"Kohinoor Telugu"',
          '"Gautami"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        bengali: [
          '"Noto Sans Bengali"',
          '"Kohinoor Bangla"',
          '"Vrinda"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        gurmukhi: [
          '"Noto Sans Gurmukhi"',
          '"Raavi"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        gujarati: [
          '"Noto Sans Gujarati"',
          '"Shruti"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        malayalam: [
          '"Noto Sans Malayalam"',
          '"Kartika"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        kannada: [
          '"Noto Sans Kannada"',
          '"Tunga"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        odia: [
          '"Noto Sans Oriya"',
          '"Kalinga"',
          '"Nirmala UI"',
          'sans-serif'
        ],
        nastaliq: [
          '"Noto Nastaliq Urdu"',
          '"Urdu Typesetting"',
          '"Jameel Noori Nastaleeq"',
          'serif'
        ],
        olchiki: [
          '"Noto Sans Ol Chiki"',
          'sans-serif'
        ],
        display: [
          '"Rozha One"',
          '"Plus Jakarta Sans"',
          'Inter',
          'serif'
        ],
        mono: [
          '"JetBrains Mono"',
          '"Fira Code"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace'
        ]
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(15, 23, 42, 0.05)',
        'card': '0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.04)',
        'card-hover': '0 10px 20px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04)',
        'glow-sm': '0 0 15px rgba(51, 94, 234, 0.2)',
        'glow-md': '0 0 25px rgba(51, 94, 234, 0.3)',
        'glow-saffron': '0 0 24px rgba(255, 103, 31, 0.25)',
        'glow-emerald': '0 0 24px rgba(4, 106, 56, 0.25)',
        'glow-chakra': '0 0 24px rgba(6, 3, 141, 0.22)',
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.07)',
      },
      animation: {
        'fadeIn': 'fadeIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'scaleUp': 'scaleUp 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'scaleIn': 'scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'slideDown': 'slideDown 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'slideUp': 'slideUp 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'pulseSlow': 'pulseSlow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-subtle': 'pulseSlow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2.5s ease-in-out infinite',
        'float': 'float 4s ease-in-out infinite',
        'spinSlow': 'spinSlow 20s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleUp: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.96)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSlow: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-6px)' },
        },
        spinSlow: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
      }
    },
  },
  plugins: [],
}
