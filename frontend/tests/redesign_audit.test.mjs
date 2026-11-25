import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

// ============================================================================
// Comprehensive Frontend Redesign & Accessibility Quality Audit Test Suite
// ============================================================================

const SRC_DIR = 'src';
const COMPONENTS_DIR = path.join(SRC_DIR, 'components');

const readFile = (filePath) => fs.readFileSync(filePath, 'utf-8');

test('1. Architectural Design System & Token Audit', () => {
  const tailwindConfig = readFile('tailwind.config.js');
  const indexCss = readFile(path.join(SRC_DIR, 'index.css'));

  // National micro-accents and chakra foundations
  assert.ok(tailwindConfig.includes('saffron'), 'Tailwind must configure saffron micro-accents');
  assert.ok(tailwindConfig.includes('chakra'), 'Tailwind must configure chakra/ashoka navy palette');
  assert.ok(tailwindConfig.includes('emerald'), 'Tailwind must configure emerald palette');
  assert.ok(tailwindConfig.includes('indic'), 'Tailwind must configure Indic cultural tokens');
  assert.ok(tailwindConfig.includes('brand'), 'Tailwind must configure brand colors');

  // Indic typography stacks
  assert.ok(tailwindConfig.includes('fontFamily'), 'Tailwind must specify custom font families');
  assert.ok(tailwindConfig.includes('indic'), 'Tailwind must define font-indic typography');

  // CSS Web Fonts & Glassmorphism
  assert.ok(indexCss.includes('fonts.googleapis.com'), 'CSS must load Noto Sans Indic web fonts');
  assert.ok(indexCss.includes('glass-panel'), 'CSS must define glass-panel utility');
  assert.ok(indexCss.includes('glass-panel-dark'), 'CSS must define glass-panel-dark utility');
  assert.ok(indexCss.includes('::-webkit-scrollbar'), 'CSS must theme custom scrollbars');
});

test('2. Navbar & App Shell Audit', () => {
  const appTsx = readFile(path.join(SRC_DIR, 'App.tsx'));
  const navbarTsx = readFile(path.join(COMPONENTS_DIR, 'Navbar.tsx'));

  // App shell features
  assert.ok(appTsx.includes('Navbar'), 'App must render Navbar');
  assert.ok(appTsx.includes('Dashboard'), 'App must integrate Dashboard');
  assert.ok(appTsx.includes('TextTranslator'), 'App must integrate TextTranslator');
  assert.ok(appTsx.includes('ISLTranslator'), 'App must integrate ISLTranslator');
  assert.ok(appTsx.includes('VoiceTranslator'), 'App must integrate VoiceTranslator');
  assert.ok(appTsx.includes('TranslationHistory'), 'App must integrate TranslationHistory');
  assert.ok(appTsx.includes('SettingsModal'), 'App must integrate SettingsModal');
  assert.ok(appTsx.includes('ToastProvider'), 'App must wrap with ToastProvider');
  assert.ok(appTsx.includes('ErrorBoundary'), 'App must protect views with ErrorBoundary');

  // Navbar Chakra SVG & Navigation
  assert.ok(navbarTsx.includes('svg'), 'Navbar must render Chakra emblem SVG');
  assert.ok(navbarTsx.includes('spoke') || navbarTsx.includes('line'), 'Navbar must contain Chakra spokes/lines');
  assert.ok(navbarTsx.includes('networkStatus'), 'Navbar must track network telemetry');
  assert.ok(navbarTsx.includes('popover') || navbarTsx.includes('isStatusPopoverOpen'), 'Navbar must support engine telemetry popover');
  assert.ok(navbarTsx.includes('isMobileMenuOpen'), 'Navbar must support mobile drawer navigation');
});

test('3. TextTranslator Redesign Audit (Dual-Card, Engine Switcher, Indic Tools)', () => {
  const textTranslator = readFile(path.join(COMPONENTS_DIR, 'TextTranslator.tsx'));

  // Dual-Card Responsive Grid
  assert.ok(
    textTranslator.includes('lg:grid-cols-2') || textTranslator.includes('grid-cols-1 lg:grid-cols-2'),
    'TextTranslator must render responsive side-by-side grid on desktop and stacked on mobile'
  );

  // Engine Switcher with Model 3 AI4Bharat Pralekha as primary default
  assert.ok(textTranslator.includes("'pralekha'"), 'TextTranslator must support pralekha engine');
  assert.ok(textTranslator.includes('AI4Bharat') || textTranslator.includes('Pralekha'), 'TextTranslator must prominently highlight AI4Bharat Pralekha');
  assert.ok(textTranslator.includes('PRALEKHA_SUPPORTED_LANG_CODES'), 'TextTranslator must maintain Model 3 supported language set');

  // Searchable language select with native scripts
  assert.ok(textTranslator.includes('SearchableLanguageSelect'), 'TextTranslator must use searchable language dropdown');
  assert.ok(textTranslator.includes('font-indic'), 'TextTranslator must format Indic scripts with font-indic');

  // Essential tools: Copy, Audio TTS, Transliteration, Character/Word Count
  assert.ok(textTranslator.includes('playSpeechAudio'), 'TextTranslator must support TTS audio playback');
  assert.ok(textTranslator.includes('transliterate') || textTranslator.includes('Transliteration'), 'TextTranslator must support Indic transliteration');
  assert.ok(textTranslator.includes('navigator.clipboard') || textTranslator.includes('Copy'), 'TextTranslator must support clipboard copy');
  assert.ok(textTranslator.includes('confidence') || textTranslator.includes('Confidence'), 'TextTranslator must display confidence telemetry');
});

test('4. Dashboard Redesign Audit (Hero, 22 Languages Explorer, System Health)', () => {
  const dashboard = readFile(path.join(COMPONENTS_DIR, 'Dashboard.tsx'));

  // Hero section and quick translate widget
  assert.ok(dashboard.includes('Quick-Translate') || dashboard.includes('quickInputText') || dashboard.includes('handleHeroSubmit'), 'Dashboard must feature a quick translate widget in hero');
  assert.ok(dashboard.includes('Uniting Voice, Script & Sign Across Bharat') || dashboard.includes('Universal Indian Language'), 'Dashboard must present authoritative headline');

  // 22 Scheduled Languages Explorer
  assert.ok(dashboard.includes('Indo-Aryan') && dashboard.includes('Dravidian'), 'Dashboard must support language family filters');
  assert.ok(dashboard.includes('font-indic'), 'Dashboard language explorer must render native script typography');
  assert.ok(dashboard.includes('playSpeechAudio'), 'Dashboard language explorer must support cultural sample audio playback');

  // Multimodal highlight cards
  assert.ok(dashboard.includes('ISL') || dashboard.includes('Sign Language'), 'Dashboard must highlight ISL Vision');
  assert.ok(dashboard.includes('Voice') || dashboard.includes('Duplex'), 'Dashboard must highlight Voice Duplex');
});

test('5. ISLTranslator Redesign Audit (Studio CV, Gauge, Cheat Sheet, Reverse Visualizer)', () => {
  const islTranslator = readFile(path.join(COMPONENTS_DIR, 'ISLTranslator.tsx'));

  // Studio Mode & Computer Vision
  assert.ok(islTranslator.includes('videoRef') && islTranslator.includes('canvasRef'), 'ISLTranslator must manage video and canvas elements');
  assert.ok(islTranslator.includes('recognizeLandmarks'), 'ISLTranslator must interface with landmark recognition API');

  // Circular Gauge & Prediction Readout
  assert.ok(islTranslator.includes('topCandidates') || islTranslator.includes('Circular Confidence'), 'ISLTranslator must display candidates and confidence gauge');
  assert.ok(islTranslator.includes('recentSigns'), 'ISLTranslator must track recent signs stream ticker');

  // Cheat Sheet Drawer
  assert.ok(islTranslator.includes('isCheatSheetOpen'), 'ISLTranslator must include ISL cheat sheet drawer');
  assert.ok(islTranslator.includes('islAlphabet'), 'ISLTranslator must provide ISL alphabet descriptions');

  // Reverse Sign Communication Visualizer
  assert.ok(islTranslator.includes('reverseCanvasRef') || islTranslator.includes('selectedReverseSign'), 'ISLTranslator must include reverse sign visualizer');
});

test('6. VoiceTranslator Redesign Audit (Waveform, Dual Panels, TTS Controls)', () => {
  const voiceTranslator = readFile(path.join(COMPONENTS_DIR, 'VoiceTranslator.tsx'));

  // Audio wave visualizer canvas
  assert.ok(voiceTranslator.includes('canvasRef'), 'VoiceTranslator must feature canvas audio visualizer');
  assert.ok(voiceTranslator.includes('createLinearGradient') || voiceTranslator.includes('roundRect'), 'VoiceTranslator canvas must render gradient audio waveform');

  // Dual speech panels
  assert.ok(voiceTranslator.includes('transcript'), 'VoiceTranslator must display source speech transcript');
  assert.ok(voiceTranslator.includes('translatedResult'), 'VoiceTranslator must display translated target speech');

  // TTS playback controls & speed
  assert.ok(voiceTranslator.includes('speechRate'), 'VoiceTranslator must support playback speed adjustment');
  assert.ok(voiceTranslator.includes('autoPlayTTS'), 'VoiceTranslator must support auto-play toggle');
  assert.ok(voiceTranslator.includes('playSpeechAudio'), 'VoiceTranslator must support audio speech synthesis');

  // Conversational timeline
  assert.ok(voiceTranslator.includes('conversation'), 'VoiceTranslator must maintain conversational timeline');
});

test('7. TranslationHistory Redesign Audit (Table/Card Switcher, Filters, CSV/JSON Export)', () => {
  const history = readFile(path.join(COMPONENTS_DIR, 'TranslationHistory.tsx'));

  // Layout switcher
  assert.ok(history.includes('viewMode'), 'TranslationHistory must support card and table view modes');

  // Multi-dimensional filters
  assert.ok(history.includes('selectedModule'), 'TranslationHistory must filter by module (Text, Voice, ISL)');
  assert.ok(history.includes('favoritesOnly'), 'TranslationHistory must filter by favorites');
  assert.ok(history.includes('selectedSourceLang'), 'TranslationHistory must filter by source language');
  assert.ok(history.includes('selectedTargetLang'), 'TranslationHistory must filter by target language');

  // Export & Clear History
  assert.ok(history.includes('exportToCSV') || history.includes('CSV'), 'TranslationHistory must support CSV export');
  assert.ok(history.includes('exportToJSON') || history.includes('JSON'), 'TranslationHistory must support JSON export');
  assert.ok(history.includes('clearAllHistory') || history.includes('confirmClearOpen'), 'TranslationHistory must support clearing history with confirmation');
});

test('8. Accessibility & High-Contrast Visual Standards Audit', () => {
  const files = [
    'App.tsx',
    'components/Navbar.tsx',
    'components/TextTranslator.tsx',
    'components/Dashboard.tsx',
    'components/ISLTranslator.tsx',
    'components/VoiceTranslator.tsx',
    'components/TranslationHistory.tsx'
  ];

  for (const relPath of files) {
    const content = readFile(path.join(SRC_DIR, relPath));

    // Focus state checks
    assert.ok(
      content.includes('focus:ring') || content.includes('focus:outline') || content.includes('focus-visible'),
      `${relPath} must implement accessible focus ring or outline styling`
    );

    // High contrast text classes
    assert.ok(
      content.includes('text-slate-900') || content.includes('text-slate-800') || content.includes('text-white') || content.includes('text-chakra-950'),
      `${relPath} must maintain high-contrast text color tokens`
    );

    // ARIA or semantic markup
    assert.ok(
      content.includes('aria-') || content.includes('role=') || content.includes('<button') || content.includes('<main'),
      `${relPath} must include semantic HTML elements and ARIA accessibility attributes`
    );
  }
});

test('9. Responsive Viewport Breakpoint Coverage Audit', () => {
  const files = [
    'App.tsx',
    'components/Navbar.tsx',
    'components/TextTranslator.tsx',
    'components/Dashboard.tsx',
    'components/ISLTranslator.tsx',
    'components/VoiceTranslator.tsx',
    'components/TranslationHistory.tsx'
  ];

  for (const relPath of files) {
    const content = readFile(path.join(SRC_DIR, relPath));

    // Must utilize mobile (base), tablet (sm: / md:), and desktop (lg: / xl:) responsive classes
    assert.ok(
      content.includes('sm:') || content.includes('md:') || content.includes('lg:'),
      `${relPath} must include responsive Tailwind breakpoint prefixes (sm:, md:, lg:)`
    );
  }
});
