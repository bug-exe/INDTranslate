import React, { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './components/Dashboard';
import { TextTranslator } from './components/TextTranslator';
import { ISLTranslator } from './components/ISLTranslator';
import { VoiceTranslator } from './components/VoiceTranslator';
import { TranslationHistory } from './components/TranslationHistory';
import { SettingsModal } from './components/SettingsModal';
import { ModelMetrics } from './components/ModelMetrics';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ToastProvider, useToast } from './context/ToastContext';
import { Language, TranslationEngine } from './types';
import { fetchLanguages } from './services/api';
import { 
  LayoutDashboard, 
  Hand, 
  Languages as LangIcon, 
  Mic, 
  History,
  ShieldCheck,
  Cpu,
  Sparkles,
  Command,
  X,
  Gauge,
  LifeBuoy
} from 'lucide-react';

export const AppContent: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [languages, setLanguages] = useState<Language[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [isShortcutsOpen, setIsShortcutsOpen] = useState<boolean>(false);
  const [activeEngine, setActiveEngine] = useState<TranslationEngine>('pralekha');
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  
  // Quick translation bridge
  const [quickText, setQuickText] = useState<string>('');
  const [quickTarget, setQuickTarget] = useState<string>('hi');

  const { showInfo } = useToast();

  // Load languages, theme and engine preferences on mount
  useEffect(() => {
    fetchLanguages()
      .then(setLanguages)
      .catch((err) => console.error('Error loading languages:', err));

    const savedTheme = localStorage.getItem('indtranslate_theme');
    // Default to sleek matte dark mode unless user explicitly selected light
    if (savedTheme === 'light') {
      setIsDarkMode(false);
      document.documentElement.classList.remove('dark');
    } else {
      setIsDarkMode(true);
      document.documentElement.classList.add('dark');
    }

    const savedEngine = localStorage.getItem('indtranslate_engine') as TranslationEngine | null;
    if (savedEngine) {
      setActiveEngine(savedEngine);
    }
  }, []);

  const handleToggleTheme = () => {
    setIsDarkMode((prev) => {
      const next = !prev;
      if (next) {
        document.documentElement.classList.add('dark');
        localStorage.setItem('indtranslate_theme', 'dark');
      } else {
        document.documentElement.classList.remove('dark');
        localStorage.setItem('indtranslate_theme', 'light');
      }
      return next;
    });
  };

  const handleEngineChange = (engine: TranslationEngine) => {
    setActiveEngine(engine);
    localStorage.setItem('indtranslate_engine', engine);
    const engineNames: Record<TranslationEngine, string> = {
      pralekha: 'Locally Trained Model (Pralekha 52M)',
      nllb: 'Meta NLLB-200 Fine-Tuned (600M) [100k+]',
      nllb_base: 'Meta NLLB-200 Base Foundation (600M)',
      universal: 'Universal Auto-Adaptive',
      google: 'Google Cloud NMT'
    };
    showInfo(`Translation Engine switched to ${engineNames[engine] || engine}`);
  };

  const handleQuickTranslate = (text: string, target: string) => {
    setQuickText(text);
    setQuickTarget(target);
    setActiveTab('home');
  };

  const handleLogout = () => {
    showInfo('Logged out of sovereign session. Local session cache cleared.');
    setActiveTab('home');
  };

  // Keyboard navigation shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // If user is currently typing in an input or textarea, ignore single-key shortcuts
    const activeElem = document.activeElement;
    const isTyping = activeElem && (
      activeElem.tagName === 'INPUT' || 
      activeElem.tagName === 'TEXTAREA' || 
      (activeElem as HTMLElement).isContentEditable
    );

    // Escape closes modals
    if (e.key === 'Escape') {
      if (isShortcutsOpen) setIsShortcutsOpen(false);
      if (isSettingsOpen) setIsSettingsOpen(false);
      return;
    }

    // Help modal: '?'
    if (e.key === '?' && !isTyping && !e.altKey && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      setIsShortcutsOpen((prev) => !prev);
      return;
    }

    // Settings modal: Alt+, or Cmd+,
    if ((e.altKey || e.metaKey) && e.key === ',') {
      e.preventDefault();
      setIsSettingsOpen((prev) => !prev);
      return;
    }

    // Tab Navigation: Alt+1..7 or single keys 1..7 when not typing
    const isTabShortcut = (e.altKey || (!isTyping && !e.ctrlKey && !e.metaKey));
    if (isTabShortcut) {
      if (e.key === '1') {
        e.preventDefault();
        setActiveTab('home');
      } else if (e.key === '2') {
        e.preventDefault();
        setActiveTab('isl');
      } else if (e.key === '3') {
        e.preventDefault();
        setActiveTab('text');
      } else if (e.key === '4') {
        e.preventDefault();
        setActiveTab('voice');
      } else if (e.key === '5') {
        e.preventDefault();
        setActiveTab('history');
      } else if (e.key === '6') {
        e.preventDefault();
        setActiveTab('metrics');
      } else if (e.key === '7') {
        e.preventDefault();
        setActiveTab('dashboard');
      }
    }
  }, [isShortcutsOpen, isSettingsOpen]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const mobileNavTabs = [
    { id: 'home', label: 'Studio', icon: LangIcon },
    { id: 'isl', label: 'ISL Vision', icon: Hand },
    { id: 'voice', label: 'Voice', icon: Mic },
    { id: 'history', label: 'History', icon: History },
    { id: 'metrics', label: 'Metrics', icon: Gauge },
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  ];

  return (
    <div className="relative min-h-screen flex flex-col bg-[#0e1015] text-[#f3f4f6] selection:bg-blue-600 selection:text-white transition-colors duration-200">
      
      {/* Matte Dark Ambient Mesh Backdrop */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden="true">
        {/* Electric Blue Subtle Ambient Arc (Center-Top) */}
        <div 
          className="absolute -top-40 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full bg-blue-600/5 blur-3xl" 
        />
        {/* Subtle Saffron Kesari Micro-Warmth (North-West) */}
        <div 
          className="absolute -top-40 -left-40 w-80 h-80 rounded-full bg-orange-600/5 blur-3xl" 
        />
        {/* Subtle Emerald Grounding Glow (North-East) */}
        <div 
          className="absolute -top-40 -right-40 w-80 h-80 rounded-full bg-emerald-600/5 blur-3xl" 
        />
        {/* High-Precision Subtle Dot Matrix */}
        <div 
          className="absolute inset-0 bg-[radial-gradient(#262933_1px,transparent_1px)] [background-size:24px_24px] opacity-30 [mask-image:radial-gradient(ellipse_75%_50%_at_50%_0%,#000_70%,transparent_100%)]" 
        />
      </div>

      {/* Modern Navigation Architecture: Sidebar + Sticky Topbar */}
      <div className="relative z-40">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          openSettings={() => setIsSettingsOpen(true)}
          activeEngine={activeEngine}
          onEngineChange={handleEngineChange}
          isDark={isDarkMode}
          onToggleTheme={handleToggleTheme}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onLogout={handleLogout}
        />
      </div>

      {/* Main Studio Viewport offset for desktop sidebar (lg:pl-64) */}
      <div className="relative z-10 flex-1 flex flex-col lg:pl-64 min-w-0 transition-all duration-200">
        <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 lg:px-8 pt-4 sm:pt-6 pb-24 sm:pb-16">
          <ErrorBoundary>
            <div key={activeTab} className="animate-fadeIn">
              
              {/* Primary Translation Studio View (Default 'home', 'text', or 'documents') */}
              {(activeTab === 'home' || activeTab === 'text' || activeTab === 'documents') && (
                <TextTranslator
                  languages={languages}
                  initialText={quickText}
                  initialTarget={quickTarget}
                />
              )}

              {/* Comprehensive System Dashboard */}
              {activeTab === 'dashboard' && (
                <Dashboard
                  languages={languages}
                  setActiveTab={setActiveTab}
                  onQuickTranslate={handleQuickTranslate}
                />
              )}

              {/* ISL Vision Real-time Tracking */}
              {activeTab === 'isl' && (
                <ISLTranslator
                  languages={languages}
                />
              )}

              {/* Duplex Voice Speech Studio */}
              {activeTab === 'voice' && (
                <VoiceTranslator
                  languages={languages}
                />
              )}

              {/* Translation History & Bookmarks */}
              {(activeTab === 'history' || activeTab === 'downloads' || activeTab === 'bookmarks') && (
                <TranslationHistory />
              )}

              {/* Neural Model Telemetry & Benchmarks */}
              {activeTab === 'metrics' && (
                <ModelMetrics
                  currentEngine={activeEngine}
                  onSelectEngine={handleEngineChange}
                />
              )}

              {/* Support & Accessibility Center */}
              {activeTab === 'support' && (
                <div className="card-matte p-6 sm:p-8 space-y-6 max-w-3xl mx-auto animate-fadeIn">
                  <div className="flex items-center space-x-3 pb-4 border-b border-[#262933]">
                    <div className="w-10 h-10 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                      <LifeBuoy className="w-5 h-5" />
                    </div>
                    <div>
                      <h2 className="text-lg font-bold text-white">Support & Multilingual Resources</h2>
                      <p className="text-xs text-zinc-400">IndTranslate Sovereign Multimodal Accessibility Platform</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                    <div className="p-4 rounded-2xl bg-[#14161d] border border-[#262933] space-y-2">
                      <h4 className="font-semibold text-white flex items-center space-x-2">
                        <Sparkles className="w-4 h-4 text-blue-400" />
                        <span>ISLRTC 2024 Standards</span>
                      </h4>
                      <p className="text-zinc-400 leading-relaxed">
                        Full compliance with Indian Sign Language Research and Training Centre gestures and sign taxonomy.
                      </p>
                    </div>

                    <div className="p-4 rounded-2xl bg-[#14161d] border border-[#262933] space-y-2">
                      <h4 className="font-semibold text-white flex items-center space-x-2">
                        <Cpu className="w-4 h-4 text-emerald-400" />
                        <span>Sovereign Local Engines</span>
                      </h4>
                      <p className="text-zinc-400 leading-relaxed">
                        Pralekha 52M and Meta NLLB-200 models execute securely with on-device privacy guarantee.
                      </p>
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-between">
                    <button
                      onClick={() => setIsShortcutsOpen(true)}
                      className="pill-chip pill-chip-active text-xs"
                    >
                      <Command className="w-3.5 h-3.5" />
                      <span>View Keyboard Shortcuts</span>
                    </button>
                    <button
                      onClick={() => setActiveTab('home')}
                      className="pill-chip text-xs"
                    >
                      <span>Return to Translation Studio &rarr;</span>
                    </button>
                  </div>
                </div>
              )}

            </div>
          </ErrorBoundary>
        </main>

        {/* Sleek Architectural Bharat Sovereign Footer */}
        <footer className="border-t border-[#262933] bg-[#111317]/90 backdrop-blur-md py-6 text-xs text-zinc-400">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
            
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              
              {/* Identity & Mission */}
              <div className="flex items-center space-x-3 text-center md:text-left">
                <div className="w-8 h-8 rounded-xl bg-zinc-900 border border-[#262933] flex items-center justify-center text-white text-sm shadow-sm">
                  🇮🇳
                </div>
                <div>
                  <div className="font-bold text-white text-xs tracking-tight flex items-center space-x-2">
                    <span>IndTranslate Platform</span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
                      Sovereign Architecture
                    </span>
                  </div>
                  <p className="text-zinc-500 text-[11px]">
                    Connecting Voice, Script and Sign Across 11 Scheduled Languages (+ English)
                  </p>
                </div>
              </div>

              {/* Architecture Standards Badges */}
              <div className="flex flex-wrap items-center justify-center gap-2 text-[11px]">
                <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-xl bg-[#181a20] text-zinc-300 font-medium border border-[#262933]">
                  <Cpu className="w-3.5 h-3.5 text-blue-400" />
                  <span>Meta NLLB-200 / Pralekha</span>
                </span>
                <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-xl bg-[#181a20] text-zinc-300 font-medium border border-[#262933]">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                  <span>11 Scheduled Scripts</span>
                </span>
                <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-xl bg-[#181a20] text-zinc-300 font-medium border border-[#262933]">
                  <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
                  <span>Zero Data Leakage</span>
                </span>
              </div>

            </div>

            {/* Sub-Footer Line */}
            <div className="pt-3 border-t border-[#1c1f26] flex flex-col sm:flex-row items-center justify-between gap-2 text-zinc-500 text-[11px]">
              <div className="flex items-center space-x-2">
                <span>ISLRTC 2024 Compliant &bull; High-Precision Neural Translation Stack</span>
                <button
                  onClick={() => setIsShortcutsOpen(true)}
                  className="underline hover:text-zinc-300 ml-2 focus:outline-none focus:ring-1 focus:ring-blue-500 rounded"
                  aria-label="View keyboard shortcuts"
                >
                  Shortcuts (?)
                </button>
              </div>
              <div>
                v1.0.0 Sovereign &bull; Matte Dark Studio
              </div>
            </div>

          </div>
        </footer>
      </div>

      {/* Mobile Bottom Quick-Access Action Dock */}
      <nav 
        aria-label="Mobile Bottom Navigation"
        className="lg:hidden fixed bottom-0 inset-x-0 z-40 bg-[#111317]/95 backdrop-blur-md border-t border-[#262933] shadow-2xl px-2 py-1.5 flex items-center justify-around"
      >
        {mobileNavTabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex flex-col items-center justify-center py-1 px-2.5 rounded-xl transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                isActive
                  ? 'text-white font-bold scale-105'
                  : 'text-zinc-500 hover:text-zinc-300 font-medium'
              }`}
              aria-label={`Navigate to ${tab.label}`}
              aria-current={isActive ? 'page' : undefined}
            >
              <div className={`p-1.5 rounded-lg transition-colors ${isActive ? 'bg-blue-600 text-white shadow-sm' : 'text-zinc-400'}`}>
                <Icon className="w-4 h-4" />
              </div>
              <span className="text-[10px] mt-0.5 tracking-tight">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Platform Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        languages={languages}
        currentEngine={activeEngine}
        onEngineChange={handleEngineChange}
      />

      {/* Keyboard Shortcuts Cheatsheet Modal */}
      {isShortcutsOpen && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn"
          role="dialog"
          aria-modal="true"
          aria-labelledby="shortcuts-title"
        >
          <div className="absolute inset-0" onClick={() => setIsShortcutsOpen(false)} aria-hidden="true" />
          <div className="relative bg-[#181a20] rounded-3xl max-w-sm w-full p-6 shadow-2xl border border-[#262933] z-10 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#262933]">
              <div className="flex items-center space-x-2">
                <Command className="w-4 h-4 text-blue-400" />
                <h3 id="shortcuts-title" className="font-bold text-white text-sm">
                  Keyboard Shortcuts
                </h3>
              </div>
              <button
                onClick={() => setIsShortcutsOpen(false)}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-[#262933] transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Close shortcuts modal"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2.5 text-xs">
              {[
                { key: '1 or Alt+1', desc: 'Navigate to Translation Studio' },
                { key: '2 or Alt+2', desc: 'Navigate to ISL Vision' },
                { key: '3 or Alt+3', desc: 'Navigate to Text Translator' },
                { key: '4 or Alt+4', desc: 'Navigate to Voice Translator' },
                { key: '5 or Alt+5', desc: 'Navigate to History & Logs' },
                { key: '6 or Alt+6', desc: 'Navigate to Model Metrics' },
                { key: '7 or Alt+7', desc: 'Navigate to Dashboard' },
                { key: 'Alt+, or Cmd+,', desc: 'Open Platform Settings' },
                { key: '?', desc: 'Toggle this Shortcuts Guide' },
                { key: 'Escape', desc: 'Close any active modal dialog' },
              ].map((sc, i) => (
                <div key={i} className="flex items-center justify-between py-1">
                  <span className="text-zinc-300">{sc.desc}</span>
                  <kbd className="px-2 py-0.5 rounded-md bg-[#262933] border border-[#373c49] font-mono text-[11px] font-semibold text-zinc-200">
                    {sc.key}
                  </kbd>
                </div>
              ))}
            </div>

            <div className="pt-2 border-t border-[#262933] text-center text-[10px] text-zinc-500">
              IndTranslate Multimodal Accessibility Stack
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </ErrorBoundary>
  );
};

export default App;
