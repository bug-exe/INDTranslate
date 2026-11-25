import React, { useState, useEffect, useRef } from 'react';
import { 
  Languages, 
  Hand, 
  Mic, 
  History, 
  Settings, 
  LayoutDashboard,
  Menu,
  X,
  ChevronDown,
  Activity,
  Wifi,
  Sparkles,
  ShieldCheck,
  Moon,
  Sun,
  Cpu,
  Check,
  Gauge,
  FileText,
  Download,
  Bookmark,
  HelpCircle,
  LogOut,
  Search,
  User,
  Bell
} from 'lucide-react';
import { fetchNetworkStatus } from '../services/api';
import { TranslationEngine } from '../types';

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  openSettings?: () => void;
  isMobileOpen?: boolean;
  setIsMobileOpen?: (open: boolean) => void;
  className?: string;
  onLogout?: () => void;
}

export interface TopbarProps {
  activeEngine?: TranslationEngine | string;
  onEngineChange?: (engine: TranslationEngine) => void;
  openSettings?: () => void;
  isDark?: boolean;
  onToggleTheme?: () => void;
  onOpenMobileMenu?: () => void;
  activeTab?: string;
  setActiveTab?: (tab: string) => void;
  searchQuery?: string;
  onSearchChange?: (query: string) => void;
  className?: string;
}

export interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  openSettings: () => void;
  activeEngine?: TranslationEngine | string;
  onEngineChange?: (engine: TranslationEngine) => void;
  isDark?: boolean;
  onToggleTheme?: () => void;
  searchQuery?: string;
  onSearchChange?: (query: string) => void;
  mode?: 'combined' | 'sidebar' | 'topbar';
  className?: string;
  onLogout?: () => void;
}

// Engine metadata definitions
export const ENGINE_LABELS: Record<string, { title: string; subtitle: string; tag: string }> = {
  pralekha: { title: 'Local Model (52M)', subtitle: 'Locally Trained Pralekha Edge Transformer', tag: 'Local 52M' },
  nllb: { title: 'NLLB Fine-Tuned', subtitle: 'Meta NLLB-200 LoRA Fine-Tuned (600M) [100k+]', tag: '100k+ LoRA' },
  nllb_base: { title: 'NLLB Foundation', subtitle: 'Meta NLLB-200 Base Foundation (600M)', tag: 'Meta 600M' },
  universal: { title: 'Universal AI', subtitle: 'Google Cloud + Local Multi-Tier Hybrid', tag: 'Hybrid' },
  google: { title: 'Google NMT', subtitle: 'Standard Neural Translation API', tag: 'Cloud NMT' }
};

// ============================================================================
// Ashoka Chakra Emblem SVG Component (With 24 Spokes & Tiranga Arches)
// ============================================================================

export const AshokaChakraEmblem: React.FC<{ size?: 'sm' | 'md' | 'lg' }> = ({ size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-8 h-8 rounded-xl',
    md: 'w-10 h-10 rounded-2xl',
    lg: 'w-12 h-12 rounded-2xl'
  };

  return (
    <div className={`relative ${sizeClasses[size]} bg-gradient-to-br from-[#181a20] via-chakra-900 to-[#0e1015] flex items-center justify-center p-1.5 shadow-md border border-[#262933] group-hover:border-blue-500/60 transition-all duration-300 select-none flex-shrink-0`}>
      <svg 
        viewBox="0 0 48 48" 
        fill="none" 
        xmlns="http://www.w3.org/2000/svg" 
        className="w-full h-full transform group-hover:rotate-45 transition-transform duration-700 ease-out"
        aria-hidden="true"
      >
        {/* Saffron Top Solar Arch */}
        <path 
          d="M9 17C12 9 20 6 24 6C28 6 36 9 39 17" 
          stroke="#FF671F" 
          strokeWidth="2.8" 
          strokeLinecap="round" 
        />
        {/* Emerald Bottom Terrestrial Arch */}
        <path 
          d="M9 31C12 39 20 42 24 42C28 42 36 39 39 31" 
          stroke="#046A38" 
          strokeWidth="2.8" 
          strokeLinecap="round" 
        />
        
        {/* Ashoka Chakra Outer Ring */}
        <circle cx="24" cy="24" r="10" stroke="#06038D" strokeWidth="1.8" className="text-chakra-300" />
        <circle cx="24" cy="24" r="2.8" fill="#F8FAFC" stroke="#06038D" strokeWidth="1.2" />
        
        {/* 24 Radiating Spokes */}
        {Array.from({ length: 24 }).map((_, i) => {
          const angle = (i * 360) / 24;
          const rad = (angle * Math.PI) / 180;
          const x1 = 24 + 3.2 * Math.cos(rad);
          const y1 = 24 + 3.2 * Math.sin(rad);
          const x2 = 24 + 9.5 * Math.cos(rad);
          const y2 = 24 + 9.5 * Math.sin(rad);
          return (
            <line 
              key={i} 
              x1={x1} 
              y1={y1} 
              x2={x2} 
              y2={y2} 
              stroke="#818CF8" 
              strokeWidth="0.9" 
              strokeLinecap="round" 
            />
          );
        })}
      </svg>

      {/* Tiranga Live Emerald Pulse */}
      <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-[#181a20]" />
    </div>
  );
};

// ============================================================================
// Sleek Modern Sidebar Component
// ============================================================================

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  openSettings,
  isMobileOpen = false,
  setIsMobileOpen,
  className = '',
  onLogout
}) => {
  const [logoutNotice, setLogoutNotice] = useState<string | null>(null);

  // Core navigation items matching reference design specification
  const primaryNavItems = [
    { 
      id: 'home', 
      label: 'Home', 
      icon: LayoutDashboard,
      description: 'System overview & quick actions'
    },
    { 
      id: 'text', 
      label: 'Documents', 
      icon: FileText,
      description: 'Document translation studio'
    },
    { 
      id: 'history', 
      label: 'History', 
      icon: History,
      description: 'Searchable archived multilingual logs'
    },
    { 
      id: 'downloads', 
      label: 'Downloads', 
      icon: Download,
      description: 'Exported translations & datasets'
    },
    { 
      id: 'bookmarks', 
      label: 'Bookmarks', 
      icon: Bookmark,
      description: 'Starred phrases & terminology'
    },
  ];

  // Multimodal AI Studio items
  const studioNavItems = [
    {
      id: 'isl',
      label: 'ISL Vision',
      icon: Hand,
      badge: 'Disconnected',
      description: 'ISL neural model disconnected and unloaded to conserve resources'
    },
    {
      id: 'voice',
      label: 'Voice Translator',
      icon: Mic,
      badge: 'Neural',
      description: 'Duplex Indic speech recognition & TTS'
    },
    {
      id: 'metrics',
      label: 'Model Metrics',
      icon: Gauge,
      badge: 'Bench',
      description: 'Head-to-head neural performance'
    }
  ];

  const handleItemClick = (id: string) => {
    setActiveTab(id);
    if (setIsMobileOpen) {
      setIsMobileOpen(false);
    }
  };

  const handleLogoutClick = () => {
    if (onLogout) {
      onLogout();
    } else {
      setLogoutNotice('Logged out of session');
      setTimeout(() => setLogoutNotice(null), 3000);
    }
  };

  const sidebarContent = (
    <div className="flex flex-col h-full justify-between bg-[#181a20] text-slate-300 select-none">
      {/* Top Header / Brand Logo */}
      <div className="p-5 border-b border-[#262933]">
        <div 
          onClick={() => handleItemClick('home')}
          className="flex items-center space-x-3 cursor-pointer group"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && handleItemClick('home')}
          aria-label="IndTranslate Platform Home"
        >
          <AshokaChakraEmblem size="md" />
          <div className="min-w-0">
            <div className="flex items-center space-x-2">
              <span className="text-base font-extrabold tracking-tight text-white group-hover:text-blue-400 transition-colors">
                IndTranslate
              </span>
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-black uppercase tracking-wider bg-blue-500/20 text-blue-400 border border-blue-500/30">
                Bharat AI
              </span>
            </div>
            <p className="text-[11px] text-slate-400 truncate">
              Multimodal Indian AI
            </p>
          </div>
        </div>
      </div>

      {/* Main Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
        {/* Primary Vertical Links */}
        <div>
          <p className="px-3 text-[10px] uppercase font-bold tracking-wider text-slate-300 mb-2">
            Navigation
          </p>
          <div className="space-y-1">
            {primaryNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleItemClick(item.id)}
                  className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    isActive
                      ? 'bg-[#2563eb] text-white font-semibold shadow-md shadow-blue-600/30 ring-1 ring-blue-400/50'
                      : 'text-slate-400 hover:text-white hover:bg-[#262933]/60'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-white'}`} />
                  <span className="truncate">{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Multimodal AI Studio Links */}
        <div>
          <p className="px-3 text-[10px] uppercase font-bold tracking-wider text-slate-300 mb-2">
            Multimodal Studio
          </p>
          <div className="space-y-1">
            {studioNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleItemClick(item.id)}
                  className={`w-full flex items-center justify-between px-3.5 py-2 rounded-xl text-xs font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    isActive
                      ? 'bg-[#2563eb] text-white font-semibold shadow-md shadow-blue-600/30'
                      : 'text-slate-400 hover:text-white hover:bg-[#262933]/60'
                  }`}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <div className="flex items-center space-x-3 min-w-0">
                    <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                    <span className="truncate">{item.label}</span>
                  </div>
                  {item.badge && (
                    <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wide flex-shrink-0 ${
                      isActive 
                        ? 'bg-white/20 text-white' 
                        : 'bg-[#262933] text-slate-300 border border-slate-700'
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Footer Section: Support with Notification Pill '2' & Log Out */}
      <div className="p-3 border-t border-[#262933] space-y-1">
        {/* Support Link with Notification Pill '2' */}
        <button
          onClick={() => handleItemClick('support')}
          className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            activeTab === 'support' 
              ? 'bg-[#2563eb] text-white font-semibold' 
              : 'text-slate-400 hover:text-white hover:bg-[#262933]/60'
          }`}
          aria-label="Support and Help Center"
        >
          <div className="flex items-center space-x-3">
            <HelpCircle className="w-4 h-4 flex-shrink-0 text-slate-400" />
            <span>Support</span>
          </div>
          {/* Notification pill '2' */}
          <span className="inline-flex items-center justify-center px-1.5 py-0.2 min-w-[18px] h-[18px] text-[10px] font-black bg-blue-600 text-white rounded-full shadow-sm">
            2
          </span>
        </button>

        {/* Log out link */}
        <button
          onClick={handleLogoutClick}
          className="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500"
          aria-label="Log out"
        >
          <LogOut className="w-4 h-4 flex-shrink-0" />
          <span>Log out</span>
        </button>

        {logoutNotice && (
          <div className="px-3 py-1 text-[11px] text-amber-400 text-center animate-fadeIn">
            {logoutNotice}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Fixed Sidebar */}
      <aside 
        className={`hidden lg:flex flex-col w-64 fixed inset-y-0 left-0 z-40 border-r border-[#262933] shadow-xl ${className}`}
        aria-label="Main Navigation Sidebar"
      >
        {sidebarContent}
      </aside>

      {/* Mobile Drawer Navigation Overlay */}
      {isMobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          {/* Backdrop Scrim */}
          <div 
            className="fixed inset-0 bg-black/70 backdrop-blur-sm transition-opacity"
            onClick={() => setIsMobileOpen && setIsMobileOpen(false)}
            aria-hidden="true"
          />

          {/* Drawer Slide-in Container */}
          <div className="fixed inset-y-0 left-0 max-w-xs w-full bg-[#181a20] shadow-2xl z-50 flex flex-col animate-slideIn">
            <div className="absolute top-4 right-4 z-50">
              <button
                onClick={() => setIsMobileOpen && setIsMobileOpen(false)}
                className="p-1.5 rounded-xl bg-[#262933] text-slate-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Close navigation menu"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
};

// ============================================================================
// Sleek Modern Topbar Component
// ============================================================================

export const Topbar: React.FC<TopbarProps> = ({
  activeEngine = 'pralekha',
  onEngineChange,
  openSettings,
  isDark = true,
  onToggleTheme,
  onOpenMobileMenu,
  activeTab,
  setActiveTab,
  searchQuery = '',
  onSearchChange,
  className = ''
}) => {
  const [internalSearch, setInternalSearch] = useState<string>(searchQuery);
  const [isEngineDropdownOpen, setIsEngineDropdownOpen] = useState<boolean>(false);
  const [isStatusPopoverOpen, setIsStatusPopoverOpen] = useState<boolean>(false);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState<boolean>(false);
  const [isListeningMic, setIsListeningMic] = useState<boolean>(false);
  const [networkStatus, setNetworkStatus] = useState<{ online: boolean; offlineReady: boolean }>({
    online: true,
    offlineReady: true
  });

  const popoverRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<HTMLDivElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchNetworkStatus()
      .then((data) => {
        setNetworkStatus({
          online: data?.network_online ?? true,
          offlineReady: data?.offline_ready ?? true
        });
      })
      .catch(() => {
        setNetworkStatus({ online: true, offlineReady: true });
      });

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (popoverRef.current && !popoverRef.current.contains(target)) {
        setIsStatusPopoverOpen(false);
      }
      if (engineRef.current && !engineRef.current.contains(target)) {
        setIsEngineDropdownOpen(false);
      }
      if (profileRef.current && !profileRef.current.contains(target)) {
        setIsProfileMenuOpen(false);
      }
    };

    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, []);

  const handleSearchInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInternalSearch(val);
    if (onSearchChange) {
      onSearchChange(val);
    }
  };

  const handleMicClick = () => {
    setIsListeningMic(true);
    setTimeout(() => {
      setIsListeningMic(false);
    }, 2500);
  };

  const handleSelectEngine = (engine: TranslationEngine) => {
    if (onEngineChange) {
      onEngineChange(engine);
    }
    setIsEngineDropdownOpen(false);
  };

  const currentEngineMeta = ENGINE_LABELS[activeEngine] || ENGINE_LABELS['pralekha'];

  return (
    <header 
      className={`sticky top-0 z-30 w-full bg-[#0e1015]/95 backdrop-blur-md border-b border-[#262933] transition-colors duration-200 ${className}`}
      aria-label="Top Bar"
    >
      {/* Tiranga Micro-Glow Accent Line on Top */}
      <div 
        className="w-full h-[2px] bg-gradient-to-r from-[#FF671F] via-[#ffffff] to-[#046A38] opacity-90"
        aria-hidden="true" 
      />

      <div className="w-full px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-3">
          
          {/* Left section: Mobile menu trigger & optional mobile logo */}
          <div className="flex items-center space-x-3 lg:hidden">
            <button
              onClick={onOpenMobileMenu}
              className="p-2 rounded-xl bg-[#181a20] border border-[#262933] text-slate-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label="Toggle navigation drawer"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center space-x-2">
              <AshokaChakraEmblem size="sm" />
              <span className="font-extrabold text-sm text-white tracking-tight">
                IndTranslate
              </span>
            </div>
          </div>

          {/* Centered Pill Search Bar: 'Search in docs, history ...' */}
          <div className="flex-1 max-w-xl mx-auto px-2">
            <div className="relative flex items-center w-full rounded-full bg-[#181a20] border border-[#262933] hover:border-slate-700 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 px-3.5 py-1.5 transition-all shadow-inner">
              <Search className="w-4 h-4 text-slate-400 flex-shrink-0 mr-2.5" />
              <input
                ref={searchInputRef}
                type="text"
                value={internalSearch}
                onChange={handleSearchInput}
                placeholder="Search in docs, history ..."
                className="w-full bg-transparent text-xs sm:text-sm text-slate-200 placeholder-slate-400 focus:outline-none"
                aria-label="Search documents and history"
              />
              <div className="flex items-center space-x-2 flex-shrink-0 ml-2">
                <button
                  type="button"
                  onClick={handleMicClick}
                  className={`p-1 rounded-full text-slate-400 hover:text-blue-400 transition-colors focus:outline-none ${isListeningMic ? 'text-blue-400 animate-pulse' : ''}`}
                  title="Voice Search"
                  aria-label="Voice Search"
                >
                  <Mic className="w-4 h-4" />
                </button>
                <kbd className="hidden sm:inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono text-slate-400 bg-[#262933] rounded border border-slate-700 select-none">
                  ⌘K
                </kbd>
              </div>
            </div>
          </div>

          {/* Right Header Actions: Engine Selector Pill, Health Badge, Theme, User Profile */}
          <div className="flex items-center space-x-2 sm:space-x-2.5 flex-shrink-0">
            
            {/* Active Translation Engine Selector Pill */}
            <div className="relative" ref={engineRef}>
              <button
                onClick={() => setIsEngineDropdownOpen(!isEngineDropdownOpen)}
                className="flex items-center space-x-2 px-3 py-1.5 rounded-full border border-[#262933] bg-[#181a20] hover:border-blue-500/50 text-slate-200 text-xs font-medium transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                title="Active Translation Engine"
                aria-expanded={isEngineDropdownOpen}
              >
                <Cpu className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                <span className="font-semibold text-slate-200 hidden md:inline truncate max-w-[130px]">
                  {currentEngineMeta.title}
                </span>
                <span className="text-[10px] px-1.5 py-0.2 rounded font-bold uppercase bg-blue-500/20 text-blue-400 border border-blue-500/30 hidden xl:inline">
                  {currentEngineMeta.tag}
                </span>
                <ChevronDown className={`w-3 h-3 text-slate-400 transition-transform duration-200 ${isEngineDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Engine Switcher Dropdown */}
              {isEngineDropdownOpen && (
                <div className="absolute right-0 mt-2 w-72 rounded-2xl bg-[#181a20] border border-[#262933] shadow-2xl p-2.5 z-50 animate-fadeIn">
                  <div className="px-2 py-1.5 border-b border-[#262933] flex items-center justify-between">
                    <span className="text-[10px] uppercase font-bold text-slate-300 tracking-wider">
                      Translation Engine
                    </span>
                    <span className="text-[10px] text-blue-400 font-bold">
                      Direct Indic Seq2Seq
                    </span>
                  </div>
                  <div className="py-1.5 space-y-1">
                    {(['pralekha', 'nllb', 'nllb_base', 'universal', 'google'] as TranslationEngine[]).map((engKey) => {
                      const meta = ENGINE_LABELS[engKey];
                      const isSelected = activeEngine === engKey;
                      return (
                        <button
                          key={engKey}
                          onClick={() => handleSelectEngine(engKey)}
                          className={`w-full flex items-start justify-between p-2 rounded-xl text-left transition-colors focus:outline-none focus:ring-1 focus:ring-blue-500 ${
                            isSelected
                              ? 'bg-blue-600/20 border border-blue-500/40 text-blue-300'
                              : 'hover:bg-[#262933] text-slate-200'
                          }`}
                        >
                          <div>
                            <div className="flex items-center space-x-1.5">
                              <span className={`text-xs font-semibold ${isSelected ? 'text-white' : 'text-slate-200'}`}>
                                {meta.title}
                              </span>
                              <span className="text-[9px] px-1 py-0.2 rounded bg-[#262933] text-slate-400 font-bold">
                                {meta.tag}
                              </span>
                            </div>
                            <p className="text-[11px] text-slate-400 mt-0.5 leading-tight">
                              {meta.subtitle}
                            </p>
                          </div>
                          {isSelected && <Check className="w-4 h-4 text-blue-400 mt-0.5" />}
                        </button>
                      );
                    })}
                  </div>
                  {setActiveTab && (
                    <div className="pt-1.5 border-t border-[#262933]">
                      <button
                        type="button"
                        onClick={() => {
                          setIsEngineDropdownOpen(false);
                          setActiveTab('metrics');
                        }}
                        className="w-full flex items-center justify-center space-x-1.5 py-1.5 px-2 rounded-xl text-xs font-bold text-blue-400 hover:bg-blue-500/10 transition-colors"
                      >
                        <Gauge className="w-3.5 h-3.5" />
                        <span>Compare Model Metrics & Latency</span>
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Live Backend Health Status Badge */}
            <div className="relative" ref={popoverRef}>
              <button
                onClick={() => setIsStatusPopoverOpen(!isStatusPopoverOpen)}
                className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-full border border-[#262933] bg-[#181a20] hover:bg-[#262933] text-slate-300 text-xs font-medium transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                title="System Health & Telemetry"
                aria-expanded={isStatusPopoverOpen}
              >
                <span className="relative flex h-2 w-2">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${networkStatus.online ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${networkStatus.online ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
                </span>
                <span className="text-[11px] font-bold hidden lg:inline text-slate-300">
                  {networkStatus.online ? 'Online' : 'Offline'}
                </span>
                <ChevronDown className={`w-3 h-3 text-slate-400 transition-transform duration-200 ${isStatusPopoverOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Status Telemetry Dropdown Popover */}
              {isStatusPopoverOpen && (
                <div className="absolute right-0 mt-2 w-72 sm:w-80 rounded-2xl bg-[#181a20] border border-[#262933] shadow-2xl p-4 z-50 animate-fadeIn">
                  <div className="flex items-center justify-between pb-3 border-b border-[#262933]">
                    <div className="flex items-center space-x-2">
                      <Activity className="w-4 h-4 text-blue-400" />
                      <span className="font-bold text-white text-xs uppercase tracking-wider">
                        Telemetry & Engine Health
                      </span>
                    </div>
                    <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">
                      Operational
                    </span>
                  </div>

                  <div className="py-3 space-y-2.5 text-xs">
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-semibold text-slate-200">AI4Bharat Pralekha (Model 3)</p>
                        <p className="text-[11px] text-slate-400">Direct Indic Seq2Seq (6.33M Params)</p>
                      </div>
                      <span className="text-[10px] font-bold text-blue-400 bg-blue-500/20 px-2 py-0.5 rounded">Active</span>
                    </div>

                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-semibold text-slate-200">ISL Vision Tracking</p>
                        <p className="text-[11px] text-slate-400">MediaPipe Holistic (543 Landmarks)</p>
                      </div>
                      <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/20 px-2 py-0.5 rounded">Ready</span>
                    </div>

                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-semibold text-slate-200">Indic Speech Synthesis</p>
                        <p className="text-[11px] text-slate-400">gTTS & Web Speech Fallback</p>
                      </div>
                      <span className="text-[10px] font-bold text-slate-400 bg-[#262933] px-2 py-0.5 rounded">Loaded</span>
                    </div>

                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-semibold text-slate-200">Offline Fallback Engine</p>
                        <p className="text-[11px] text-slate-400">
                          {networkStatus.online ? 'Online Connected' : 'Local Lexicon Only'} &bull; 11 Languages (+ English)
                        </p>
                      </div>
                      <Wifi className="w-3.5 h-3.5 text-emerald-400 mt-1" />
                    </div>
                  </div>

                  <div className="pt-2.5 border-t border-[#262933] flex items-center justify-between text-[11px] text-slate-400">
                    <span>11 Scheduled Indic Scripts (+ English)</span>
                    <span className="font-semibold text-slate-300">v1.0 Sovereign</span>
                  </div>
                </div>
              )}
            </div>

            {/* Theme Toggle Button */}
            {onToggleTheme && (
              <button
                onClick={onToggleTheme}
                title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
                className="p-2 rounded-full border border-[#262933] bg-[#181a20] hover:bg-[#262933] text-slate-300 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Toggle color theme"
              >
                {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
              </button>
            )}

            {/* Settings Trigger Button */}
            {openSettings && (
              <button
                onClick={openSettings}
                title="System Settings & Privacy"
                className="p-2 rounded-full border border-[#262933] bg-[#181a20] hover:bg-[#262933] text-slate-300 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Open Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
            )}

            {/* User Profile Avatar with Online Status */}
            <div className="relative" ref={profileRef}>
              <button
                onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                className="relative flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 text-white font-bold text-xs border border-[#262933] hover:ring-2 hover:ring-blue-500/50 transition-all focus:outline-none"
                aria-label="User Profile and Account Menu"
                aria-expanded={isProfileMenuOpen}
              >
                <span className="select-none">IT</span>
                <span className="absolute bottom-0 right-0 w-2 h-2 rounded-full bg-emerald-500 ring-2 ring-[#181a20]" />
              </button>

              {/* User Profile Dropdown Menu */}
              {isProfileMenuOpen && (
                <div className="absolute right-0 mt-2 w-56 rounded-2xl bg-[#181a20] border border-[#262933] shadow-2xl p-2 z-50 animate-fadeIn text-xs text-slate-300">
                  <div className="px-3 py-2 border-b border-[#262933]">
                    <p className="font-bold text-white text-sm">IndTranslate User</p>
                    <p className="text-[11px] text-slate-400">researcher@bharat.ai</p>
                  </div>
                  <div className="py-1">
                    <button
                      onClick={() => {
                        setIsProfileMenuOpen(false);
                        if (openSettings) openSettings();
                      }}
                      className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl hover:bg-[#262933] text-left transition-colors"
                    >
                      <Settings className="w-3.5 h-3.5 text-slate-400" />
                      <span>Account Preferences</span>
                    </button>
                    {setActiveTab && (
                      <button
                        onClick={() => {
                          setIsProfileMenuOpen(false);
                          setActiveTab('history');
                        }}
                        className="w-full flex items-center space-x-2.5 px-3 py-2 rounded-xl hover:bg-[#262933] text-left transition-colors"
                      >
                        <History className="w-3.5 h-3.5 text-slate-400" />
                        <span>Saved Translations</span>
                      </button>
                    )}
                  </div>
                  <div className="pt-1 border-t border-[#262933]">
                    <div className="px-3 py-1.5 text-[10px] text-slate-300">
                      IndTranslate v1.0 &bull; Sovereign Bharat
                    </div>
                  </div>
                </div>
              )}
            </div>

          </div>

        </div>
      </div>
    </header>
  );
};

// ============================================================================
// Combined Navbar Component (Composes Sidebar & Topbar for App Integration)
// ============================================================================

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  openSettings,
  activeEngine = 'pralekha',
  onEngineChange,
  isDark = true,
  onToggleTheme,
  searchQuery,
  onSearchChange,
  mode = 'combined',
  className = '',
  onLogout
}) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  if (mode === 'sidebar') {
    return (
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        openSettings={openSettings}
        isMobileOpen={isMobileMenuOpen}
        setIsMobileOpen={setIsMobileMenuOpen}
        className={className}
        onLogout={onLogout}
      />
    );
  }

  if (mode === 'topbar') {
    return (
      <Topbar
        activeEngine={activeEngine}
        onEngineChange={onEngineChange}
        openSettings={openSettings}
        isDark={isDark}
        onToggleTheme={onToggleTheme}
        onOpenMobileMenu={() => setIsMobileMenuOpen(true)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        className={className}
      />
    );
  }

  // Combined mode: renders both the modern fixed sidebar and sticky topbar
  return (
    <>
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        openSettings={openSettings}
        isMobileOpen={isMobileMenuOpen}
        setIsMobileOpen={setIsMobileMenuOpen}
        onLogout={onLogout}
      />
      <Topbar
        activeEngine={activeEngine}
        onEngineChange={onEngineChange}
        openSettings={openSettings}
        isDark={isDark}
        onToggleTheme={onToggleTheme}
        onOpenMobileMenu={() => setIsMobileMenuOpen(true)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        className="lg:pl-64"
      />
    </>
  );
};

export default Navbar;
