import React, { useState, useEffect, useMemo } from 'react';
import { 
  Search, 
  Trash2, 
  Download, 
  Star, 
  Volume2, 
  Copy, 
  Check, 
  Filter, 
  Languages, 
  Hand, 
  Mic, 
  FileSpreadsheet, 
  FileText, 
  LayoutGrid, 
  Table, 
  Clock, 
  CheckCircle2, 
  Sparkles, 
  RefreshCw, 
  ArrowRight, 
  ChevronRight, 
  ChevronDown, 
  AlertCircle, 
  RotateCcw, 
  Zap, 
  Cpu, 
  Info, 
  Calendar, 
  X,
  FileCode,
  SlidersHorizontal
} from 'lucide-react';
import { HistoryItem } from '../types';
import { 
  fetchHistory, 
  toggleHistoryFavorite, 
  deleteHistoryItem, 
  clearAllHistory, 
  playSpeechAudio,
  translateText 
} from '../services/api';
import { useToast } from '../context/ToastContext';

export const TranslationHistory: React.FC = () => {
  const { showSuccess, showError, showInfo } = useToast();

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedModule, setSelectedModule] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [favoritesOnly, setFavoritesOnly] = useState<boolean>(false);
  const [selectedSourceLang, setSelectedSourceLang] = useState<string>('all');
  const [selectedTargetLang, setSelectedTargetLang] = useState<string>('all');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest' | 'confidence'>('newest');
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');

  // Expanded items for Pipeline Details Dropdown
  const [expandedItemIds, setExpandedItemIds] = useState<Set<number>>(new Set());

  // Interactive Action States
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [retranslatingId, setRetranslatingId] = useState<number | null>(null);
  const [confirmClearOpen, setConfirmClearOpen] = useState<boolean>(false);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const items = await fetchHistory(
        selectedModule === 'all' ? undefined : selectedModule,
        searchQuery.trim() || undefined,
        favoritesOnly
      );
      setHistoryItems(items);
    } catch (e: any) {
      console.error('History load error:', e);
      showError('Failed to load translation history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [selectedModule, searchQuery, favoritesOnly]);

  // Toggle Pipeline Details Dropdown
  const toggleDetails = (id: number) => {
    setExpandedItemIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Filtered and Sorted Items
  const filteredAndSortedItems = useMemo(() => {
    let items = historyItems.filter((item) => {
      if (selectedSourceLang !== 'all' && item.source_language.toLowerCase() !== selectedSourceLang.toLowerCase()) {
        return false;
      }
      if (selectedTargetLang !== 'all' && item.target_language.toLowerCase() !== selectedTargetLang.toLowerCase()) {
        return false;
      }
      return true;
    });

    // Sorting
    return items.sort((a, b) => {
      if (sortOrder === 'oldest') {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      } else if (sortOrder === 'confidence') {
        return (b.confidence || 0.98) - (a.confidence || 0.98);
      } else {
        // newest
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
    });
  }, [historyItems, selectedSourceLang, selectedTargetLang, sortOrder]);

  // Unique languages for dropdowns
  const availableSourceLangs = useMemo(() => {
    const set = new Set<string>();
    historyItems.forEach((i) => set.add(i.source_language.toUpperCase()));
    return Array.from(set).sort();
  }, [historyItems]);

  const availableTargetLangs = useMemo(() => {
    const set = new Set<string>();
    historyItems.forEach((i) => set.add(i.target_language.toUpperCase()));
    return Array.from(set).sort();
  }, [historyItems]);

  // High-Level Metrics
  const stats = useMemo(() => {
    const total = historyItems.length;
    const textCount = historyItems.filter((i) => i.module === 'text').length;
    const voiceCount = historyItems.filter((i) => i.module === 'voice').length;
    const islCount = historyItems.filter((i) => i.module === 'isl').length;
    const favCount = historyItems.filter((i) => i.is_favorite).length;
    const avgConfidence = total > 0
      ? Math.round(
          (historyItems.reduce((acc, curr) => acc + (curr.confidence || 0.98), 0) / total) * 100
        )
      : 98;
    return { total, textCount, voiceCount, islCount, favCount, avgConfidence };
  }, [historyItems]);

  // Relative Time Formatter
  const formatRelativeTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffSecs = Math.floor(diffMs / 1000);
      const diffMins = Math.floor(diffSecs / 60);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      if (diffSecs < 60) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays === 1) return 'Yesterday';
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
      return isoString;
    }
  };

  const handleToggleFavorite = async (id: number) => {
    try {
      const newStatus = await toggleHistoryFavorite(id);
      setHistoryItems((prev) =>
        prev.map((item) => (item.id === id ? { ...item, is_favorite: newStatus } : item))
      );
      showInfo(newStatus ? 'Starred translation entry' : 'Removed from favorites');
    } catch (e) {
      showError('Failed to update favorite status');
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await deleteHistoryItem(id);
      setHistoryItems((prev) => prev.filter((item) => item.id !== id));
      showSuccess('Translation entry deleted');
    } catch (e) {
      showError('Failed to delete item');
    }
  };

  const handleClearAll = async () => {
    try {
      await clearAllHistory();
      setHistoryItems([]);
      setConfirmClearOpen(false);
      showSuccess('All translation history cleared');
    } catch (e) {
      showError('Failed to clear history');
    }
  };

  const handleCopy = (id: number, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    showSuccess('Copied translation to clipboard');
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handlePlay = async (id: number, text: string, lang: string) => {
    setPlayingId(id);
    try {
      await playSpeechAudio(text, lang);
    } catch (e) {
      showError('Speech playback unavailable');
    } finally {
      setPlayingId(null);
    }
  };

  // One-Click Re-translate Action
  const handleRetranslate = async (item: HistoryItem) => {
    setRetranslatingId(item.id);
    try {
      const res = await translateText(
        item.source_language,
        item.target_language,
        item.source_text,
        item.module
      );
      // Update local state with the newly retranslated text and updated confidence
      setHistoryItems((prev) =>
        prev.map((i) =>
          i.id === item.id
            ? {
                ...i,
                translated_text: res.translated_text,
                confidence: res.confidence,
                pipeline_notes: res.pipeline || i.pipeline_notes
              }
            : i
        )
      );
      showSuccess(`Re-translated with fresh neural model: "${res.translated_text}"`);
    } catch (e: any) {
      showError(`Re-translation failed: ${e.message || 'Model error'}`);
    } finally {
      setRetranslatingId(null);
    }
  };

  // Export 1: JSON
  const handleExportJSON = () => {
    if (filteredAndSortedItems.length === 0) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(filteredAndSortedItems, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `indtranslate_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showSuccess('Exported history as JSON');
  };

  // Export 2: CSV
  const handleExportCSV = () => {
    if (filteredAndSortedItems.length === 0) return;
    const headers = ['ID', 'Date', 'Module', 'Source Lang', 'Target Lang', 'Source Text', 'Translated Text', 'Confidence', 'Pipeline'];
    const rows = filteredAndSortedItems.map((item) => [
      item.id,
      `"${item.created_at}"`,
      `"${item.module}"`,
      `"${item.source_language}"`,
      `"${item.target_language}"`,
      `"${(item.source_text || '').replace(/"/g, '""')}"`,
      `"${(item.translated_text || '').replace(/"/g, '""')}"`,
      item.confidence || 0.98,
      `"${(item.pipeline_notes || '').replace(/"/g, '""')}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', encodeURI(csvContent));
    downloadAnchor.setAttribute('download', `indtranslate_history_${Date.now()}.csv`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showSuccess('Exported history as CSV');
  };

  // Export 3: Text (.txt)
  const handleExportText = () => {
    if (filteredAndSortedItems.length === 0) return;
    const dateStr = new Date().toLocaleString();
    let txtContent = `================================================================================\n`;
    txtContent += `INDTRANSLATE - BHARAT MULTILINGUAL TRANSLATION ACTIVITY LOG\n`;
    txtContent += `Generated: ${dateStr}\n`;
    txtContent += `Total Records: ${filteredAndSortedItems.length}\n`;
    txtContent += `Filter: Module=${selectedModule.toUpperCase()} | Search="${searchQuery}"\n`;
    txtContent += `================================================================================\n\n`;

    filteredAndSortedItems.forEach((item, index) => {
      txtContent += `[Record #${index + 1} | ID: ${item.id}]\n`;
      txtContent += `Timestamp:   ${item.created_at}\n`;
      txtContent += `Module:      ${item.module.toUpperCase()}\n`;
      txtContent += `Direction:   ${item.source_language.toUpperCase()} -> ${item.target_language.toUpperCase()}\n`;
      txtContent += `Source Text: "${item.source_text}"\n`;
      txtContent += `Translation: "${item.translated_text}"\n`;
      txtContent += `Confidence:  ${Math.round((item.confidence || 0.98) * 100)}%\n`;
      txtContent += `Pipeline:    ${item.pipeline_notes || 'AI4Bharat Pralekha Indic Neural Pipeline'}\n`;
      txtContent += `Starred:     ${item.is_favorite ? 'Yes' : 'No'}\n`;
      txtContent += `--------------------------------------------------------------------------------\n\n`;
    });

    const dataStr = 'data:text/plain;charset=utf-8,' + encodeURIComponent(txtContent);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `indtranslate_history_${Date.now()}.txt`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showSuccess('Exported history as Text log');
  };

  const getModuleBadge = (module: string) => {
    switch (module) {
      case 'isl':
        return (
          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-purple-50 text-purple-700 border border-purple-200 shadow-xs">
            <Hand className="w-3 h-3" />
            <span>ISL Sign</span>
          </span>
        );
      case 'voice':
        return (
          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-50 text-amber-700 border border-amber-200 shadow-xs">
            <Mic className="w-3 h-3" />
            <span>Voice</span>
          </span>
        );
      case 'text':
      default:
        return (
          <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-blue-50 text-blue-700 border border-blue-200 shadow-xs">
            <Languages className="w-3 h-3" />
            <span>Text</span>
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-20">
      
      {/* Top Header & Actions Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
              Translation History & Activity Hub
            </h1>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-brand-50 text-brand-700 font-bold border border-brand-200 font-mono">
              {filteredAndSortedItems.length} records
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Search, replay, one-click re-translate, and export activity across Text, Voice, and ISL modules.
          </p>
        </div>

        {/* Global Export & Clear Actions */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={loadHistory}
            title="Refresh records"
            className="p-2 rounded-xl bg-white border border-slate-200 text-slate-600 hover:text-brand-600 hover:bg-slate-50 text-xs font-semibold shadow-sm transition-all active:scale-95 cursor-pointer"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-brand-600' : ''}`} />
          </button>

          {/* Export CSV */}
          <button
            onClick={handleExportCSV}
            disabled={filteredAndSortedItems.length === 0}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-brand-600 disabled:opacity-40 text-xs font-semibold shadow-sm transition-all cursor-pointer"
          >
            <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
            <span>CSV</span>
          </button>

          {/* Export JSON */}
          <button
            onClick={handleExportJSON}
            disabled={filteredAndSortedItems.length === 0}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-brand-600 disabled:opacity-40 text-xs font-semibold shadow-sm transition-all cursor-pointer"
          >
            <FileCode className="w-4 h-4 text-brand-600" />
            <span>JSON</span>
          </button>

          {/* Export Text */}
          <button
            onClick={handleExportText}
            disabled={filteredAndSortedItems.length === 0}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-brand-600 disabled:opacity-40 text-xs font-semibold shadow-sm transition-all cursor-pointer"
          >
            <FileText className="w-4 h-4 text-amber-600" />
            <span>Text</span>
          </button>

          {/* Clear History */}
          <button
            onClick={() => setConfirmClearOpen(true)}
            disabled={historyItems.length === 0}
            className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-red-50 border border-red-200 text-red-600 hover:bg-red-100 disabled:opacity-40 text-xs font-semibold shadow-sm transition-all cursor-pointer"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Metrics Summary Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-6 gap-3">
        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Total Entries</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.total}</div>
          </div>
          <Sparkles className="w-5 h-5 text-brand-500 opacity-60" />
        </div>

        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-600">Text Mode</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.textCount}</div>
          </div>
          <Languages className="w-5 h-5 text-blue-500 opacity-60" />
        </div>

        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600">Voice Mode</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.voiceCount}</div>
          </div>
          <Mic className="w-5 h-5 text-amber-500 opacity-60" />
        </div>

        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-purple-600">ISL Vision</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.islCount}</div>
          </div>
          <Hand className="w-5 h-5 text-purple-500 opacity-60" />
        </div>

        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-500">Starred</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.favCount}</div>
          </div>
          <Star className="w-5 h-5 text-amber-400 fill-amber-400 opacity-80" />
        </div>

        <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex items-center justify-between">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-600">Avg Conf</span>
            <div className="text-xl font-extrabold text-slate-900 mt-0.5">{stats.avgConfidence}%</div>
          </div>
          <CheckCircle2 className="w-5 h-5 text-emerald-500 opacity-60" />
        </div>
      </div>

      {/* Filter, Search & View Controls Bar */}
      <div className="glass-panel p-4 rounded-3xl border border-slate-200 flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 shadow-sm">
        
        {/* Search Input */}
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search keywords in source or translation..."
            className="w-full pl-10 pr-12 py-2 rounded-xl bg-slate-100 hover:bg-slate-200/60 focus:bg-white border border-transparent focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 outline-none text-xs font-medium transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600"
            >
              Clear
            </button>
          )}
        </div>

        {/* Module Filter Tabs */}
        <div className="flex items-center space-x-1 overflow-x-auto pb-1 lg:pb-0">
          {[
            { id: 'all', label: `All (${stats.total})` },
            { id: 'text', label: `Text (${stats.textCount})` },
            { id: 'voice', label: `Voice (${stats.voiceCount})` },
            { id: 'isl', label: `ISL (${stats.islCount})` },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedModule(tab.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold whitespace-nowrap transition-all cursor-pointer ${
                selectedModule === tab.id
                  ? 'bg-brand-600 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}

          {/* Starred Only Toggle */}
          <button
            onClick={() => setFavoritesOnly(!favoritesOnly)}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-bold transition-all cursor-pointer ${
              favoritesOnly
                ? 'bg-amber-50 border-amber-300 text-amber-700 shadow-sm'
                : 'bg-slate-100 border-slate-200 text-slate-600 hover:bg-slate-200'
            }`}
          >
            <Star className={`w-3.5 h-3.5 ${favoritesOnly ? 'fill-amber-500 text-amber-500' : ''}`} />
            <span>Starred ({stats.favCount})</span>
          </button>
        </div>

        {/* Language Filter Selectors, Sorting & View Toggle */}
        <div className="flex items-center space-x-2 pt-2 lg:pt-0 border-t lg:border-t-0 border-slate-200">
          {/* Source Lang Filter */}
          <select
            value={selectedSourceLang}
            onChange={(e) => setSelectedSourceLang(e.target.value)}
            className="text-xs font-semibold bg-slate-100 text-slate-700 rounded-xl px-2.5 py-1.5 border border-slate-200 outline-none cursor-pointer"
            title="Filter by Source Language"
          >
            <option value="all">Src: All</option>
            {availableSourceLangs.map((lang) => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>

          {/* Target Lang Filter */}
          <select
            value={selectedTargetLang}
            onChange={(e) => setSelectedTargetLang(e.target.value)}
            className="text-xs font-semibold bg-slate-100 text-slate-700 rounded-xl px-2.5 py-1.5 border border-slate-200 outline-none cursor-pointer"
            title="Filter by Target Language"
          >
            <option value="all">Tgt: All</option>
            {availableTargetLangs.map((lang) => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>

          {/* Sort Order Selector */}
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as any)}
            className="text-xs font-semibold bg-slate-100 text-slate-700 rounded-xl px-2.5 py-1.5 border border-slate-200 outline-none cursor-pointer"
            title="Sort Order"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="confidence">Top Confidence</option>
          </select>

          {/* View Mode Toggle: Grid Cards vs Compact Table */}
          <div className="flex items-center bg-slate-100 p-0.5 rounded-xl border border-slate-200">
            <button
              onClick={() => setViewMode('cards')}
              title="Card View"
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${viewMode === 'cards' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
            >
              <LayoutGrid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('table')}
              title="Table View"
              className={`p-1.5 rounded-lg transition-all cursor-pointer ${viewMode === 'table' ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-400 hover:text-slate-600'}`}
            >
              <Table className="w-4 h-4" />
            </button>
          </div>
        </div>

      </div>

      {/* Main Records Display */}
      {filteredAndSortedItems.length === 0 ? (
        <div className="glass-panel p-16 text-center rounded-3xl border border-slate-200 space-y-3">
          <div className="w-16 h-16 mx-auto rounded-3xl bg-slate-100 flex items-center justify-center text-slate-400">
            <Sparkles className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-slate-800">No translation records match your filter</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            {searchQuery || selectedModule !== 'all' || favoritesOnly || selectedSourceLang !== 'all' || selectedTargetLang !== 'all'
              ? 'Try resetting your search query or filters to view all recorded translations.'
              : 'Translate spoken voice phrases, sign gestures in ISL Vision studio, or text to start building your personal translation archive.'}
          </p>
          {(searchQuery || selectedModule !== 'all' || favoritesOnly || selectedSourceLang !== 'all' || selectedTargetLang !== 'all') && (
            <button
              onClick={() => {
                setSearchQuery('');
                setSelectedModule('all');
                setFavoritesOnly(false);
                setSelectedSourceLang('all');
                setSelectedTargetLang('all');
              }}
              className="mt-2 inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-brand-50 text-brand-600 hover:bg-brand-100 text-xs font-bold transition-colors cursor-pointer"
            >
              <span>Reset All Filters</span>
            </button>
          )}
        </div>
      ) : viewMode === 'cards' ? (
        /* CARD VIEW */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredAndSortedItems.map((item) => {
            const isExpanded = expandedItemIds.has(item.id);
            return (
              <div
                key={item.id}
                className="glass-panel p-5 rounded-3xl border border-slate-200/90 hover:border-brand-300 hover:shadow-lg transition-all duration-200 flex flex-col justify-between group space-y-4"
              >
                {/* Card Header: Module Badge, Language Pair, Time & Star */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {getModuleBadge(item.module)}
                    <div className="flex items-center space-x-1.5 text-xs font-extrabold text-slate-800">
                      <span className="px-2 py-0.5 rounded-lg bg-slate-100 border border-slate-200 font-mono">
                        {item.source_language.toUpperCase()}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-slate-400" />
                      <span className="px-2 py-0.5 rounded-lg bg-brand-50 text-brand-700 border border-brand-200 font-mono">
                        {item.target_language.toUpperCase()}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1">
                    {/* Formatted Timestamp */}
                    <span 
                      className="text-[11px] text-slate-400 font-medium mr-1"
                      title={new Date(item.created_at).toLocaleString()}
                    >
                      {formatRelativeTime(item.created_at)}
                    </span>

                    {/* Star Toggle */}
                    <button
                      onClick={() => handleToggleFavorite(item.id)}
                      title={item.is_favorite ? 'Remove Favorite' : 'Mark Favorite'}
                      className="p-1.5 rounded-xl hover:bg-amber-50 text-slate-300 hover:text-amber-500 transition-colors cursor-pointer"
                    >
                      <Star className={`w-4 h-4 ${item.is_favorite ? 'fill-amber-400 text-amber-400' : ''}`} />
                    </button>

                    {/* Delete Item */}
                    <button
                      onClick={() => handleDelete(item.id)}
                      title="Delete record"
                      className="p-1.5 rounded-xl hover:bg-red-50 text-slate-300 hover:text-red-500 transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Source & Translated Text */}
                <div className="space-y-2.5">
                  <div className="p-3 rounded-2xl bg-slate-50/80 border border-slate-100 text-slate-600 text-xs font-medium leading-relaxed">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">Original</span>
                    "{item.source_text}"
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white border border-brand-100/80 text-slate-900 text-sm font-bold leading-relaxed shadow-xs">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-brand-600 block mb-0.5">Translation</span>
                    {item.translated_text}
                  </div>
                </div>

                {/* Pipeline Details Expandable Dropdown */}
                {isExpanded && (
                  <div className="p-3.5 rounded-2xl bg-slate-900 text-white space-y-2.5 animate-fadeIn text-xs">
                    <div className="flex items-center justify-between border-b border-white/10 pb-2">
                      <div className="flex items-center space-x-1.5 text-brand-300 font-bold">
                        <Cpu className="w-3.5 h-3.5" />
                        <span>Pipeline Diagnostics</span>
                      </div>
                      <span className="font-mono text-[10px] text-slate-400">Record ID: #{item.id}</span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div>
                        <span className="text-slate-400 block text-[10px]">Confidence Score:</span>
                        <span className="font-mono font-bold text-emerald-400">
                          {Math.round((item.confidence || 0.98) * 100)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px]">Length Delta:</span>
                        <span className="font-mono text-slate-200">
                          {item.source_text.length} &rarr; {item.translated_text.length} chars
                        </span>
                      </div>
                      <div className="col-span-2">
                        <span className="text-slate-400 block text-[10px]">Inference Pipeline:</span>
                        <span className="text-slate-200 font-medium">
                          {item.pipeline_notes || 'AI4Bharat Pralekha Indic Neural Model'}
                        </span>
                      </div>
                      <div className="col-span-2 text-[10px] text-slate-400">
                        Recorded at {new Date(item.created_at).toLocaleString()}
                      </div>
                    </div>
                  </div>
                )}

                {/* Card Footer: Pipeline Toggle & Quick Actions */}
                <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                  {/* Pipeline Details Toggle Button */}
                  <button
                    onClick={() => toggleDetails(item.id)}
                    className="flex items-center space-x-1 text-[11px] text-slate-500 hover:text-slate-800 font-medium cursor-pointer"
                  >
                    <span>Pipeline Details</span>
                    {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                  </button>

                  <div className="flex items-center space-x-1.5">
                    {/* One-Click Re-translate Action */}
                    <button
                      onClick={() => handleRetranslate(item)}
                      disabled={retranslatingId === item.id}
                      title="One-click re-translate with fresh model inference"
                      className="flex items-center space-x-1 px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold transition-colors disabled:opacity-40 cursor-pointer"
                    >
                      <RotateCcw className={`w-3.5 h-3.5 ${retranslatingId === item.id ? 'animate-spin text-brand-600' : ''}`} />
                      <span className="hidden sm:inline">Re-translate</span>
                    </button>

                    {/* Copy Translation */}
                    <button
                      onClick={() => handleCopy(item.id, item.translated_text)}
                      title="Copy translation"
                      className="flex items-center space-x-1 px-2.5 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold transition-colors cursor-pointer"
                    >
                      {copiedId === item.id ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copiedId === item.id ? 'Copied' : 'Copy'}</span>
                    </button>

                    {/* Audio Playback */}
                    <button
                      onClick={() => handlePlay(item.id, item.translated_text, item.target_language)}
                      title="Play speech pronunciation"
                      className="flex items-center space-x-1 px-2.5 py-1.5 rounded-xl bg-brand-50 hover:bg-brand-100 text-brand-700 font-semibold transition-colors cursor-pointer"
                    >
                      <Volume2 className={`w-3.5 h-3.5 ${playingId === item.id ? 'animate-pulse text-brand-800' : ''}`} />
                      <span>Play</span>
                    </button>
                  </div>
                </div>

              </div>
            );
          })}
        </div>
      ) : (
        /* TABLE VIEW */
        <div className="glass-panel rounded-3xl border border-slate-200 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-100/80 text-slate-600 font-bold uppercase tracking-wider text-[10px] border-b border-slate-200">
                <tr>
                  <th className="py-3 px-4">Time</th>
                  <th className="py-3 px-4">Module</th>
                  <th className="py-3 px-4">Pair</th>
                  <th className="py-3 px-4">Source Text</th>
                  <th className="py-3 px-4">Translation</th>
                  <th className="py-3 px-4">Confidence</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredAndSortedItems.map((item) => {
                  const isExpanded = expandedItemIds.has(item.id);
                  return (
                    <React.Fragment key={item.id}>
                      <tr className="hover:bg-slate-50/70 transition-colors">
                        <td 
                          className="py-3.5 px-4 text-slate-500 whitespace-nowrap cursor-pointer"
                          title={new Date(item.created_at).toLocaleString()}
                          onClick={() => toggleDetails(item.id)}
                        >
                          <div className="flex items-center space-x-1">
                            <span>{formatRelativeTime(item.created_at)}</span>
                            {isExpanded ? <ChevronDown className="w-3 h-3 text-slate-400" /> : <ChevronRight className="w-3 h-3 text-slate-400" />}
                          </div>
                        </td>
                        <td className="py-3.5 px-4 whitespace-nowrap">
                          {getModuleBadge(item.module)}
                        </td>
                        <td className="py-3.5 px-4 whitespace-nowrap font-mono font-bold text-slate-700">
                          {item.source_language.toUpperCase()} &rarr; {item.target_language.toUpperCase()}
                        </td>
                        <td className="py-3.5 px-4 max-w-xs truncate text-slate-600 font-medium">
                          "{item.source_text}"
                        </td>
                        <td className="py-3.5 px-4 max-w-sm truncate text-slate-900 font-bold">
                          {item.translated_text}
                        </td>
                        <td className="py-3.5 px-4 whitespace-nowrap">
                          <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold text-[10px] border border-emerald-200 font-mono">
                            {Math.round((item.confidence || 0.98) * 100)}%
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right whitespace-nowrap">
                          <div className="flex items-center justify-end space-x-1">
                            {/* Star Toggle */}
                            <button
                              onClick={() => handleToggleFavorite(item.id)}
                              className="p-1.5 rounded-lg text-slate-300 hover:text-amber-500 hover:bg-slate-100 cursor-pointer"
                              title="Favorite"
                            >
                              <Star className={`w-3.5 h-3.5 ${item.is_favorite ? 'fill-amber-400 text-amber-400' : ''}`} />
                            </button>

                            {/* One-Click Re-translate */}
                            <button
                              onClick={() => handleRetranslate(item)}
                              disabled={retranslatingId === item.id}
                              className="p-1.5 rounded-lg text-slate-500 hover:text-brand-600 hover:bg-slate-100 cursor-pointer"
                              title="Re-translate"
                            >
                              <RotateCcw className={`w-3.5 h-3.5 ${retranslatingId === item.id ? 'animate-spin text-brand-600' : ''}`} />
                            </button>

                            {/* Play Audio */}
                            <button
                              onClick={() => handlePlay(item.id, item.translated_text, item.target_language)}
                              className="p-1.5 rounded-lg text-slate-500 hover:text-brand-600 hover:bg-slate-100 cursor-pointer"
                              title="Play audio"
                            >
                              <Volume2 className={`w-3.5 h-3.5 ${playingId === item.id ? 'animate-pulse text-brand-600' : ''}`} />
                            </button>

                            {/* Copy Translation */}
                            <button
                              onClick={() => handleCopy(item.id, item.translated_text)}
                              className="p-1.5 rounded-lg text-slate-500 hover:text-brand-600 hover:bg-slate-100 cursor-pointer"
                              title="Copy"
                            >
                              {copiedId === item.id ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                            </button>

                            {/* Delete Item */}
                            <button
                              onClick={() => handleDelete(item.id)}
                              className="p-1.5 rounded-lg text-slate-300 hover:text-red-500 hover:bg-red-50 cursor-pointer"
                              title="Delete"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>

                      {/* Expandable Pipeline Details Row */}
                      {isExpanded && (
                        <tr className="bg-slate-900 text-slate-200">
                          <td colSpan={7} className="p-4">
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-xs">
                              <div>
                                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block">Pipeline Notes & Engine</span>
                                <span className="text-white font-medium">{item.pipeline_notes || 'AI4Bharat Pralekha Indic Neural Model'}</span>
                              </div>
                              <div className="flex items-center space-x-4 text-[11px] font-mono text-slate-300">
                                <span>Length: {item.source_text.length} &rarr; {item.translated_text.length} chars</span>
                                <span>Record: #{item.id}</span>
                                <span>Confidence: {Math.round((item.confidence || 0.98) * 100)}%</span>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confirmation Modal for Clear All */}
      {confirmClearOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel p-6 rounded-3xl max-w-sm w-full space-y-4 shadow-2xl border border-slate-200 bg-white">
            <div className="w-12 h-12 rounded-2xl bg-red-100 text-red-600 flex items-center justify-center mx-auto">
              <AlertCircle className="w-6 h-6" />
            </div>
            <div className="text-center space-y-1">
              <h4 className="text-lg font-bold text-slate-900">Clear Entire History?</h4>
              <p className="text-xs text-slate-500">
                This action will permanently delete all {historyItems.length} saved translations across Text, Voice, and ISL modules. This cannot be undone.
              </p>
            </div>
            <div className="flex items-center space-x-2 pt-2">
              <button
                onClick={() => setConfirmClearOpen(false)}
                className="flex-1 px-4 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-100 text-xs font-bold transition-all cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={handleClearAll}
                className="flex-1 px-4 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-bold shadow-md shadow-red-500/20 transition-all cursor-pointer"
              >
                Yes, Clear All
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default TranslationHistory;