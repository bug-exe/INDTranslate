import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Globe, ChevronDown, Search, Check, Sparkles, X } from 'lucide-react';
import { Language } from '../types';

export interface LanguageSelectorProps {
  languages: Language[];
  value: string;
  onChange: (value: string) => void;
  includeAuto?: boolean;
  autoLabel?: string;
  label?: string;
  id?: string;
  disabled?: boolean;
  className?: string;
  placeholder?: string;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  languages = [],
  value,
  onChange,
  includeAuto = false,
  autoLabel = '⚡ Auto Detect Script',
  label,
  id,
  disabled = false,
  className = '',
  placeholder = 'Select language'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFamily, setSelectedFamily] = useState<string>('all');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const selectId = id || (label ? `lang-select-${label.toLowerCase().replace(/\s+/g, '-')}` : undefined);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  // Extract distinct families
  const families = useMemo(() => {
    const set = new Set<string>();
    languages.forEach((l) => {
      if (l.family) set.add(l.family);
    });
    return Array.from(set);
  }, [languages]);

  // Filtered languages based on search and family
  const filteredLanguages = useMemo(() => {
    return languages.filter((l) => {
      const matchesFamily = selectedFamily === 'all' || l.family === selectedFamily;
      if (!matchesFamily) return false;

      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase().trim();
      return (
        l.name.toLowerCase().includes(q) ||
        (l.native_name && l.native_name.toLowerCase().includes(q)) ||
        (l.script && l.script.toLowerCase().includes(q)) ||
        (l.region && l.region.toLowerCase().includes(q)) ||
        l.code.toLowerCase().includes(q)
      );
    });
  }, [languages, searchQuery, selectedFamily]);

  const selectedLang = languages.find((l) => l.code === value);
  const isAuto = value === 'auto';

  const handleSelect = (code: string) => {
    onChange(code);
    setIsOpen(false);
    setSearchQuery('');
  };

  return (
    <div ref={dropdownRef} className={`relative inline-flex flex-col text-left ${className}`}>
      {label && (
        <label htmlFor={selectId} className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-1">
          {label}
        </label>
      )}

      {/* Trigger Button */}
      <div className="relative inline-flex items-center">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setIsOpen(!isOpen)}
          aria-haspopup="listbox"
          aria-expanded={isOpen}
          aria-label={label || placeholder}
          className={`flex items-center justify-between space-x-2 text-xs sm:text-sm font-bold text-slate-800 
            bg-slate-100/95 hover:bg-slate-200/80 active:bg-slate-200 focus:bg-white border border-slate-200/90 
            focus:border-chakra-600 rounded-xl px-3 py-2 outline-none cursor-pointer transition-all 
            disabled:opacity-50 disabled:cursor-not-allowed shadow-subtle min-w-[140px]`}
        >
          <div className="flex items-center space-x-2 truncate">
            {isAuto ? (
              <span className="flex items-center space-x-1.5 text-chakra-700">
                <Sparkles className="w-3.5 h-3.5 text-saffron-500 animate-pulse" />
                <span className="font-bold">{autoLabel}</span>
              </span>
            ) : selectedLang ? (
              <div className="flex items-center space-x-2 truncate">
                <span className="truncate">{selectedLang.name}</span>
                {selectedLang.native_name && (
                  <span className="font-indic text-slate-500 text-[11px] font-normal truncate">
                    ({selectedLang.native_name})
                  </span>
                )}
                {selectedLang.direct_indic_trans && (
                  <span className="hidden sm:inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" title="Direct Indic Translation" />
                )}
              </div>
            ) : (
              <span className="text-slate-400 font-normal">{placeholder}</span>
            )}
          </div>
          <ChevronDown
            className={`w-3.5 h-3.5 text-slate-500 transition-transform duration-200 flex-shrink-0 ${isOpen ? 'rotate-180 text-chakra-700' : ''
              }`}
          />
        </button>

        {/* Hidden Fallback Native Select for Test Automation & Form Semantics */}
        <select
          id={selectId}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          aria-hidden="true"
          tabIndex={-1}
          className="sr-only"
        >
          {includeAuto && (
            <option value="auto">
              {autoLabel}
            </option>
          )}
          {languages.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} {lang.native_name ? `(${lang.native_name})` : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Rich Dropdown Panel */}
      {isOpen && (
        <div
          role="listbox"
          className="absolute z-50 top-full left-0 mt-1.5 w-72 sm:w-80 md:w-96 max-h-[420px] bg-white rounded-2xl 
            border border-slate-200/90 shadow-2xl overflow-hidden flex flex-col animate-scaleIn backdrop-blur-xl"
        >
          {/* Header & Search */}
          <div className="p-3 bg-slate-50/90 border-b border-slate-100 space-y-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search languages, scripts, regions..."
                className="w-full pl-8 pr-7 py-1.5 text-xs rounded-xl bg-white border border-slate-200/90 
                  focus:border-chakra-600 focus:ring-2 focus:ring-chakra-500/20 outline-none transition-all placeholder:text-slate-400 font-medium"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>

            {/* Language Family Chips */}
            {families.length > 0 && (
              <div className="flex items-center space-x-1 overflow-x-auto pb-0.5 scrollbar-none">
                <button
                  type="button"
                  onClick={() => setSelectedFamily('all')}
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full transition-colors whitespace-nowrap ${selectedFamily === 'all'
                      ? 'bg-chakra-700 text-white shadow-xs'
                      : 'bg-slate-200/70 text-slate-600 hover:bg-slate-200'
                    }`}
                >
                  All ({languages.length})
                </button>
                {families.map((fam) => (
                  <button
                    key={fam}
                    type="button"
                    onClick={() => setSelectedFamily(fam)}
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full transition-colors whitespace-nowrap ${selectedFamily === fam
                        ? 'bg-chakra-700 text-white shadow-xs'
                        : 'bg-slate-200/70 text-slate-600 hover:bg-slate-200'
                      }`}
                  >
                    {fam}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* List Options */}
          <div className="overflow-y-auto max-h-[300px] p-1.5 space-y-1 divide-y divide-slate-50">
            {/* Auto Detect Option */}
            {includeAuto && (
              <button
                type="button"
                role="option"
                aria-selected={isAuto}
                onClick={() => handleSelect('auto')}
                className={`w-full flex items-center justify-between p-2.5 rounded-xl text-left transition-all ${isAuto
                    ? 'bg-chakra-50/90 text-chakra-800 border border-chakra-200/70'
                    : 'hover:bg-slate-100/80 text-slate-800'
                  }`}
              >
                <div className="flex items-center space-x-2.5">
                  <div className="w-7 h-7 rounded-lg bg-saffron-100 text-saffron-700 flex items-center justify-center flex-shrink-0">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-bold text-slate-900">{autoLabel}</div>
                    <div className="text-[10px] text-slate-500">Universal script & dialect identifier</div>
                  </div>
                </div>
                {isAuto && <Check className="w-4 h-4 text-chakra-700 flex-shrink-0" />}
              </button>
            )}

            {filteredLanguages.map((lang) => {
              const isSelected = lang.code === value;
              return (
                <button
                  key={lang.code}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => handleSelect(lang.code)}
                  className={`w-full flex items-center justify-between p-2 rounded-xl text-left transition-all ${isSelected
                      ? 'bg-chakra-50/90 text-chakra-900 border border-chakra-200/70 font-semibold'
                      : 'hover:bg-slate-100/80 text-slate-700'
                    }`}
                >
                  <div className="flex flex-col flex-1 min-w-0 pr-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-bold text-slate-900 truncate">
                        {lang.name}
                      </span>
                      {lang.native_name && (
                        <span className="font-indic text-xs text-slate-500" dir={lang.direction || 'ltr'}>
                          {lang.native_name}
                        </span>
                      )}
                    </div>

                    {/* Metadata Badges & Region */}
                    <div className="flex flex-wrap items-center gap-1.5 mt-1">
                      {lang.script && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-medium border border-slate-200/50">
                          {lang.script}
                        </span>
                      )}
                      {lang.family && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-500 font-medium">
                          {lang.family}
                        </span>
                      )}
                      {lang.direct_indic_trans && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-50 text-emerald-700 font-bold border border-emerald-200/60">
                          ⚡ Direct Indic
                        </span>
                      )}
                      {lang.direction === 'rtl' && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-50 text-amber-700 font-bold">
                          RTL
                        </span>
                      )}
                    </div>
                  </div>

                  {isSelected && <Check className="w-4 h-4 text-chakra-700 flex-shrink-0" />}
                </button>
              );
            })}

            {filteredLanguages.length === 0 && (
              <div className="py-6 text-center text-xs text-slate-400">
                No languages match "{searchQuery}"
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default LanguageSelector;
