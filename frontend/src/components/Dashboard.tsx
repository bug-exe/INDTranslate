import React, { useState, useMemo } from 'react';
import { 
  Hand, 
  Languages, 
  Mic, 
  Volume2, 
  ArrowRight, 
  Search, 
  Sparkles, 
  Cpu, 
  Zap, 
  Radio, 
  ShieldCheck, 
  ChevronRight,
  ArrowLeftRight,
  CheckCircle2,
  Lock,
  Layers,
  Heart,
  ExternalLink
} from 'lucide-react';
import { Language } from '../types';
import { playSpeechAudio } from '../services/api';
import { ModelMetrics } from './ModelMetrics';

// Canonical 11 Eighth Schedule Constitutional Languages of India fallback definition
const SCHEDULED_INDIC_LANGUAGES: Language[] = [
  {
    code: 'hi',
    iso_639_3: 'hin',
    name: 'Hindi',
    native_name: 'हिन्दी',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'North & Central India',
    sample: 'नमस्ते, आप कैसे हैं?',
    sample_translation: 'Hello, how are you?',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: true
  },
  {
    code: 'bn',
    iso_639_3: 'ben',
    name: 'Bengali',
    native_name: 'বাংলা',
    script: 'Bengali',
    script_code: 'Beng',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'West Bengal, Tripura & Assam',
    sample: 'নমস্কার, আপনি কেমন আছেন?',
    sample_translation: 'Greetings, how are you?',
    direct_indic_trans: true,
    gtts_code: 'bn',
    speech_recognition_code: 'bn-IN',
    transliteration_support: true
  },
  {
    code: 'te',
    iso_639_3: 'tel',
    name: 'Telugu',
    native_name: 'తెలుగు',
    script: 'Telugu',
    script_code: 'Telu',
    direction: 'ltr',
    family: 'Dravidian',
    region: 'Andhra Pradesh & Telangana',
    sample: 'నమస్కారం, మీరు ఎలా ఉన్నారు?',
    sample_translation: 'Greetings, how do you do?',
    direct_indic_trans: true,
    gtts_code: 'te',
    speech_recognition_code: 'te-IN',
    transliteration_support: true
  },
  {
    code: 'mr',
    iso_639_3: 'mar',
    name: 'Marathi',
    native_name: 'मराठी',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Maharashtra & Goa',
    sample: 'नमस्कार, तुम्ही कसे आहात?',
    sample_translation: 'Hello, how are you doing?',
    direct_indic_trans: true,
    gtts_code: 'mr',
    speech_recognition_code: 'mr-IN',
    transliteration_support: true
  },
  {
    code: 'ta',
    iso_639_3: 'tam',
    name: 'Tamil',
    native_name: 'தமிழ்',
    script: 'Tamil',
    script_code: 'Taml',
    direction: 'ltr',
    family: 'Dravidian',
    region: 'Tamil Nadu & Puducherry',
    sample: 'வணக்கம், நீங்கள் நலமாக இருக்கிறீர்களா?',
    sample_translation: 'Greetings, are you doing well?',
    direct_indic_trans: true,
    gtts_code: 'ta',
    speech_recognition_code: 'ta-IN',
    transliteration_support: true
  },
  {
    code: 'ur',
    iso_639_3: 'urd',
    name: 'Urdu',
    native_name: 'اردو',
    script: 'Nastaliq',
    script_code: 'Arab',
    direction: 'rtl',
    family: 'Indo-Aryan',
    region: 'Pan-India, Telangana, UP, J&K',
    sample: 'آداب، آپ کا مزاج کیسا ہے؟',
    sample_translation: 'Greetings, how is your health?',
    direct_indic_trans: true,
    gtts_code: 'ur',
    speech_recognition_code: 'ur-IN',
    transliteration_support: true
  },
  {
    code: 'gu',
    iso_639_3: 'guj',
    name: 'Gujarati',
    native_name: 'ગુજરાતી',
    script: 'Gujarati',
    script_code: 'Gujr',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Gujarat, Daman & Diu',
    sample: 'નમસ્તે, તમે કેમ છો?',
    sample_translation: 'Hello, how are you?',
    direct_indic_trans: true,
    gtts_code: 'gu',
    speech_recognition_code: 'gu-IN',
    transliteration_support: true
  },
  {
    code: 'kn',
    iso_639_3: 'kan',
    name: 'Kannada',
    native_name: 'ಕನ್ನಡ',
    script: 'Kannada',
    script_code: 'Knda',
    direction: 'ltr',
    family: 'Dravidian',
    region: 'Karnataka',
    sample: 'ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?',
    sample_translation: 'Hello, how are you?',
    direct_indic_trans: true,
    gtts_code: 'kn',
    speech_recognition_code: 'kn-IN',
    transliteration_support: true
  },
  {
    code: 'ml',
    iso_639_3: 'mal',
    name: 'Malayalam',
    native_name: 'മലയാളം',
    script: 'Malayalam',
    script_code: 'Mlym',
    direction: 'ltr',
    family: 'Dravidian',
    region: 'Kerala & Lakshadweep',
    sample: 'നമസ്കാരം, സുഖമാണോ?',
    sample_translation: 'Hello, are you fine?',
    direct_indic_trans: true,
    gtts_code: 'ml',
    speech_recognition_code: 'ml-IN',
    transliteration_support: true
  },
  {
    code: 'or',
    iso_639_3: 'ori',
    name: 'Odia',
    native_name: 'ଓଡ଼ିଆ',
    script: 'Odia',
    script_code: 'Orya',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Odisha',
    sample: 'ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?',
    sample_translation: 'Hello, how are you doing?',
    direct_indic_trans: true,
    gtts_code: 'or',
    speech_recognition_code: 'or-IN',
    transliteration_support: true
  },
  {
    code: 'pa',
    iso_639_3: 'pan',
    name: 'Punjabi',
    native_name: 'ਪੰਜਾਬੀ',
    script: 'Gurmukhi',
    script_code: 'Guru',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Punjab & Delhi',
    sample: 'ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਹਾਡਾ ਕੀ ਹਾਲ ਹੈ?',
    sample_translation: 'Greetings, how are you?',
    direct_indic_trans: true,
    gtts_code: 'pa',
    speech_recognition_code: 'pa-IN',
    transliteration_support: true
  },
  {
    code: 'as',
    iso_639_3: 'asm',
    name: 'Assamese',
    native_name: 'অসমীয়া',
    script: 'Bengali-Assamese',
    script_code: 'Beng',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Assam & Northeast',
    sample: 'নমস্কাৰ, আপুনি ভালে আছেনে?',
    sample_translation: 'Greetings, are you well?',
    direct_indic_trans: true,
    gtts_code: 'as',
    speech_recognition_code: 'as-IN',
    transliteration_support: true
  },
  {
    code: 'mai',
    iso_639_3: 'mai',
    name: 'Maithili',
    native_name: 'मैथिली',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Bihar & Mithila',
    sample: 'प्रणाम, अपने केहन छी?',
    sample_translation: 'Salutations, how are you?',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: true
  },
  {
    code: 'sat',
    iso_639_3: 'sat',
    name: 'Santali',
    native_name: 'ᱥᱟᱱᱛᱟᱲᱤ',
    script: 'Ol Chiki',
    script_code: 'Olck',
    direction: 'ltr',
    family: 'Austroasiatic',
    region: 'Jharkhand, Odisha & West Bengal',
    sample: 'ᱡᱚᱦᱟᱨ, ᱪᱮᱫ ᱞᱮᱠᱟ ᱢᱮᱱᱟᱜ ᱵᱤᱱᱟ?',
    sample_translation: 'Greetings, how are you?',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: false
  },
  {
    code: 'ks',
    iso_639_3: 'kas',
    name: 'Kashmiri',
    native_name: 'कॉशुर / كٲشُر',
    script: 'Nastaliq / Devanagari',
    script_code: 'Arab',
    direction: 'rtl',
    family: 'Indo-Aryan',
    region: 'Jammu & Kashmir',
    sample: 'سلام، توہہِ چھوا وارہ؟',
    sample_translation: 'Peace, are you doing well?',
    direct_indic_trans: true,
    gtts_code: 'ur',
    speech_recognition_code: 'ur-IN',
    transliteration_support: true
  },
  {
    code: 'ne',
    iso_639_3: 'nep',
    name: 'Nepali',
    native_name: 'नेपाली',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Sikkim & North Bengal',
    sample: 'नमस्ते, तपाईलाई कस्तो छ?',
    sample_translation: 'Hello, how are you?',
    direct_indic_trans: true,
    gtts_code: 'ne',
    speech_recognition_code: 'ne-NP',
    transliteration_support: true
  },
  {
    code: 'kok',
    iso_639_3: 'kok',
    name: 'Konkani',
    native_name: 'कोंकणी',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Goa & Konkan Coast',
    sample: 'नमस्कार, तुमी कशे आसात?',
    sample_translation: 'Hello, how are you?',
    direct_indic_trans: true,
    gtts_code: 'mr',
    speech_recognition_code: 'mr-IN',
    transliteration_support: true
  },
  {
    code: 'sd',
    iso_639_3: 'snd',
    name: 'Sindhi',
    native_name: 'سنڌي / सिन्धी',
    script: 'Perso-Arabic / Devanagari',
    script_code: 'Arab',
    direction: 'rtl',
    family: 'Indo-Aryan',
    region: 'Pan-India & Gujarat/Maharashtra',
    sample: 'سلام، توهان ڪيئن آهيو؟',
    sample_translation: 'Greetings, how are you?',
    direct_indic_trans: true,
    gtts_code: 'sd',
    speech_recognition_code: 'sd-IN',
    transliteration_support: true
  },
  {
    code: 'doi',
    iso_639_3: 'doi',
    name: 'Dogri',
    native_name: 'डोगरी',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Jammu & Kashmir and Himachal',
    sample: 'नमस्ते, तुंदा के हाल ऐ?',
    sample_translation: 'Greetings, how is your condition?',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: true
  },
  {
    code: 'mni',
    iso_639_3: 'mni',
    name: 'Manipuri (Meitei)',
    native_name: 'মৈতৈলোন্',
    script: 'Meetei Mayek / Bengali',
    script_code: 'Beng',
    direction: 'ltr',
    family: 'Tibeto-Burman',
    region: 'Manipur & Northeast',
    sample: 'খুরুমহরি, নহাক কমদৌরি?',
    sample_translation: 'Greetings, how are you doing?',
    direct_indic_trans: true,
    gtts_code: 'bn',
    speech_recognition_code: 'bn-IN',
    transliteration_support: true
  },
  {
    code: 'brx',
    iso_639_3: 'brx',
    name: 'Bodo',
    native_name: 'बर\'',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Tibeto-Burman',
    region: 'Bodoland & Assam',
    sample: 'खुलुमबाय, नोंथाङा माबोरै दं?',
    sample_translation: 'Welcome, how are you doing?',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: true
  },
  {
    code: 'sa',
    iso_639_3: 'san',
    name: 'Sanskrit',
    native_name: 'संस्कृतम्',
    script: 'Devanagari',
    script_code: 'Deva',
    direction: 'ltr',
    family: 'Indo-Aryan',
    region: 'Pan-India Classical',
    sample: 'सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।',
    sample_translation: 'May all beings be happy, may all be free from illness.',
    direct_indic_trans: true,
    gtts_code: 'hi',
    speech_recognition_code: 'hi-IN',
    transliteration_support: true
  }
];

interface DashboardProps {
  languages: Language[];
  setActiveTab: (tab: string) => void;
  onQuickTranslate: (text: string, targetLang: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ 
  languages, 
  setActiveTab, 
  onQuickTranslate 
}) => {
  const [selectedFamily, setSelectedFamily] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [playingAudioLang, setPlayingAudioLang] = useState<string | null>(null);

  // Quick translation bridge state
  const [quickInputText, setQuickInputText] = useState<string>('');
  const [quickSourceLang, setQuickSourceLang] = useState<string>('auto');
  const [quickTargetLang, setQuickTargetLang] = useState<string>('hi');

  const families = ['All', 'Indo-Aryan', 'Dravidian', 'Tibeto-Burman', 'Austroasiatic'];

  // Combine provided languages or fallback with canonical 11 scheduled Indian languages (+ English)
  const combinedLanguages = useMemo(() => {
    const supportedCodes = new Set(['hi', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te', 'ur', 'en']);
    const sourceList = (languages && languages.length >= 5) ? languages : SCHEDULED_INDIC_LANGUAGES;
    const filtered = sourceList.filter(l => supportedCodes.has(l.code));
    return filtered.length > 0 ? filtered : SCHEDULED_INDIC_LANGUAGES.filter(l => supportedCodes.has(l.code));
  }, [languages]);

  // High-frequency starter prompts for instant bridge testing
  const samplePrompts = [
    { label: 'नमस्ते, आप कैसे हैं?', text: 'नमस्ते, आप कैसे हैं?', source: 'hi', target: 'en' },
    { label: 'Nearest hospital (Urgent)', text: 'Where is the nearest hospital? It is an emergency.', source: 'en', target: 'hi' },
    { label: 'வணக்கம் (Tamil Greetings)', text: 'வணக்கம், நீங்கள் நலமாக இருக்கிறீர்களா?', source: 'ta', target: 'hi' },
    { label: 'ధన్యవాदాలు (Telugu Thanks)', text: 'చాలా ధన్యవాదాలు, మీ సహాయానికి కృతజ్ఞతలు.', source: 'te', target: 'bn' },
    { label: 'ISL Gesture: "Thank you"', text: 'Thank you very much', source: 'en', target: 'hi' },
  ];

  // Filter languages by family and search query
  const filteredLanguages = useMemo(() => {
    return combinedLanguages.filter((l) => {
      const matchesFamily = 
        selectedFamily === 'All' || (l.family && l.family.toLowerCase().includes(selectedFamily.toLowerCase()));

      const query = searchQuery.trim().toLowerCase();
      const matchesSearch = 
        !query ||
        l.name.toLowerCase().includes(query) ||
        l.native_name.toLowerCase().includes(query) ||
        l.code.toLowerCase().includes(query) ||
        (l.script && l.script.toLowerCase().includes(query)) ||
        (l.region && l.region.toLowerCase().includes(query));

      return matchesFamily && matchesSearch;
    });
  }, [combinedLanguages, selectedFamily, searchQuery]);

  const handlePlaySample = async (lang: Language) => {
    setPlayingAudioLang(lang.code);
    try {
      await playSpeechAudio(lang.sample, lang.code);
    } catch (e) {
      console.error('Audio playback error', e);
    } finally {
      setPlayingAudioLang(null);
    }
  };

  const handleHeroSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (quickInputText.trim()) {
      onQuickTranslate(quickInputText.trim(), quickTargetLang);
    } else {
      setActiveTab('text');
    }
  };

  const handleSwapQuickLanguages = () => {
    if (quickSourceLang === 'auto') {
      setQuickSourceLang(quickTargetLang);
      setQuickTargetLang('en');
    } else {
      const prevSource = quickSourceLang;
      setQuickSourceLang(quickTargetLang);
      setQuickTargetLang(prevSource);
    }
  };

  return (
    <div className="space-y-12 pb-16">
      
      {/* =========================================================================
          HERO SECTION: High-Impact Indian Multilingual Hub
          ========================================================================= */}
      <section className="relative overflow-hidden rounded-3xl bg-slate-900 dark:bg-slate-950 text-white p-6 sm:p-10 lg:p-12 shadow-2xl border border-slate-800">
        
        {/* Subtle Sovereign Tiranga Radial Ambient Glows */}
        <div 
          className="absolute -top-24 -left-24 w-96 h-96 rounded-full bg-saffron-500/15 blur-3xl pointer-events-none" 
          aria-hidden="true" 
        />
        <div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[36rem] h-[36rem] rounded-full bg-chakra-600/10 blur-3xl pointer-events-none" 
          aria-hidden="true" 
        />
        <div 
          className="absolute -bottom-24 -right-24 w-96 h-96 rounded-full bg-emerald-500/15 blur-3xl pointer-events-none" 
          aria-hidden="true" 
        />

        <div className="relative z-10 max-w-4xl space-y-6">
          
          {/* Mission Crest Badge */}
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-white/10 dark:bg-white/5 backdrop-blur-md border border-white/20 shadow-sm text-xs font-bold text-slate-200">
            <span className="w-2 h-2 rounded-full bg-saffron-400 animate-pulse" />
            <span className="tracking-wide">11 Scheduled Languages &bull; One Bharat &bull; ISL AI</span>
            <span className="text-[10px] uppercase font-black px-1.5 py-0.2 rounded bg-emerald-500/30 text-emerald-300 border border-emerald-400/40">
              Sovereign
            </span>
          </div>

          {/* Main Headline */}
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black tracking-tight leading-tight text-white">
            Uniting Voice, Script & Sign Across Bharat
          </h1>

          {/* Subtitle */}
          <p className="text-sm sm:text-base lg:text-lg text-slate-300 leading-relaxed max-w-3xl font-normal">
            A sovereign multimodal accessibility engine combining <strong className="text-white font-semibold">Indian Sign Language (ISL)</strong> computer vision with <strong className="text-white font-semibold">AI4Bharat direct IndicSeq2Seq neural translation</strong> across all <strong className="text-white font-semibold">11 Constitutionally Scheduled Indian Languages (+ English)</strong> without English intermediate pivot.
          </p>

          {/* Quick Translation Bridge Input Widget */}
          <form 
            onSubmit={handleHeroSubmit}
            className="mt-6 p-2.5 sm:p-3 rounded-2xl bg-white/10 dark:bg-white/5 backdrop-blur-md border border-white/20 shadow-2xl space-y-3"
          >
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
              
              {/* Text Input */}
              <div className="relative flex-1">
                <input
                  type="text"
                  value={quickInputText}
                  onChange={(e) => setQuickInputText(e.target.value)}
                  placeholder="Type or paste any Indian language phrase or English text..."
                  className="w-full px-4 py-3 rounded-xl bg-white dark:bg-slate-900 text-slate-900 dark:text-white placeholder:text-slate-400 text-sm sm:text-base font-medium focus:outline-none focus:ring-2 focus:ring-brand-400 border border-transparent"
                />
                {quickInputText && (
                  <button
                    type="button"
                    onClick={() => setQuickInputText('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 text-sm p-1"
                  >
                    &times;
                  </button>
                )}
              </div>

              {/* Language Bridge Controls */}
              <div className="flex items-center gap-1.5 flex-shrink-0">
                
                {/* Target Language Selector */}
                <select
                  value={quickTargetLang}
                  onChange={(e) => setQuickTargetLang(e.target.value)}
                  className="px-3 py-3 rounded-xl bg-slate-800 text-white border border-white/20 text-xs sm:text-sm font-bold focus:outline-none focus:ring-2 focus:ring-brand-400 cursor-pointer"
                  aria-label="Target translation language"
                >
                  <option value="hi">Hindi (हिन्दी)</option>
                  <option value="en">English</option>
                  <option value="bn">Bengali (বাংলা)</option>
                  <option value="ta">Tamil (தமிழ்)</option>
                  <option value="te">Telugu (తెలుగు)</option>
                  <option value="mr">Marathi (मराठी)</option>
                  <option value="gu">Gujarati (ગુજરાતી)</option>
                  <option value="kn">Kannada (ಕನ್ನಡ)</option>
                  <option value="ml">Malayalam (മലയാളം)</option>
                  <option value="pa">Punjabi (ਪੰਜਾਬੀ)</option>
                  <option value="ur">Urdu (اردو)</option>
                  <option value="or">Odia (ଓଡ଼ିଆ)</option>
                  <option value="as">Assamese (অসমীয়া)</option>
                  <option value="sa">Sanskrit (संस्कृतम्)</option>
                </select>

                {/* Submit Action Button */}
                <button
                  type="submit"
                  className="flex items-center justify-center space-x-1.5 px-5 sm:px-6 py-3 rounded-xl bg-gradient-to-r from-saffron-500 to-brand-600 hover:from-saffron-600 hover:to-brand-700 text-white font-bold text-xs sm:text-sm shadow-lg shadow-brand-500/20 hover:scale-[1.02] active:scale-[0.98] transition-all flex-shrink-0"
                >
                  <Zap className="w-4 h-4" />
                  <span>Translate</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>

            </div>

            {/* Quick Starter Prompt Chips */}
            <div className="flex items-center space-x-2 overflow-x-auto px-1 pt-1 text-xs">
              <span className="text-slate-400 font-semibold whitespace-nowrap">Try Instant:</span>
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setQuickInputText(p.text);
                    setQuickTargetLang(p.target);
                    onQuickTranslate(p.text, p.target);
                  }}
                  className="px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 border border-white/10 whitespace-nowrap transition-colors font-medium text-[11px]"
                >
                  {p.label}
                </button>
              ))}
            </div>

          </form>

        </div>

        {/* Live Architecture & Sovereign Telemetry Metrics */}
        <div className="mt-8 pt-6 border-t border-white/10 grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="text-2xl sm:text-3xl font-black text-saffron-400">11</div>
            <div className="text-xs text-slate-200 font-bold mt-0.5">Scheduled Languages</div>
            <div className="text-[11px] text-slate-400">Eighth Schedule &bull; Direct Indic</div>
          </div>
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="text-2xl sm:text-3xl font-black text-brand-300">52.3M</div>
            <div className="text-xs text-slate-200 font-bold mt-0.5">Model Parameters</div>
            <div className="text-[11px] text-slate-400">Locally Trained Pralekha Edge</div>
          </div>
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="text-2xl sm:text-3xl font-black text-emerald-400">543</div>
            <div className="text-xs text-slate-200 font-bold mt-0.5">Holistic Landmarks</div>
            <div className="text-[11px] text-slate-400">ISL Bi-GRU Camera AI</div>
          </div>
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="text-2xl sm:text-3xl font-black text-amber-300">100%</div>
            <div className="text-xs text-slate-200 font-bold mt-0.5">Offline Fallback</div>
            <div className="text-[11px] text-slate-400">Zero English Intermediary Pivot</div>
          </div>
        </div>

      </section>

      {/* =========================================================================
          FEATURE HIGHLIGHT CARDS: 4 Multimodal Pillars
          ========================================================================= */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white tracking-tight">
              Multimodal Translation Studios
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400">
              Four specialized interfaces engineered for total accessibility across Bharat
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* Pillar 1: ISL Vision Studio (Disconnected) */}
          <div 
            onClick={() => setActiveTab('isl')}
            className="group glass-panel dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 hover:border-slate-500 hover:shadow-xl cursor-pointer transition-all duration-200 flex flex-col justify-between opacity-80"
          >
            <div>
              <div className="w-11 h-11 rounded-2xl bg-zinc-800 text-zinc-400 flex items-center justify-center mb-4 transition-all">
                <Hand className="w-5 h-5" />
              </div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  ISL Vision AI
                </h3>
                <span className="text-[9px] uppercase font-black px-1.5 py-0.2 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700">
                  Disconnected
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                ISL neural recognition model is disconnected and unloaded from the website to conserve resources.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-zinc-400">
              <span>Model Unloaded</span>
              <ChevronRight className="w-4 h-4" />
            </div>
          </div>

          {/* Pillar 2: Neural Text Studio */}
          <div 
            onClick={() => setActiveTab('text')}
            className="group glass-panel dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 hover:border-chakra-500 hover:shadow-xl cursor-pointer transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              <div className="w-11 h-11 rounded-2xl bg-chakra-50 dark:bg-chakra-950 text-chakra-700 dark:text-chakra-300 flex items-center justify-center mb-4 group-hover:bg-chakra-700 group-hover:text-white transition-all">
                <Languages className="w-5 h-5" />
              </div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-chakra-700 dark:group-hover:text-chakra-300 transition-colors">
                  Neural Text Studio
                </h3>
                <span className="text-[9px] uppercase font-black px-1.5 py-0.2 rounded-full bg-chakra-50 dark:bg-chakra-950 text-chakra-700 dark:text-chakra-300 border border-chakra-200 dark:border-chakra-800">
                  Pralekha
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                Direct Indic Seq2Seq transformer across 11 scripts with phonetic transliteration and confidence telemetry gauges.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-chakra-700 dark:text-chakra-300">
              <span>Open Text Studio</span>
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Pillar 3: Duplex Voice Studio */}
          <div 
            onClick={() => setActiveTab('voice')}
            className="group glass-panel dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 hover:border-emerald-500 hover:shadow-xl cursor-pointer transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              <div className="w-11 h-11 rounded-2xl bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 flex items-center justify-center mb-4 group-hover:bg-emerald-700 group-hover:text-white transition-all">
                <Mic className="w-5 h-5" />
              </div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-emerald-700 dark:group-hover:text-emerald-400 transition-colors">
                  Duplex Voice Studio
                </h3>
                <span className="text-[9px] uppercase font-black px-1.5 py-0.2 rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                  Real-time
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                Bilingual conversational speaker cards with pulsating microphone visualizers, live waveform bars, and regional audio synthesis.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-emerald-700 dark:text-emerald-400">
              <span>Start Voice Studio</span>
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          {/* Pillar 4: Privacy-First Sovereign AI */}
          <div 
            onClick={() => setActiveTab('history')}
            className="group glass-panel dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 hover:border-saffron-500 hover:shadow-xl cursor-pointer transition-all duration-200 flex flex-col justify-between"
          >
            <div>
              <div className="w-11 h-11 rounded-2xl bg-saffron-50 dark:bg-saffron-950 text-saffron-700 dark:text-saffron-400 flex items-center justify-center mb-4 group-hover:bg-saffron-600 group-hover:text-white transition-all">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white group-hover:text-saffron-700 dark:group-hover:text-saffron-400 transition-colors">
                  Privacy & History
                </h3>
                <span className="text-[9px] uppercase font-black px-1.5 py-0.2 rounded-full bg-saffron-50 dark:bg-saffron-950 text-saffron-700 dark:text-saffron-300 border border-saffron-200 dark:border-saffron-800">
                  Local RAM
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mt-2 leading-relaxed">
                Zero camera retention, local memory landmark classification, starred favorites, and searchable translation audit logs.
              </p>
            </div>
            <div className="mt-5 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-saffron-700 dark:text-saffron-400">
              <span>Explore History</span>
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

        </div>
      </section>

      {/* =========================================================================
          NEURAL ENGINE BENCHMARKS & PERFORMANCE METRICS
          ========================================================================= */}
      <section className="glass-panel dark:bg-slate-900 p-6 sm:p-8 rounded-3xl border border-slate-200 dark:border-slate-800 space-y-6">
        <ModelMetrics 
          onSelectEngine={(engine) => {
            if (onQuickTranslate) {
              onQuickTranslate('', 'hi');
            } else {
              setActiveTab('text');
            }
          }}
        />
      </section>

      {/* =========================================================================
          11 SCHEDULED INDIAN LANGUAGES INTERACTIVE GRID
          ========================================================================= */}
      <section className="glass-panel dark:bg-slate-900 p-6 sm:p-8 rounded-3xl border border-slate-200 dark:border-slate-800 space-y-6">
        
        {/* Explorer Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-lg">🇮🇳</span>
              <h2 className="text-2xl font-black text-slate-900 dark:text-white tracking-tight">
                The 11 Scheduled Languages of India (+ English)
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1">
              Constitution of India &bull; Eighth Schedule &bull; Bidirectional Direct Indic & Offline Lexicon
            </p>
          </div>

          {/* Search bar inside explorer */}
          <div className="relative w-full md:w-80">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by name, script, region..."
              className="w-full pl-9 pr-8 py-2 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="p-1 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 absolute right-2.5 top-1/2 -translate-y-1/2 text-sm"
              >
                &times;
              </button>
            )}
          </div>
        </div>

        {/* Family Filter Tabs */}
        <div className="flex items-center space-x-1.5 overflow-x-auto pb-1 border-b border-slate-100 dark:border-slate-800">
          {families.map((fam) => (
            <button
              key={fam}
              type="button"
              onClick={() => setSelectedFamily(fam)}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-bold whitespace-nowrap transition-colors ${
                selectedFamily === fam
                  ? 'bg-slate-900 dark:bg-brand-600 text-white shadow-sm'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              {fam}
            </button>
          ))}
          <span className="text-xs text-slate-400 ml-auto hidden sm:inline font-medium">
            Showing {filteredLanguages.length} of {combinedLanguages.length} Scheduled Languages
          </span>
        </div>

        {/* Languages Interactive Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredLanguages.map((lang) => (
            <div
              key={lang.code}
              className="p-4 rounded-2xl bg-white dark:bg-slate-800/80 border border-slate-200/90 dark:border-slate-700/80 hover:border-brand-400 hover:shadow-lg transition-all flex flex-col justify-between group"
            >
              <div>
                {/* Language Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-bold text-slate-900 dark:text-white text-base group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">
                      {lang.name}
                    </h3>
                    <p className="text-xs text-brand-700 dark:text-brand-400 font-bold font-indic">
                      {lang.native_name}
                    </p>
                  </div>
                  <div className="flex flex-col items-end space-y-1">
                    <span className="text-[10px] uppercase font-black tracking-wider px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-600">
                      {lang.code}
                    </span>
                    <span className="text-[9px] uppercase font-bold px-1.5 py-0.2 rounded bg-saffron-50 dark:bg-saffron-950 text-saffron-800 dark:text-saffron-300 border border-saffron-200 dark:border-saffron-800">
                      {lang.script}
                    </span>
                  </div>
                </div>

                {/* Region & Family */}
                <div className="mt-2.5 text-xs text-slate-500 dark:text-slate-400 space-y-0.5">
                  <div className="truncate"><strong className="text-slate-700 dark:text-slate-300 font-semibold">Region:</strong> {lang.region}</div>
                  <div className="truncate"><strong className="text-slate-700 dark:text-slate-300 font-semibold">Family:</strong> {lang.family}</div>
                </div>

                {/* Cultural Sample Phrase Box */}
                <div className="mt-3 p-2.5 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-700/60">
                  <p className="text-xs font-semibold text-slate-800 dark:text-slate-200 line-clamp-1 font-indic">
                    {lang.sample}
                  </p>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 italic mt-0.5 line-clamp-1">
                    {lang.sample_translation}
                  </p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-700/60 flex items-center justify-between text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => handlePlaySample(lang)}
                  disabled={playingAudioLang === lang.code}
                  className="flex items-center space-x-1.5 text-brand-600 dark:text-brand-400 hover:text-brand-800 dark:hover:text-brand-300 transition-colors disabled:opacity-40"
                  aria-label={`Pronounce ${lang.name} sample phrase`}
                >
                  <Volume2 className={`w-3.5 h-3.5 ${playingAudioLang === lang.code ? 'animate-pulse text-brand-700' : ''}`} />
                  <span>{playingAudioLang === lang.code ? 'Speaking...' : 'Pronounce'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    onQuickTranslate(lang.sample, 'en');
                    setActiveTab('text');
                  }}
                  className="text-slate-600 dark:text-slate-300 hover:text-brand-600 dark:hover:text-brand-400 font-bold flex items-center space-x-1 transition-colors"
                >
                  <span>Translate</span>
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}

          {filteredLanguages.length === 0 && (
            <div className="col-span-full py-12 text-center text-slate-400 text-sm">
              No languages match your search query &quot;{searchQuery}&quot;
            </div>
          )}
        </div>

      </section>

      {/* =========================================================================
          LIVE TELEMETRY & SOVEREIGN STANDARDS
          ========================================================================= */}
      <section className="p-6 rounded-3xl bg-slate-900 dark:bg-slate-950 text-white border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
            <h3 className="font-extrabold text-sm text-white">Live System Telemetry & Sovereign Standards</h3>
          </div>
          <span className="text-[11px] text-slate-400">
            Zero English Intermediate Pivot &bull; ISLRTC 2024 Standards
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="font-bold text-slate-200">Model 3 (Pralekha)</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              52.3M Direct Indic Seq2Seq Transformer &bull; 11 Benchmark Languages Active
            </p>
          </div>

          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="font-bold text-slate-200">Offline Fallback Engine</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              100% Eighth Schedule Lexicon Loaded &bull; Zero External Network Required
            </p>
          </div>

          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-brand-400" />
              <span className="font-bold text-slate-200">MediaPipe Holistic AI</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              543 Landmarks Dual-Hand & Pose &bull; In-Memory RAM Processing
            </p>
          </div>

          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-saffron-400" />
              <span className="font-bold text-slate-200">ISLRTC Vocabulary</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Compliant with Indian Sign Language Research & Training Centre Standards
            </p>
          </div>
        </div>
      </section>

    </div>
  );
};
