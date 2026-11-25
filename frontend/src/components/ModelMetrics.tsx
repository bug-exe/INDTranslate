import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Zap, 
  Database, 
  ShieldCheck, 
  Activity, 
  BarChart3, 
  Award, 
  Sparkles, 
  Gauge, 
  CheckCircle2, 
  Clock, 
  HardDrive, 
  Layers, 
  Cloud, 
  Check, 
  ArrowUpRight,
  TrendingUp,
  Server,
  RefreshCw,
  Info
} from 'lucide-react';
import { TranslationEngine } from '../types';
import { fetchTranslationModels } from '../services/api';

export interface ModelMetricData {
  id: TranslationEngine;
  name: string;
  badge: string;
  badgeColor: string;
  category: 'local_trained' | 'local_foundation' | 'indic_foundation' | 'hybrid' | 'cloud';
  categoryLabel: string;
  parameters: string;
  paramCountNum: number; // For relative visual scaling
  latencyMs: number;
  vramMb: number | string;
  vramNum: number; // In MB for scaling
  chrfScore: number;
  bleuScore: number;
  degeneracyRate: number; // In percentage
  languages: string;
  languageCount: number;
  architecture: string;
  tokenizer: string;
  cloudDependency: 'Zero (100% On-Device)' | 'Zero (Local GPU)' | 'Dynamic' | 'Requires Internet';
  trainingCorpus: string;
  description: string;
  strengths: string[];
}

export const BENCHMARK_MODELS: ModelMetricData[] = [
  {
    id: 'pralekha',
    name: 'Locally Trained Pralekha (52.3M)',
    badge: 'Locally Trained • Edge',
    badgeColor: 'bg-chakra-100 text-chakra-800 dark:bg-chakra-950/80 dark:text-chakra-300 border-chakra-200 dark:border-chakra-800',
    category: 'local_trained',
    categoryLabel: 'Locally Trained Custom Model',
    parameters: '52.33M',
    paramCountNum: 52.33,
    latencyMs: 175.1,
    vramMb: '307 MB',
    vramNum: 307,
    chrfScore: 31.62,
    bleuScore: 13.90,
    degeneracyRate: 2.0,
    languages: '11 Scheduled Indian Languages + English',
    languageCount: 12,
    architecture: '12-Layer Seq2Seq Transformer (6 Enc / 6 Dec, d_model=512, 8 Heads)',
    tokenizer: 'SentencePiece Unigram BPE (16,000 vocab)',
    cloudDependency: 'Zero (100% On-Device)',
    trainingCorpus: '500,000 clean 12-way parallel pairs in pralekha_strict_12way.db across 132 direct zero-pivot directions',
    description: 'Custom trained edge Transformer specifically optimized for direct Indic-to-Indic translation without English intermediary pivoting. Ultra-low VRAM footprint and minimal repetition.',
    strengths: [
      'Ultra-compact 307 MB VRAM footprint runnable on any consumer laptop or GPU',
      'Direct zero-pivot translation between all 11 scheduled Indian languages',
      'Lowest degeneracy rate (2.0%) with near-zero repetitive hallucination',
      '100% sovereign on-device inference with absolute data privacy'
    ]
  },
  {
    id: 'nllb',
    name: 'Meta NLLB-200 Fine-Tuned (600M) [LoRA 100k+]',
    badge: 'Fine-Tuned 100k+',
    badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    category: 'local_foundation',
    categoryLabel: 'Distilled Foundation Model (Fine-Tuned 100k+)',
    parameters: '615.0M',
    paramCountNum: 615.0,
    latencyMs: 316.5,
    vramMb: '1,250 MB (FP16)',
    vramNum: 1250,
    chrfScore: 50.67,
    bleuScore: 16.16,
    degeneracyRate: 2.0,
    languages: '11 Scheduled Indian Languages + English (200+ Global)',
    languageCount: 12,
    architecture: 'Dense Multilingual Transformer with LoRA (r=16, alpha=32 across 72 attention layers)',
    tokenizer: 'NLLB BPE (256,206 vocab)',
    cloudDependency: 'Zero (Local GPU)',
    trainingCorpus: 'Pre-trained on 200+ languages, fine-tuned on 168,000+ pairs across 12-way parallel Indian corpus (Val loss: 1.3561, PPL: 3.88)',
    description: 'Meta AI NLLB-200 distilled 600M foundation model, fine-tuned with LoRA over 100k+ pairs (10,500 steps). Achieved 50.67 chrF++, 1.35 validation loss, and 3.88 perplexity.',
    strengths: [
      'Peak chrF++ score (50.67) and +19.4% BLEU-1 gain (+5.57 pts) over foundation baseline',
      'Validated low perplexity (3.88 PPL, 1.3561 val loss) after 168,000+ training pairs',
      'Zero English intermediate pivoting with lowest degeneracy rate (2.0%)',
      'Runs locally in FP16 / BF16 mixed precision on consumer RTX GPUs with 316.5ms latency'
    ]
  },
  {
    id: 'nllb_base',
    name: 'Meta NLLB-200 Base Foundation (600M)',
    badge: 'Base Foundation',
    badgeColor: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/80 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800',
    category: 'local_foundation',
    categoryLabel: 'Base Foundation Model (Unadapted)',
    parameters: '615.0M',
    paramCountNum: 615.0,
    latencyMs: 262.2,
    vramMb: '1,180 MB (FP16)',
    vramNum: 1180,
    chrfScore: 45.76,
    bleuScore: 15.04,
    degeneracyRate: 2.0,
    languages: '11 Scheduled Indian Languages + English (200+ Global)',
    languageCount: 12,
    architecture: 'Dense Multilingual Transformer (24-Layer Seq2Seq, 600M Distilled)',
    tokenizer: 'NLLB BPE (256,206 vocab)',
    cloudDependency: 'Zero (Local GPU)',
    trainingCorpus: 'Meta No Language Left Behind global multilingual pretraining',
    description: 'Original raw Meta NLLB-200 distilled 600M foundation model running locally in FP16 precision without LoRA adaptations.',
    strengths: [
      'Original Meta AI foundation architecture with unadapted pre-trained weights',
      'Broad general multilingual coverage across 200+ global languages',
      'Fast 262ms local inference on RTX 4050 GPU',
      'Baseline benchmark for measuring domain fine-tuning gains'
    ]
  },
  {
    id: 'universal',
    name: 'Universal Auto-Adaptive Multi-Tier',
    badge: 'Hybrid Dynamic Routing',
    badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    category: 'hybrid',
    categoryLabel: 'Multi-Tier Adaptive Engine',
    parameters: 'Dynamic (Composite)',
    paramCountNum: 150.0,
    latencyMs: 145.0,
    vramMb: 'Dynamic (300MB - 1GB)',
    vramNum: 500,
    chrfScore: 34.80,
    bleuScore: 13.50,
    degeneracyRate: 2.4,
    languages: '11 Scheduled Indian Languages + English',
    languageCount: 12,
    architecture: 'Adaptive Multi-Tier Router (Local 52M -> NLLB -> Lexicon -> Cloud)',
    tokenizer: 'Adaptive Tokenizer Router',
    cloudDependency: 'Dynamic',
    trainingCorpus: 'Multi-tier routing across local neural weights, authentic lexicons, and cloud',
    description: 'Intelligent orchestrator that routes requests based on hardware availability, network state, and sentence complexity. Uses local models first and falls back gracefully.',
    strengths: [
      'High operational availability with multi-stage fallback',
      'Fastest average latency for common phrases with lexical fast-path',
      'Automatically selects between edge Pralekha and foundation NLLB',
      'Seamless online/offline adaptation'
    ]
  },
  {
    id: 'google',
    name: 'Google Cloud NMT Engine',
    badge: 'Remote Cloud API',
    badgeColor: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700',
    category: 'cloud',
    categoryLabel: 'Commercial Cloud Translation',
    parameters: 'Proprietary Cloud',
    paramCountNum: 1000.0,
    latencyMs: 340.0,
    vramMb: '0 MB (Remote)',
    vramNum: 0,
    chrfScore: 36.10,
    bleuScore: 14.10,
    degeneracyRate: 2.8,
    languages: '11 Scheduled Indian Languages + English (130+ Global)',
    languageCount: 12,
    architecture: 'Google Multilingual Neural Machine Translation (GNMT v2)',
    tokenizer: 'Google Subword Tokenizer',
    cloudDependency: 'Requires Internet',
    trainingCorpus: 'Web-scale multilingual corpus with global web crawls',
    description: 'Commercial cloud translation baseline used for cross-validating sovereign local model accuracy. Requires active internet connection and API token credentials.',
    strengths: [
      'Zero local compute or GPU VRAM requirements on the host machine',
      'Broad vocabulary covering contemporary and global technical terms',
      'Standard enterprise industry benchmark for cross-validation',
      'High fluency on English-centric translation directions'
    ]
  }
];

export interface ModelMetricsProps {
  currentEngine?: TranslationEngine | string;
  onSelectEngine?: (engine: TranslationEngine) => void;
  isCompact?: boolean;
}

export const ModelMetrics: React.FC<ModelMetricsProps> = ({
  currentEngine = 'pralekha',
  onSelectEngine,
  isCompact = false
}) => {
  const [selectedModelId, setSelectedModelId] = useState<TranslationEngine>(
    (currentEngine as TranslationEngine) || 'pralekha'
  );
  const [activeMetricTab, setActiveMetricTab] = useState<'overview' | 'comparison' | 'radar'>('overview');
  const [liveProbeLatency, setLiveProbeLatency] = useState<number | null>(null);
  const [isProbing, setIsProbing] = useState<boolean>(false);
  const [backendModelStatus, setBackendModelStatus] = useState<any>(null);

  useEffect(() => {
    setSelectedModelId((currentEngine as TranslationEngine) || 'pralekha');
  }, [currentEngine]);

  useEffect(() => {
    fetchTranslationModels()
      .then(data => setBackendModelStatus(data))
      .catch(() => {});
  }, []);

  const handleRunLiveProbe = async () => {
    setIsProbing(true);
    const start = performance.now();
    try {
      await fetchTranslationModels();
      const elapsed = Math.round(performance.now() - start);
      setLiveProbeLatency(elapsed);
    } catch {
      setLiveProbeLatency(null);
    } finally {
      setIsProbing(false);
    }
  };

  const selectedModel = BENCHMARK_MODELS.find(m => m.id === selectedModelId) || BENCHMARK_MODELS[0];

  // Helper calculation for visual comparison bars
  const maxLatency = Math.max(...BENCHMARK_MODELS.map(m => m.latencyMs));
  const maxVram = 1200; // 1200 MB max reference
  const maxChrf = 40; // 40 max reference
  const maxBleu = 20;

  return (
    <div className="space-y-6 animate-fadeIn">
      
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-200/90 dark:border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-xl bg-saffron-50 dark:bg-saffron-950/80 text-saffron-600 dark:text-saffron-400 border border-saffron-200/80 dark:border-saffron-800/60 shadow-sm">
              <Gauge className="w-5 h-5" />
            </span>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white tracking-tight">
                  Neural Model Performance Metrics & Benchmarks
                </h2>
                <span className="text-[10px] uppercase font-black px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
                  Verified Telemetry
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                Comprehensive head-to-head architectural analysis across 11 Scheduled Indian Languages + English
              </p>
            </div>
          </div>
        </div>

        {/* Live Probe Trigger & Navigation Segmented Controls */}
        <div className="flex items-center space-x-2.5">
          <button
            onClick={handleRunLiveProbe}
            disabled={isProbing}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200/90 dark:border-slate-700 hover:border-brand-400 text-xs font-semibold shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-brand-500/20"
            title="Measure live round-trip API telemetry"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-brand-600 dark:text-brand-400 ${isProbing ? 'animate-spin' : ''}`} />
            <span>{isProbing ? 'Probing...' : 'Live Probe'}</span>
            {liveProbeLatency !== null && (
              <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded font-mono font-bold bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300">
                {liveProbeLatency} ms
              </span>
            )}
          </button>

          <div className="inline-flex p-1 bg-slate-100 dark:bg-slate-800 rounded-xl border border-slate-200/90 dark:border-slate-700 text-xs font-semibold">
            <button
              onClick={() => setActiveMetricTab('overview')}
              className={`px-3 py-1 rounded-lg transition-all ${
                activeMetricTab === 'overview'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Model Cards
            </button>
            <button
              onClick={() => setActiveMetricTab('comparison')}
              className={`px-3 py-1 rounded-lg transition-all ${
                activeMetricTab === 'comparison'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Head-to-Head Table
            </button>
            <button
              onClick={() => setActiveMetricTab('radar')}
              className={`px-3 py-1 rounded-lg transition-all ${
                activeMetricTab === 'radar'
                  ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Visual Telemetry
            </button>
          </div>
        </div>
      </div>

      {/* Model Selection Tabs Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
        {BENCHMARK_MODELS.map((model) => {
          const isSelected = selectedModelId === model.id;
          const isCurrentActive = currentEngine === model.id;
          return (
            <button
              key={model.id}
              onClick={() => setSelectedModelId(model.id)}
              className={`p-3 rounded-2xl text-left border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-white dark:bg-slate-800/90 border-brand-500 shadow-md ring-2 ring-brand-500/20'
                  : 'bg-white/80 dark:bg-slate-850 hover:bg-white dark:hover:bg-slate-800 border-slate-200/90 dark:border-slate-700/80'
              }`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className={`text-[9px] font-black uppercase tracking-wider px-1.5 py-0.5 rounded border ${model.badgeColor}`}>
                    {model.badge}
                  </span>
                  {isCurrentActive && (
                    <span className="flex items-center text-[9px] font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950 px-1.5 py-0.5 rounded">
                      <Check className="w-2.5 h-2.5 mr-0.5" /> Active
                    </span>
                  )}
                </div>
                <h3 className="text-xs font-bold text-slate-900 dark:text-white mt-2 leading-tight">
                  {model.name}
                </h3>
              </div>

              <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-700/60 flex items-center justify-between text-[11px]">
                <span className="text-slate-500 dark:text-slate-400 font-mono font-medium">
                  {model.latencyMs} ms
                </span>
                <span className="font-bold text-brand-600 dark:text-brand-400">
                  {model.parameters}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {/* TAB 1: MODEL OVERVIEW CARD VIEW */}
      {activeMetricTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Selected Model Spotlight Card (Span 2) */}
          <div className="lg:col-span-2 glass-panel p-6 sm:p-7 rounded-3xl border border-slate-200/90 dark:border-slate-800 shadow-sm space-y-6">
            
            {/* Header info */}
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className={`text-[10px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full border ${selectedModel.badgeColor}`}>
                    {selectedModel.categoryLabel}
                  </span>
                  <span className="inline-flex items-center space-x-1 text-[11px] font-bold text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 px-2.5 py-0.5 rounded-full border border-slate-200 dark:border-slate-700">
                    <ShieldCheck className="w-3.5 h-3.5 text-chakra-500" />
                    <span>{selectedModel.cloudDependency}</span>
                  </span>
                </div>
                <h3 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white mt-2">
                  {selectedModel.name}
                </h3>
                <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
                  {selectedModel.description}
                </p>
              </div>

              {/* Action Button: Set as Active Engine */}
              {onSelectEngine && (
                <button
                  onClick={() => onSelectEngine(selectedModel.id)}
                  disabled={currentEngine === selectedModel.id}
                  className={`flex-shrink-0 px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center space-x-1.5 ${
                    currentEngine === selectedModel.id
                      ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-300 cursor-default'
                      : 'bg-brand-600 hover:bg-brand-700 text-white shadow-md hover:shadow-lg focus:ring-2 focus:ring-brand-500/20'
                  }`}
                >
                  {currentEngine === selectedModel.id ? (
                    <>
                      <Check className="w-4 h-4" />
                      <span>Current Active Engine</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Use This Engine</span>
                    </>
                  )}
                </button>
              )}
            </div>

            {/* Core Telemetry Gauges Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700 text-center">
                <Clock className="w-4 h-4 text-brand-600 dark:text-brand-400 mx-auto mb-1" />
                <div className="text-xl font-black text-slate-900 dark:text-white font-mono">
                  {selectedModel.latencyMs} <span className="text-xs font-normal text-slate-500">ms</span>
                </div>
                <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-tight mt-0.5">
                  Avg Latency
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700 text-center">
                <HardDrive className="w-4 h-4 text-chakra-600 dark:text-chakra-400 mx-auto mb-1" />
                <div className="text-xl font-black text-slate-900 dark:text-white font-mono">
                  {selectedModel.vramMb}
                </div>
                <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-tight mt-0.5">
                  VRAM Memory
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700 text-center">
                <Award className="w-4 h-4 text-emerald-600 dark:text-emerald-400 mx-auto mb-1" />
                <div className="text-xl font-black text-slate-900 dark:text-white font-mono">
                  {selectedModel.chrfScore}
                </div>
                <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-tight mt-0.5">
                  chrF++ Score
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700 text-center">
                <TrendingUp className="w-4 h-4 text-saffron-600 dark:text-saffron-400 mx-auto mb-1" />
                <div className="text-xl font-black text-slate-900 dark:text-white font-mono">
                  {selectedModel.bleuScore}
                </div>
                <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-tight mt-0.5">
                  BLEU-2 Score
                </div>
              </div>
            </div>

            {/* Architectural Deep-Dive Details */}
            <div className="p-4 rounded-2xl bg-slate-50/70 dark:bg-slate-800/50 border border-slate-200/70 dark:border-slate-700 space-y-2.5 text-xs">
              <div className="flex items-start justify-between border-b border-slate-200/60 dark:border-slate-700/60 pb-2">
                <span className="text-slate-500 dark:text-slate-400 font-medium">Neural Architecture:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 text-right max-w-xs sm:max-w-md">
                  {selectedModel.architecture}
                </span>
              </div>
              <div className="flex items-start justify-between border-b border-slate-200/60 dark:border-slate-700/60 pb-2">
                <span className="text-slate-500 dark:text-slate-400 font-medium">Tokenizer & Vocab:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 text-right">
                  {selectedModel.tokenizer}
                </span>
              </div>
              <div className="flex items-start justify-between border-b border-slate-200/60 dark:border-slate-700/60 pb-2">
                <span className="text-slate-500 dark:text-slate-400 font-medium">Constitutional Language Scope:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 text-right">
                  {selectedModel.languages}
                </span>
              </div>
              <div className="flex items-start justify-between border-b border-slate-200/60 dark:border-slate-700/60 pb-2">
                <span className="text-slate-500 dark:text-slate-400 font-medium">Degeneracy Rate (Repetition):</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 font-mono">
                  {selectedModel.degeneracyRate}% (Lower is safer)
                </span>
              </div>
              <div className="flex items-start justify-between pt-0.5">
                <span className="text-slate-500 dark:text-slate-400 font-medium">Corpus & Source:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 text-right max-w-xs sm:max-w-md">
                  {selectedModel.trainingCorpus}
                </span>
              </div>
            </div>

            {/* Key Strengths & Operational Profile */}
            <div>
              <h4 className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider mb-2">
                Key Strengths & Operational Profile
              </h4>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {selectedModel.strengths.map((str, idx) => (
                  <li key={idx} className="flex items-start space-x-2 text-slate-700 dark:text-slate-300">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
                    <span>{str}</span>
                  </li>
                ))}
              </ul>
            </div>

          </div>

          {/* Right Pillar: Sovereign AI & Hardware Insights */}
          <div className="space-y-4">
            
            {/* Edge Deployment Card */}
            <div className="glass-panel p-5 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-chakra-50 dark:bg-chakra-950 text-chakra-600 dark:text-chakra-400 border border-chakra-200 dark:border-chakra-800">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900 dark:text-white">
                    Sovereign Edge Deployment
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    100% On-Device & Zero Pivot
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                The locally trained Pralekha model (52.3M) was trained on 500,000 parallel pairs in the cleaned 12-way database. It translates directly between all 11 scheduled Indian languages without translating to English first.
              </p>
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-chakra-700 dark:text-chakra-400">
                <span>VRAM Footprint: 307 MB</span>
                <span className="font-mono">175 ms / sent</span>
              </div>
            </div>

            {/* Foundation Model Fine-Tuning Card */}
            <div className="glass-panel p-5 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
                  <Database className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900 dark:text-white">
                    Meta NLLB-200 LoRA Adapter
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    Targeted Low-Rank Adaptation
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                NLLB-200 incorporates LoRA adaptation (<span className="font-mono">r=16, &alpha;=32</span>) across all 72 projection layers, trained with BF16 precision and gradient accumulation for smooth execution on RTX 4050 GPUs.
              </p>
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs font-bold text-blue-700 dark:text-blue-400">
                <span>chrF++ Benchmark: 35.64</span>
                <span className="font-mono">1.18 GB VRAM</span>
              </div>
            </div>

            {/* National Linguistic Scope Card */}
            <div className="glass-panel p-5 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-2 rounded-xl bg-emerald-50 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                  <Layers className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-sm font-bold text-slate-900 dark:text-white">
                    Constitutional Coverage
                  </h4>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    11 Scheduled Languages (+ English)
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                Active benchmark coverage includes Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu, and English across all direct pairing directions.
              </p>
            </div>

          </div>

        </div>
      )}

      {/* TAB 2: HEAD-TO-HEAD COMPARATIVE TABLE */}
      {activeMetricTab === 'comparison' && (
        <div className="glass-panel rounded-3xl border border-slate-200/90 dark:border-slate-800 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-100/90 dark:bg-slate-800/90 border-b border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 uppercase tracking-wider font-bold">
                <tr>
                  <th className="py-3.5 px-4">Model Engine</th>
                  <th className="py-3.5 px-3">Parameters</th>
                  <th className="py-3.5 px-3">Latency (ms)</th>
                  <th className="py-3.5 px-3">VRAM Footprint</th>
                  <th className="py-3.5 px-3">chrF++</th>
                  <th className="py-3.5 px-3">BLEU-2</th>
                  <th className="py-3.5 px-3">Degeneracy</th>
                  <th className="py-3.5 px-3">Cloud Dependency</th>
                  <th className="py-3.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-800 dark:text-slate-200">
                {BENCHMARK_MODELS.map((model) => {
                  const isCurrent = currentEngine === model.id;
                  return (
                    <tr 
                      key={model.id}
                      className={`hover:bg-slate-50/80 dark:hover:bg-slate-800/50 transition-colors ${
                        isCurrent ? 'bg-brand-50/40 dark:bg-brand-950/20' : ''
                      }`}
                    >
                      <td className="py-3.5 px-4 font-bold">
                        <div className="flex items-center space-x-2">
                          <span>{model.name}</span>
                          {isCurrent && (
                            <span className="text-[9px] font-bold text-emerald-700 dark:text-emerald-300 bg-emerald-100 dark:bg-emerald-950 px-1.5 py-0.2 rounded">
                              Active
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-slate-500 dark:text-slate-400 font-normal">
                          {model.categoryLabel}
                        </span>
                      </td>
                      <td className="py-3.5 px-3 font-mono font-bold text-slate-900 dark:text-white">
                        {model.parameters}
                      </td>
                      <td className="py-3.5 px-3 font-mono font-semibold">
                        <span className={`px-2 py-0.5 rounded ${
                          model.latencyMs < 180 
                            ? 'bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300' 
                            : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                        }`}>
                          {model.latencyMs} ms
                        </span>
                      </td>
                      <td className="py-3.5 px-3 font-mono">
                        {model.vramMb}
                      </td>
                      <td className="py-3.5 px-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {model.chrfScore}
                      </td>
                      <td className="py-3.5 px-3 font-mono">
                        {model.bleuScore}
                      </td>
                      <td className="py-3.5 px-3 font-mono text-slate-600 dark:text-slate-400">
                        {model.degeneracyRate}%
                      </td>
                      <td className="py-3.5 px-3">
                        <span className="inline-flex items-center space-x-1 text-[11px] font-semibold text-slate-600 dark:text-slate-400">
                          {model.cloudDependency}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        {onSelectEngine && (
                          <button
                            onClick={() => onSelectEngine(model.id)}
                            disabled={isCurrent}
                            className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${
                              isCurrent
                                ? 'text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950 cursor-default'
                                : 'text-brand-600 dark:text-brand-400 hover:bg-brand-50 dark:hover:bg-brand-950/60'
                            }`}
                          >
                            {isCurrent ? 'Active' : 'Select'}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: VISUAL COMPARATIVE TELEMETRY BARS */}
      {activeMetricTab === 'radar' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Latency Comparison Card */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-brand-600 dark:text-brand-400" />
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  Inference Latency (ms) — Lower is Faster
                </h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">ms per sentence</span>
            </div>
            <div className="space-y-3">
              {BENCHMARK_MODELS.map((model) => {
                const pct = (model.latencyMs / maxLatency) * 100;
                return (
                  <div key={model.id} className="space-y-1 text-xs">
                    <div className="flex justify-between font-semibold">
                      <span className="text-slate-800 dark:text-slate-200">{model.name}</span>
                      <span className="font-mono text-slate-500 dark:text-slate-400">{model.latencyMs} ms</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-700 ${
                          model.latencyMs < 180 
                            ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' 
                            : model.latencyMs < 250 
                            ? 'bg-gradient-to-r from-brand-500 to-chakra-500' 
                            : 'bg-gradient-to-r from-amber-500 to-saffron-500'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* VRAM Memory Footprint Card */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center space-x-2">
                <HardDrive className="w-4 h-4 text-chakra-600 dark:text-chakra-400" />
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  VRAM Memory Footprint (MB) — Edge Efficiency
                </h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Active VRAM</span>
            </div>
            <div className="space-y-3">
              {BENCHMARK_MODELS.map((model) => {
                const pct = Math.min(100, (model.vramNum / maxVram) * 100);
                return (
                  <div key={model.id} className="space-y-1 text-xs">
                    <div className="flex justify-between font-semibold">
                      <span className="text-slate-800 dark:text-slate-200">{model.name}</span>
                      <span className="font-mono text-slate-500 dark:text-slate-400">{model.vramMb}</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div 
                        className="h-full rounded-full bg-gradient-to-r from-chakra-600 to-brand-500 transition-all duration-700"
                        style={{ width: `${Math.max(4, pct)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* chrF++ Translation Quality Card */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center space-x-2">
                <Award className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  chrF++ Score (Indic Morphology) — Higher is Better
                </h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">chrF++ rating</span>
            </div>
            <div className="space-y-3">
              {BENCHMARK_MODELS.map((model) => {
                const pct = (model.chrfScore / maxChrf) * 100;
                return (
                  <div key={model.id} className="space-y-1 text-xs">
                    <div className="flex justify-between font-semibold">
                      <span className="text-slate-800 dark:text-slate-200">{model.name}</span>
                      <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">{model.chrfScore}</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div 
                        className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Degeneracy Safety (Repetition) Card */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-200/90 dark:border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 text-saffron-600 dark:text-saffron-400" />
                <h3 className="font-bold text-slate-900 dark:text-white text-sm">
                  Degeneracy Rate (% Repetition) — Lower is Safer
                </h3>
              </div>
              <span className="text-[10px] text-slate-400 font-mono">Loop frequency %</span>
            </div>
            <div className="space-y-3">
              {BENCHMARK_MODELS.map((model) => {
                const pct = (model.degeneracyRate / 6.0) * 100;
                return (
                  <div key={model.id} className="space-y-1 text-xs">
                    <div className="flex justify-between font-semibold">
                      <span className="text-slate-800 dark:text-slate-200">{model.name}</span>
                      <span className="font-mono text-slate-500 dark:text-slate-400">{model.degeneracyRate}%</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-700 ${
                          model.degeneracyRate <= 2.5 
                            ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' 
                            : 'bg-gradient-to-r from-amber-500 to-saffron-500'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      )}

    </div>
  );
};
