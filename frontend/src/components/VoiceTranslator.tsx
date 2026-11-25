import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  ArrowLeftRight, 
  Trash2, 
  Sparkles, 
  Play, 
  Pause, 
  RotateCcw, 
  Settings2,
  Copy,
  Check,
  CheckCircle2,
  Radio,
  Sliders,
  MessageSquare,
  HelpCircle,
  ExternalLink,
  ChevronDown,
  User,
  Users,
  CornerDownRight,
  Headphones,
  Zap
} from 'lucide-react';
import { Language } from '../types';
import { translateText, playSpeechAudio, streamSpeechAudio } from '../services/api';
import { LanguageSelector } from './LanguageSelector';
import { useToast } from '../context/ToastContext';

export interface VoiceTranslatorProps {
  languages: Language[];
}

export interface MessageBubble {
  id: string;
  sender: 'speaker1' | 'speaker2';
  speakerLabel: string;
  originalText: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
  timestamp: string;
}

export type RecognitionState = 'standby' | 'listening' | 'transcribing' | 'translating' | 'synthesizing' | 'error';

export const VoiceTranslator: React.FC<VoiceTranslatorProps> = ({ languages }) => {
  const { showSuccess, showError, showInfo } = useToast();

  // Speaker configuration
  const [sourceLang, setSourceLang] = useState<string>('hi');
  const [targetLang, setTargetLang] = useState<string>('ta');
  const [activeSpeaker, setActiveSpeaker] = useState<'speaker1' | 'speaker2'>('speaker1');

  // Audio & Speech status
  const [isListening, setIsListening] = useState<boolean>(false);
  const [recognitionState, setRecognitionState] = useState<RecognitionState>('standby');
  const [statusMessage, setStatusMessage] = useState<string>('Microphone ready');
  
  // Voice Controls
  const [speechRate, setSpeechRate] = useState<number>(1.0);
  const [speechPitch, setSpeechPitch] = useState<'natural' | 'low' | 'expressive'>('natural');
  const [autoPlayTTS, setAutoPlayTTS] = useState<boolean>(true);

  // Content state
  const [transcript, setTranscript] = useState<string>('');
  const [translatedResult, setTranslatedResult] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isSynthesizing, setIsSynthesizing] = useState<boolean>(false);
  const [conversation, setConversation] = useState<MessageBubble[]>([]);
  const [playingBubbleId, setPlayingBubbleId] = useState<string | null>(null);
  const [copiedField, setCopiedField] = useState<'source' | 'target' | string | null>(null);
  const [isSwapping, setIsSwapping] = useState<boolean>(false);

  // Audio wave visualizer canvas refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Web Speech API Ref
  const recognitionRef = useRef<any>(null);

  // Quick conversation starters for Indian multilingual scenarios
  const quickPrompts = [
    { text: 'नमस्ते, आप कैसे हैं?', label: 'Namaste, How are you? (Hindi)', lang: 'hi', target: 'ta' },
    { text: 'இந்த இடம் எங்கே இருக்கிறது?', label: 'Where is this place? (Tamil)', lang: 'ta', target: 'hi' },
    { text: 'దగ్గరలోని ఆसुపత్రి ఎక్కడ ఉంది?', label: 'Nearest hospital? (Telugu)', lang: 'te', target: 'hi' },
    { text: 'ট্রেন কখন ছাড়বে?', label: 'When does train leave? (Bengali)', lang: 'bn', target: 'hi' },
    { text: 'मला मदत हवी आहे', label: 'I need help (Marathi)', lang: 'mr', target: 'ta' },
    { text: 'Where is the railway station?', label: 'Station directions (English)', lang: 'en', target: 'hi' }
  ];

  // Initialize Audio Wave Visualizer Loop with Gradient Bars
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let phase = 0;

    const renderWave = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const midY = height / 2;

      if (isListening && analyserRef.current) {
        // Real-time audio frequency data from microphone
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Draw dynamic pulsing waveform bars
        const numBars = 42;
        const barWidth = (width / numBars) * 0.65;
        const gap = (width / numBars) * 0.35;

        for (let i = 0; i < numBars; i++) {
          const sampleIdx = Math.floor((i / numBars) * (bufferLength / 2));
          const val = dataArray[sampleIdx] || 0;
          const barHeight = Math.max(8, (val / 255) * (height * 0.88));

          const x = i * (barWidth + gap) + gap / 2;
          const y = midY - barHeight / 2;

          // Gradient color transition: Chakra Navy / Saffron / Emerald
          const grad = ctx.createLinearGradient(0, y, 0, y + barHeight);
          if (activeSpeaker === 'speaker1') {
            grad.addColorStop(0, '#38bdf8');
            grad.addColorStop(0.5, '#4f46e5');
            grad.addColorStop(1, '#a855f7');
          } else {
            grad.addColorStop(0, '#f97316');
            grad.addColorStop(0.5, '#10b981');
            grad.addColorStop(1, '#059669');
          }

          ctx.fillStyle = grad;
          ctx.beginPath();
          if (typeof (ctx as any).roundRect === 'function') {
            (ctx as any).roundRect(x, y, barWidth, barHeight, barWidth / 2);
          } else {
            ctx.rect(x, y, barWidth, barHeight);
          }
          ctx.fill();
        }
      } else if (isListening) {
        // Procedural animated sound wave fallback if audio context analyser is pending
        phase += 0.08;
        ctx.beginPath();
        ctx.strokeStyle = activeSpeaker === 'speaker1' ? '#6366f1' : '#10b981';
        ctx.lineWidth = 3;

        for (let x = 0; x < width; x++) {
          const y = midY + Math.sin(x * 0.04 + phase) * 16 * Math.sin(phase * 0.5) + Math.cos(x * 0.02 - phase) * 8;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else {
        // Resting ambient wave
        phase += 0.02;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
        ctx.lineWidth = 1.5;

        for (let x = 0; x < width; x++) {
          const y = midY + Math.sin(x * 0.02 + phase) * 4;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animFrameRef.current = requestAnimationFrame(renderWave);
    };

    renderWave();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isListening, activeSpeaker]);

  // Audio Context & Mic Stream Setup for Visualizer
  const setupAudioAnalyser = async () => {
    try {
      if (!audioContextRef.current) {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          audioContextRef.current = new AudioCtx();
        }
      }

      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      if (!micStreamRef.current) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        micStreamRef.current = stream;

        if (audioContextRef.current) {
          const source = audioContextRef.current.createMediaStreamSource(stream);
          const analyser = audioContextRef.current.createAnalyser();
          analyser.fftSize = 128;
          analyser.smoothingTimeConstant = 0.8;
          source.connect(analyser);
          analyserRef.current = analyser;
        }
      }
    } catch (e) {
      console.warn('Audio analyser fallback (visualizer will use procedural wave):', e);
    }
  };

  const cleanupAudioAnalyser = () => {
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    analyserRef.current = null;
  };

  // Convert pitch label to frequency multiplier
  const getPitchMultiplier = () => {
    if (speechPitch === 'low') return 0.85;
    if (speechPitch === 'expressive') return 1.15;
    return 1.0;
  };

  // Process speech translation
  const processVoiceInput = async (spokenText: string, currentSpeaker: 'speaker1' | 'speaker2' = activeSpeaker) => {
    if (!spokenText.trim()) return;
    setIsProcessing(true);
    setRecognitionState('translating');

    const fromLang = currentSpeaker === 'speaker1' ? sourceLang : targetLang;
    const toLang = currentSpeaker === 'speaker1' ? targetLang : sourceLang;
    const speakerTitle = currentSpeaker === 'speaker1' ? 'Speaker 1' : 'Speaker 2';

    try {
      const res = await translateText(fromLang, toLang, spokenText, 'voice');
      setTranslatedResult(res.translated_text);
      setStatusMessage('Translation complete');
      setRecognitionState('standby');
      showSuccess('Voice translation complete');

      // Append to conversational exchange timeline
      const newBubble: MessageBubble = {
        id: Date.now().toString(),
        sender: currentSpeaker,
        speakerLabel: speakerTitle,
        originalText: spokenText,
        translatedText: res.translated_text,
        sourceLang: fromLang,
        targetLang: toLang,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setConversation((prev) => [newBubble, ...prev]);

      // Automatically synthesize spoken output in target language if enabled
      if (autoPlayTTS) {
        setIsSynthesizing(true);
        setRecognitionState('synthesizing');
        try {
          await streamSpeechAudio(res.translated_text, toLang, speechRate);
        } catch (synthErr) {
          await playSpeechAudio(res.translated_text, toLang, speechRate);
        } finally {
          setIsSynthesizing(false);
          setRecognitionState('standby');
        }
      }
    } catch (e: any) {
      const msg = `Translation error: ${e.message}`;
      setStatusMessage(msg);
      setRecognitionState('error');
      showError(msg);
    } finally {
      setIsProcessing(false);
    }
  };

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;

      recognition.onstart = () => {
        setIsListening(true);
        setRecognitionState('listening');
        setStatusMessage('Listening to spoken words...');
        setupAudioAnalyser();
      };

      recognition.onresult = async (event: any) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);
        setRecognitionState('transcribing');

        if (event.results[0].isFinal) {
          setIsListening(false);
          cleanupAudioAnalyser();
          setStatusMessage('Translating speech...');
          await processVoiceInput(currentTranscript, activeSpeaker);
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        cleanupAudioAnalyser();
        setRecognitionState('error');
        setStatusMessage(`Mic notice: ${event.error}`);
        showError(`Microphone notice: ${event.error}`);
      };

      recognition.onend = () => {
        setIsListening(false);
        cleanupAudioAnalyser();
        if (recognitionState === 'listening') {
          setRecognitionState('standby');
        }
      };

      recognitionRef.current = recognition;
    } else {
      setStatusMessage('Web Speech API ready in fallback simulation mode');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      cleanupAudioAnalyser();
    };
  }, [sourceLang, targetLang, speechRate, autoPlayTTS, activeSpeaker]);

  // Toggle Listening for a specific speaker
  const toggleListening = async (speaker: 'speaker1' | 'speaker2' = activeSpeaker) => {
    setActiveSpeaker(speaker);

    if (!recognitionRef.current) {
      // Fallback: prompt text simulation
      const sample = prompt(`Enter phrase in Indian language to simulate voice input for ${speaker === 'speaker1' ? 'Speaker 1' : 'Speaker 2'}:`);
      if (sample) {
        setTranscript(sample);
        processVoiceInput(sample, speaker);
      }
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      setRecognitionState('standby');
      cleanupAudioAnalyser();
    } else {
      const currentCode = speaker === 'speaker1' ? sourceLang : targetLang;
      const srcMeta = languages.find((l) => l.code === currentCode);
      recognitionRef.current.lang = srcMeta?.speech_recognition_code || `${currentCode}-IN`;
      setTranscript('');
      setTranslatedResult('');
      try {
        await setupAudioAnalyser();
        recognitionRef.current.start();
      } catch (e: any) {
        console.warn('Speech start issue:', e);
        showError('Could not start microphone: ' + (e.message || 'Permission denied'));
        setRecognitionState('error');
      }
    }
  };

  const handleManualSpeakOutput = async (text: string, lang: string) => {
    if (!text) return;
    setIsSynthesizing(true);
    setRecognitionState('synthesizing');
    try {
      await streamSpeechAudio(text, lang, speechRate);
      showInfo('Playing pronunciation audio');
    } catch (e) {
      try {
        await playSpeechAudio(text, lang, speechRate);
        showInfo('Playing pronunciation audio');
      } catch (err) {
        showError('Audio playback failed');
      }
    } finally {
      setIsSynthesizing(false);
      setRecognitionState('standby');
    }
  };

  const handleReplayBubble = async (bubble: MessageBubble) => {
    setPlayingBubbleId(bubble.id);
    try {
      await streamSpeechAudio(bubble.translatedText, bubble.targetLang, speechRate);
    } catch (e) {
      try {
        await playSpeechAudio(bubble.translatedText, bubble.targetLang, speechRate);
      } catch (err) {
        showError('Audio playback failed');
      }
    } finally {
      setPlayingBubbleId(null);
    }
  };

  const handleSwap = () => {
    setIsSwapping(true);
    setTimeout(() => setIsSwapping(false), 500);

    const temp = sourceLang;
    setSourceLang(targetLang);
    setTargetLang(temp);
    setTranscript('');
    setTranslatedResult('');
    showInfo(`Swapped speaker languages`);
  };

  const handleCopy = (text: string, field: 'source' | 'target' | string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    showSuccess(`Copied text to clipboard`);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const handleSimulatePrompt = (promptItem: { text: string; lang: string; target: string }) => {
    setSourceLang(promptItem.lang);
    setTargetLang(promptItem.target);
    setActiveSpeaker('speaker1');
    setTranscript(promptItem.text);
    processVoiceInput(promptItem.text, 'speaker1');
  };

  const srcMeta = languages.find((l) => l.code === sourceLang);
  const tgtMeta = languages.find((l) => l.code === targetLang);

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-20">
      
      {/* Top Studio Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-brand-50 border border-brand-200 text-xs font-bold text-brand-700">
          <Sparkles className="w-3.5 h-3.5 text-brand-600" />
          <span>Real-Time Indic Speech-to-Speech Engine</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Bilingual Conversational Voice Studio
        </h1>
        <p className="text-sm text-slate-500">
          Speak in any of the 11 Scheduled Indian languages (+ English) for instant neural transcription, cross-lingual translation, and native vocal synthesis.
        </p>
      </div>

      {/* Main Microphone Interaction Stage with Pulsing Visualizer & Waveform */}
      <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-xl flex flex-col items-center justify-center space-y-6 text-center relative overflow-hidden">
        
        {/* Subtle decorative background gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-brand-50/50 via-transparent to-purple-50/30 pointer-events-none" />

        {/* Live Audio Wave Visualizer Canvas */}
        <div className="w-full max-w-lg h-20 rounded-2xl bg-slate-900/95 border border-slate-800 p-2 relative shadow-inner overflow-hidden flex items-center justify-center">
          <canvas
            ref={canvasRef}
            width={480}
            height={64}
            className="w-full h-full object-contain"
          />
          <div className="absolute top-2 left-3 flex items-center space-x-1.5 text-[10px] font-mono text-slate-400">
            <Radio className={`w-3 h-3 ${isListening ? 'text-red-400 animate-pulse' : 'text-slate-500'}`} />
            <span>{isListening ? 'AUDIO STREAM ACTIVE' : 'AUDIO STANDBY'}</span>
          </div>

          <div className="absolute top-2 right-3 flex items-center space-x-1 text-[10px] font-bold text-slate-300">
            <Headphones className="w-3 h-3 text-brand-400" />
            <span>{activeSpeaker === 'speaker1' ? srcMeta?.name || 'Speaker 1' : tgtMeta?.name || 'Speaker 2'}</span>
          </div>
        </div>

        {/* Big Pulsing Mic Button with Concentric Glow Waves */}
        <div className="relative z-10 flex flex-col items-center">
          {/* Animated pulsing glow rings when listening */}
          {isListening && (
            <>
              <div className="absolute -inset-4 rounded-full bg-red-500/20 animate-ping pointer-events-none" />
              <div className="absolute -inset-8 rounded-full bg-red-500/10 animate-pulse pointer-events-none" />
            </>
          )}

          <button
            type="button"
            onClick={() => toggleListening(activeSpeaker)}
            title={isListening ? 'Stop Recording' : 'Start Voice Translation'}
            className={`w-28 h-28 rounded-full flex flex-col items-center justify-center shadow-2xl transition-all duration-300 transform active:scale-95 cursor-pointer relative z-20 focus:outline-none focus:ring-4 focus:ring-brand-500/30 ${
              isListening
                ? 'bg-gradient-to-tr from-red-600 via-rose-500 to-amber-500 text-white shadow-red-500/50 scale-105 ring-4 ring-red-300/60'
                : 'bg-gradient-to-tr from-brand-600 via-brand-500 to-indigo-600 text-white shadow-brand-500/30 hover:scale-105 hover:shadow-brand-500/40 ring-4 ring-brand-100'
            }`}
          >
            {isListening ? (
              <MicOff className="w-10 h-10 animate-pulse" />
            ) : (
              <Mic className="w-10 h-10" />
            )}
            <span className="text-[10px] font-extrabold uppercase tracking-wider mt-1">
              {isListening ? 'Stop' : 'Speak'}
            </span>
          </button>

          {/* Active Speaker Toggle Pills */}
          <div className="mt-3 inline-flex p-1 bg-slate-100 rounded-2xl border border-slate-200">
            <button
              type="button"
              onClick={() => setActiveSpeaker('speaker1')}
              className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${
                activeSpeaker === 'speaker1'
                  ? 'bg-brand-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🎤 Speaker 1 ({srcMeta?.name || 'Source'})
            </button>
            <button
              type="button"
              onClick={() => setActiveSpeaker('speaker2')}
              className={`px-3 py-1 rounded-xl text-xs font-bold transition-all ${
                activeSpeaker === 'speaker2'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              🎤 Speaker 2 ({tgtMeta?.name || 'Target'})
            </button>
          </div>
        </div>

        {/* Speech Recognition Status Pill */}
        <div className="space-y-1.5 z-10">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs font-bold text-slate-700">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                recognitionState === 'listening'
                  ? 'bg-red-500 animate-ping'
                  : recognitionState === 'transcribing'
                  ? 'bg-amber-500 animate-pulse'
                  : recognitionState === 'translating'
                  ? 'bg-purple-500 animate-spin'
                  : recognitionState === 'synthesizing'
                  ? 'bg-emerald-500 animate-pulse'
                  : 'bg-emerald-500'
              }`}
            />
            <span>
              {recognitionState === 'listening'
                ? `Listening to ${activeSpeaker === 'speaker1' ? srcMeta?.name : tgtMeta?.name}...`
                : recognitionState === 'transcribing'
                ? 'Recognizing speech tokens...'
                : recognitionState === 'translating'
                ? 'Processing neural translation...'
                : recognitionState === 'synthesizing'
                ? 'Streaming vocal synthesis...'
                : statusMessage}
            </span>
          </div>
          <p className="text-xs text-slate-500 font-medium">
            {isListening
              ? 'Speak naturally into device microphone. Tap stop when finished.'
              : `Tap the microphone and speak in ${activeSpeaker === 'speaker1' ? srcMeta?.name || 'your language' : tgtMeta?.name || 'target language'}`}
          </p>
        </div>

        {/* Global Controls Toolbar: Speech Speed, Voice Pitch, Auto-Play Toggle */}
        <div className="w-full max-w-xl pt-4 border-t border-slate-200/80 flex flex-wrap items-center justify-between gap-3 text-xs z-10">
          
          {/* Auto-play toggle */}
          <label className="flex items-center space-x-2 cursor-pointer font-bold text-slate-700 select-none">
            <input
              type="checkbox"
              checked={autoPlayTTS}
              onChange={(e) => setAutoPlayTTS(e.target.checked)}
              className="w-4 h-4 rounded text-brand-600 accent-brand-600 focus:ring-brand-500 cursor-pointer"
            />
            <span className="flex items-center space-x-1">
              <Volume2 className="w-3.5 h-3.5 text-brand-600" />
              <span>Auto-Play Spoken Speech</span>
            </span>
          </label>

          {/* Voice Pitch Controls */}
          <div className="flex items-center space-x-1.5">
            <span className="text-slate-500 font-semibold">Pitch:</span>
            {(['low', 'natural', 'expressive'] as const).map((pitch) => (
              <button
                key={pitch}
                type="button"
                onClick={() => setSpeechPitch(pitch)}
                className={`px-2 py-0.5 rounded-lg text-[11px] font-bold capitalize transition-all ${
                  speechPitch === pitch
                    ? 'bg-indigo-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {pitch}
              </button>
            ))}
          </div>

          {/* Speech Playback Speed Selector */}
          <div className="flex items-center space-x-1.5">
            <span className="text-slate-500 font-semibold">Speed:</span>
            {[0.75, 1.0, 1.25, 1.5].map((rate) => (
              <button
                key={rate}
                type="button"
                onClick={() => setSpeechRate(rate)}
                className={`px-2 py-0.5 rounded-lg text-[11px] font-bold transition-all ${
                  speechRate === rate
                    ? 'bg-brand-600 text-white shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>

        </div>

      </div>

      {/* Dual Speaker Conversational Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-stretch relative">
        
        {/* Central Bidirectional Swap Button for Desktop */}
        <div className="hidden md:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20">
          <button
            type="button"
            onClick={handleSwap}
            title="Swap Speaker 1 and Speaker 2 Languages"
            className={`w-11 h-11 rounded-2xl bg-white border border-slate-200/90 shadow-lg text-brand-600 hover:text-white hover:bg-brand-600 flex items-center justify-center transition-all duration-300 hover:scale-110 active:scale-95 focus:outline-none focus:ring-2 focus:ring-brand-500 ${
              isSwapping ? 'rotate-180 scale-125' : ''
            }`}
          >
            <ArrowLeftRight className="w-5 h-5" />
          </button>
        </div>

        {/* SPEAKER 1 CARD: Primary Input Speaker */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-200 flex flex-col justify-between space-y-4 shadow-sm hover:shadow-card-hover transition-all">
          
          {/* Header: Speaker Identity & Language Selector */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-brand-600" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                Speaker 1 ({srcMeta?.name || 'Input'})
              </span>
            </div>
            <LanguageSelector
              languages={languages}
              value={sourceLang}
              onChange={setSourceLang}
              placeholder="Speaker 1 Language"
            />
          </div>

          {/* Transcript Content Box */}
          <div className="min-h-[140px] p-4 rounded-2xl bg-slate-50 border border-slate-200/80 flex flex-col justify-between">
            {transcript ? (
              <p className="text-base sm:text-lg font-bold text-slate-900 leading-relaxed font-indic">
                "{transcript}"
              </p>
            ) : (
              <div className="text-slate-400 text-sm italic flex flex-col items-center justify-center h-full my-auto text-center py-6 select-none">
                <Mic className="w-6 h-6 mb-1.5 opacity-40" />
                <span>Speak into microphone or select a prompt below...</span>
              </div>
            )}

            {transcript && (
              <div className="text-[11px] text-slate-400 font-mono text-right pt-2 border-t border-slate-200/50">
                {transcript.split(/\s+/).filter(Boolean).length} words &bull; {transcript.length} chars
              </div>
            )}
          </div>

          {/* Speaker 1 Footer Actions */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => setTranscript('')}
                disabled={!transcript}
                className="text-xs text-slate-400 hover:text-red-500 disabled:opacity-30 flex items-center space-x-1 font-semibold transition-colors focus:ring-1 focus:ring-red-400"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
              
              {transcript && (
                <button
                  type="button"
                  onClick={() => handleManualSpeakOutput(transcript, sourceLang)}
                  className="text-xs text-brand-600 hover:text-brand-800 flex items-center space-x-1 font-semibold transition-colors"
                  title="Listen to original transcript"
                >
                  <Volume2 className="w-3.5 h-3.5" />
                  <span>Listen</span>
                </button>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => toggleListening('speaker1')}
                className={`flex items-center space-x-1 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                  isListening && activeSpeaker === 'speaker1'
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-brand-50 hover:bg-brand-100 text-brand-700'
                }`}
              >
                <Mic className="w-3.5 h-3.5" />
                <span>{isListening && activeSpeaker === 'speaker1' ? 'Recording...' : 'Speak'}</span>
              </button>

              <button
                type="button"
                onClick={() => handleCopy(transcript, 'source')}
                disabled={!transcript}
                className="flex items-center space-x-1 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 disabled:opacity-30 text-xs font-bold text-slate-700 transition-colors focus:ring-1 focus:ring-brand-500"
              >
                {copiedField === 'source' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedField === 'source' ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>

        </div>

        {/* SPEAKER 2 CARD: Target Translation Speaker */}
        <div className="glass-panel p-5 rounded-3xl border border-brand-200/80 bg-gradient-to-br from-white via-brand-50/20 to-purple-50/20 flex flex-col justify-between space-y-4 shadow-sm hover:shadow-card-hover transition-all">
          
          {/* Header: Speaker 2 Language Selector */}
          <div className="flex items-center justify-between pb-3 border-b border-brand-100">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-xs font-bold uppercase tracking-wider text-brand-700">
                Speaker 2 ({tgtMeta?.name || 'Output'})
              </span>
            </div>
            <LanguageSelector
              languages={languages}
              value={targetLang}
              onChange={setTargetLang}
              placeholder="Speaker 2 Language"
            />
          </div>

          {/* Translation Content Box */}
          <div className="min-h-[140px] p-4 rounded-2xl bg-white border border-brand-200/70 flex flex-col justify-between shadow-xs">
            {translatedResult ? (
              <p className="text-base sm:text-xl font-extrabold text-brand-900 leading-relaxed font-indic">
                {translatedResult}
              </p>
            ) : (
              <div className="text-slate-400 text-sm italic flex flex-col items-center justify-center h-full my-auto text-center py-6 select-none">
                <Volume2 className="w-6 h-6 mb-1.5 opacity-40" />
                <span>Spoken translation will appear and synthesize here...</span>
              </div>
            )}

            {translatedResult && (
              <div className="flex items-center justify-between text-[11px] text-brand-700 font-semibold pt-2 border-t border-brand-50">
                <span className="flex items-center space-x-1">
                  <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                  <span>Indic Neural Vocals Ready</span>
                </span>
                <span className="font-mono">{speechRate}x speed</span>
              </div>
            )}
          </div>

          {/* Speaker 2 Footer Actions */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={() => toggleListening('speaker2')}
                className={`flex items-center space-x-1 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                  isListening && activeSpeaker === 'speaker2'
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-800'
                }`}
              >
                <Mic className="w-3.5 h-3.5" />
                <span>{isListening && activeSpeaker === 'speaker2' ? 'Recording...' : 'Reply'}</span>
              </button>

              <button
                type="button"
                onClick={() => handleCopy(translatedResult, 'target')}
                disabled={!translatedResult}
                className="flex items-center space-x-1 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 disabled:opacity-30 text-xs font-bold text-slate-700 transition-colors focus:ring-1 focus:ring-brand-500"
              >
                {copiedField === 'target' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copiedField === 'target' ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            <button
              type="button"
              onClick={() => handleManualSpeakOutput(translatedResult, targetLang)}
              disabled={!translatedResult || isSynthesizing}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:scale-105 active:scale-95 disabled:opacity-40 text-white font-bold text-xs shadow-md shadow-brand-500/20 transition-all focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <Volume2 className={`w-4 h-4 ${isSynthesizing ? 'animate-pulse' : ''}`} />
              <span>{isSynthesizing ? 'Speaking...' : 'Listen Translation'}</span>
            </button>
          </div>

        </div>

      </div>

      {/* Quick Interactive Prompt Starters */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-200 space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold text-slate-800">
          <Sparkles className="w-4 h-4 text-brand-600" />
          <span>Quick Indic Conversation Starters (Click to translate & synthesize):</span>
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          {quickPrompts.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSimulatePrompt(p)}
              className="px-3 py-1.5 rounded-xl bg-white hover:bg-brand-50 hover:text-brand-700 hover:border-brand-300 border border-slate-200 text-xs font-semibold text-slate-700 shadow-sm transition-all hover:scale-105 active:scale-95 focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Turn-by-Turn Conversational Exchange Timeline with Transcript Bubbles */}
      {conversation.length > 0 && (
        <div className="glass-panel p-6 rounded-3xl border border-slate-200 space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center space-x-2">
              <MessageSquare className="w-5 h-5 text-brand-600" />
              <h3 className="font-bold text-slate-900 text-base">Conversational Exchange Timeline</h3>
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold">
                {conversation.length} {conversation.length === 1 ? 'turn' : 'turns'}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setConversation([])}
              className="text-xs text-slate-400 hover:text-red-500 flex items-center space-x-1 font-semibold transition-colors focus:ring-1 focus:ring-red-400 p-1 rounded-lg"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear Timeline</span>
            </button>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
            {conversation.map((msg) => (
              <div
                key={msg.id}
                className={`p-4 rounded-2xl border shadow-xs flex flex-col justify-between space-y-2.5 hover:shadow-sm transition-all ${
                  msg.sender === 'speaker1'
                    ? 'bg-white border-slate-200/90'
                    : 'bg-emerald-50/40 border-emerald-200/80'
                }`}
              >
                <div className="flex items-center justify-between text-xs text-slate-500">
                  <div className="flex items-center space-x-2">
                    <span className={`font-bold px-2 py-0.5 rounded-md text-[11px] ${
                      msg.sender === 'speaker1'
                        ? 'bg-brand-50 text-brand-800 border border-brand-200'
                        : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                    }`}>
                      {msg.speakerLabel}: {msg.sourceLang.toUpperCase()} → {msg.targetLang.toUpperCase()}
                    </span>
                    <span className="text-[11px] text-slate-400">{msg.timestamp}</span>
                  </div>

                  <div className="flex items-center space-x-1.5">
                    <button
                      type="button"
                      onClick={() => handleCopy(msg.translatedText, msg.id)}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
                      title="Copy translation"
                    >
                      {copiedField === msg.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleReplayBubble(msg)}
                      className="flex items-center space-x-1 px-2.5 py-1 rounded-xl bg-brand-50 hover:bg-brand-100 text-brand-700 text-xs font-bold transition-colors"
                    >
                      <Volume2 className={`w-3.5 h-3.5 ${playingBubbleId === msg.id ? 'animate-pulse text-brand-800' : ''}`} />
                      <span>Replay</span>
                    </button>
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-xs text-slate-500 italic">"{msg.originalText}"</p>
                  <p className="text-base font-bold text-slate-900 font-indic">{msg.translatedText}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};

export default VoiceTranslator;
