import { Language, TranslationResponse, TranslationEngine, ISLSign, ISLRecognitionResult, HistoryItem, UserSettings } from '../types';

const API_BASE = '/api';

export async function fetchLanguages(): Promise<Language[]> {
  const res = await fetch(`${API_BASE}/languages`);
  if (!res.ok) throw new Error('Failed to fetch languages');
  const data = await res.json();
  return data.languages;
}

export async function translateText(
  sourceLanguage: string,
  targetLanguage: string,
  text: string,
  module: 'text' | 'voice' | 'isl' = 'text',
  engine: TranslationEngine = 'pralekha'
): Promise<TranslationResponse> {
  const res = await fetch(`${API_BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source_language: sourceLanguage,
      target_language: targetLanguage,
      text,
      module,
      engine
    })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Translation failed');
  }
  return res.json();
}

export async function fetchTranslationModels(): Promise<any> {
  const res = await fetch(`${API_BASE}/translation/models`);
  if (!res.ok) throw new Error('Failed to fetch translation models');
  return res.json();
}

export async function fetchNetworkStatus(): Promise<any> {
  const res = await fetch(`${API_BASE}/translation/network-status`);
  if (!res.ok) throw new Error('Failed to fetch network status');
  return res.json();
}

export async function detectLanguage(text: string): Promise<any> {
  const res = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) throw new Error('Language detection failed');
  return res.json();
}

export async function transliterateText(text: string, sourceScript: string, targetScript: string): Promise<string> {
  const res = await fetch(`${API_BASE}/transliterate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source_script: sourceScript, target_script: targetScript })
  });
  if (!res.ok) throw new Error('Transliteration failed');
  const data = await res.json();
  return data.transliterated;
}

export async function fetchISLDictionary(category?: string, search?: string): Promise<ISLSign[]> {
  const params = new URLSearchParams();
  if (category) params.append('category', category);
  if (search) params.append('search', search);
  
  const res = await fetch(`${API_BASE}/isl/dictionary?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch ISL dictionary');
  const data = await res.json();
  return data.signs;
}

export async function searchReverseSign(query: string): Promise<{ found: boolean; primary_sign?: ISLSign; results: ISLSign[] }> {
  const res = await fetch(`${API_BASE}/isl/reverse-sign?query=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error('Failed to query reverse sign');
  return res.json();
}

export async function recognizeLandmarks(
  leftHand: any[] | null,
  rightHand: any[] | null,
  pose: any[] | null,
  threshold?: number
): Promise<ISLRecognitionResult> {
  const res = await fetch(`${API_BASE}/isl/recognize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      left_hand_landmarks: leftHand,
      right_hand_landmarks: rightHand,
      pose_landmarks: pose,
      confidence_threshold: threshold
    })
  });
  if (!res.ok) throw new Error('ISL recognition error');
  return res.json();
}

export async function clearISLSentence(): Promise<any> {
  const res = await fetch(`${API_BASE}/isl/sentence/clear`, { method: 'POST' });
  return res.json();
}

export async function simulateSign(signId: string): Promise<ISLRecognitionResult> {
  const res = await fetch(`${API_BASE}/isl/simulate-sign/${encodeURIComponent(signId)}`, {
    method: 'POST'
  });
  if (!res.ok) throw new Error('Simulation failed');
  return res.json();
}

export async function undoISLToken(): Promise<any> {
  const res = await fetch(`${API_BASE}/isl/sentence/undo`, { method: 'POST' });
  return res.json();
}

export async function playSpeechAudio(text: string, langCode: string, speed: number = 1.0): Promise<void> {
  try {
    // 1. First attempt Backend neural gTTS streaming
    const res = await fetch(`${API_BASE}/speech/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: langCode, speed })
    });
    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.playbackRate = speed;
      await audio.play();
      return;
    }
  } catch (e) {
    console.warn('Backend audio synthesis error, using Web Speech API fallback', e);
  }

  // 2. Client-side Web Speech API Fallback
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speed;
    
    // Attempt to match Indic voice
    const voices = window.speechSynthesis.getVoices();
    const matchedVoice = voices.find(v => v.lang.startsWith(langCode) || v.lang.includes('IN'));
    if (matchedVoice) {
      utterance.voice = matchedVoice;
    }
    window.speechSynthesis.speak(utterance);
  }
}

export const streamSpeechAudio = playSpeechAudio;

export async function fetchHistory(module?: string, search?: string, favoritesOnly: boolean = false): Promise<HistoryItem[]> {
  const params = new URLSearchParams();
  if (module) params.append('module', module);
  if (search) params.append('search', search);
  if (favoritesOnly) params.append('favorites_only', 'true');

  const res = await fetch(`${API_BASE}/history?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to fetch history');
  const data = await res.json();
  return data.history;
}

export async function toggleHistoryFavorite(id: number): Promise<boolean> {
  const res = await fetch(`${API_BASE}/history/${id}/favorite`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to toggle favorite');
  const data = await res.json();
  return data.is_favorite;
}

export async function deleteHistoryItem(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete history item');
}

export async function clearAllHistory(): Promise<void> {
  const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear history');
}

export async function fetchSettings(): Promise<UserSettings> {
  const res = await fetch(`${API_BASE}/settings`);
  if (!res.ok) throw new Error('Failed to fetch settings');
  const data = await res.json();
  return data.settings;
}

export async function updateSettings(settings: Partial<UserSettings>): Promise<void> {
  const res = await fetch(`${API_BASE}/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ settings })
  });
  if (!res.ok) throw new Error('Failed to save settings');
}
