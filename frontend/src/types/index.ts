export interface Language {
  code: string;
  iso_639_3: string;
  name: string;
  native_name: string;
  script: string;
  script_code: string;
  direction: 'ltr' | 'rtl';
  family: string;
  region: string;
  sample: string;
  sample_translation: string;
  direct_indic_trans: boolean;
  gtts_code: string;
  speech_recognition_code: string;
  transliteration_support: boolean;
}

export type TranslationEngine = 'pralekha' | 'nllb' | 'nllb_base' | 'universal' | 'google';

export interface TranslationResponse {
  source_language: string;
  target_language: string;
  source_text: string;
  translated_text: string;
  confidence: number;
  pipeline: string;
  engine?: TranslationEngine | string;
  model_status?: string;
  cached: boolean;
  detected_lang?: {
    code: string;
    confidence: number;
    script: string;
    name: string;
  };
  history_id?: number;
}

export interface ISLSign {
  id: string;
  label: string;
  label_hi: string;
  category: string;
  handedness: string;
  description: string;
  hand_shape: string;
  movement: string;
  facial_expression: string;
  is_two_handed: boolean;
  primary_hand: string;
  islrtc_ref: string;
  avatar_frames: Array<{
    left_hand: number[];
    right_hand: number[];
    head: number[];
    pose: string;
  }>;
}

export interface ISLRecognitionResult {
  sign_recognized: string | null;
  sign_label?: string | null;
  sign_label_hi?: string | null;
  confidence: number;
  status: string;
  is_confirmed: boolean;
  sentence: string;
  sentence_tokens: string[];
  top_candidates: Array<{
    sign_id: string;
    label: string;
    confidence: number;
  }>;
}

export interface HistoryItem {
  id: number;
  created_at: string;
  source_language: string;
  target_language: string;
  source_text: string;
  translated_text: string;
  module: 'text' | 'voice' | 'isl';
  confidence: number;
  pipeline_notes: string;
  is_favorite: boolean;
}

export interface UserSettings {
  isl_confidence_threshold: string;
  default_source_lang: string;
  default_target_lang: string;
  camera_show_landmarks: string;
  speech_speed: string;
  auto_speech_playback: string;
  privacy_local_processing: string;
  transliteration_enabled: string;
}
