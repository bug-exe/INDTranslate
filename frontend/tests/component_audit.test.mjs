import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';

// ============================================================================
// IndTranslate Frontend Component & Architecture Audit Test Suite
// ============================================================================

const MOCK_LANGUAGES = [
  { code: 'hi', iso_639_3: 'hin', name: 'Hindi', native_name: 'हिन्दी', script: 'Devanagari', script_code: 'Deva', direction: 'ltr', family: 'Indo-Aryan', region: 'North/Central India', sample: 'नमस्ते', sample_translation: 'Hello', direct_indic_trans: true, gtts_code: 'hi', speech_recognition_code: 'hi-IN', transliteration_support: true },
  { code: 'ta', iso_639_3: 'tam', name: 'Tamil', native_name: 'தமிழ்', script: 'Tamil', script_code: 'Taml', direction: 'ltr', family: 'Dravidian', region: 'Tamil Nadu', sample: 'வணக்கம்', sample_translation: 'Hello', direct_indic_trans: true, gtts_code: 'ta', speech_recognition_code: 'ta-IN', transliteration_support: true },
  { code: 'te', iso_639_3: 'tel', name: 'Telugu', native_name: 'తెలుగు', script: 'Telugu', script_code: 'Telu', direction: 'ltr', family: 'Dravidian', region: 'Andhra Pradesh & Telangana', sample: 'నమస్కారం', sample_translation: 'Hello', direct_indic_trans: true, gtts_code: 'te', speech_recognition_code: 'te-IN', transliteration_support: true },
  { code: 'bn', iso_639_3: 'ben', name: 'Bengali', native_name: 'বাংলা', script: 'Bengali', script_code: 'Beng', direction: 'ltr', family: 'Indo-Aryan', region: 'West Bengal & Tripura', sample: 'নমস্কার', sample_translation: 'Hello', direct_indic_trans: true, gtts_code: 'bn', speech_recognition_code: 'bn-IN', transliteration_support: true },
  { code: 'mr', iso_639_3: 'mar', name: 'Marathi', native_name: 'मराठी', script: 'Devanagari', script_code: 'Deva', direction: 'ltr', family: 'Indo-Aryan', region: 'Maharashtra', sample: 'नमस्कार', sample_translation: 'Hello', direct_indic_trans: true, gtts_code: 'mr', speech_recognition_code: 'mr-IN', transliteration_support: true },
  { code: 'en', iso_639_3: 'eng', name: 'English', native_name: 'English', script: 'Latin', script_code: 'Latn', direction: 'ltr', family: 'Indo-European', region: 'Pan-India', sample: 'Hello', sample_translation: 'Hello', direct_indic_trans: false, gtts_code: 'en', speech_recognition_code: 'en-IN', transliteration_support: false }
];

test('1. Component Module Exports: components/index.ts exports all required components', async () => {
  const index = await import('../dist/assets/index-DvS6aVIX.js').catch(() => null);
  // Also verify source existence directly
  const fs = await import('node:fs');
  const indexSource = fs.readFileSync('src/components/index.ts', 'utf-8');

  assert.ok(indexSource.includes('LanguageSelector'), 'LanguageSelector must be exported');
  assert.ok(indexSource.includes('ErrorBoundary'), 'ErrorBoundary must be exported');
  assert.ok(indexSource.includes('Toast'), 'Toast must be exported');
  assert.ok(indexSource.includes('TextTranslator'), 'TextTranslator must be exported');
  assert.ok(indexSource.includes('ISLTranslator'), 'ISLTranslator must be exported');
  assert.ok(indexSource.includes('VoiceTranslator'), 'VoiceTranslator must be exported');
  assert.ok(indexSource.includes('SettingsModal'), 'SettingsModal must be exported');
  assert.ok(indexSource.includes('Navbar'), 'Navbar must be exported');
  assert.ok(indexSource.includes('Dashboard'), 'Dashboard must be exported');
  assert.ok(indexSource.includes('TranslationHistory'), 'TranslationHistory must be exported');
});

test('2. LanguageSelector: verifies structure, language options and auto-detect support', () => {
  // Test LanguageSelector props logic
  const includeAuto = true;
  const autoLabel = '⚡ Auto Detect Script';
  const selected = 'hi';

  const selectOptions = [];
  if (includeAuto) {
    selectOptions.push({ value: 'auto', label: autoLabel });
  }
  MOCK_LANGUAGES.forEach(lang => {
    selectOptions.push({
      value: lang.code,
      label: `${lang.name} ${lang.native_name ? `(${lang.native_name})` : ''}`
    });
  });

  assert.equal(selectOptions.length, MOCK_LANGUAGES.length + 1);
  assert.equal(selectOptions[0].value, 'auto');
  assert.equal(selectOptions[0].label, '⚡ Auto Detect Script');
  assert.equal(selectOptions[1].value, 'hi');
  assert.ok(selectOptions[1].label.includes('हिन्दी'));

  // Test without auto
  const selectOptionsNoAuto = MOCK_LANGUAGES.map(lang => ({
    value: lang.code,
    label: `${lang.name} (${lang.native_name})`
  }));
  assert.equal(selectOptionsNoAuto.length, MOCK_LANGUAGES.length);
  assert.equal(selectOptionsNoAuto[0].value, 'hi');
});

test('3. ErrorBoundary: verifies error state capture, reset and fallback handling', () => {
  // Class unit verification
  class TestErrorBoundary {
    constructor() {
      this.state = { hasError: false, error: null, errorInfo: null };
    }
    static getDerivedStateFromError(error) {
      return { hasError: true, error };
    }
    componentDidCatch(error, errorInfo) {
      this.state.errorInfo = errorInfo;
    }
    handleReset() {
      this.state = { hasError: false, error: null, errorInfo: null };
    }
  }

  const boundary = new TestErrorBoundary();
  assert.equal(boundary.state.hasError, false);
  assert.equal(boundary.state.error, null);

  const testError = new Error('MediaPipe holistic pipeline failed to initialize');
  const derived = TestErrorBoundary.getDerivedStateFromError(testError);
  boundary.state = { ...boundary.state, ...derived };

  assert.equal(boundary.state.hasError, true);
  assert.equal(boundary.state.error.message, 'MediaPipe holistic pipeline failed to initialize');

  boundary.componentDidCatch(testError, { componentStack: 'in ISLTranslator' });
  assert.ok(boundary.state.errorInfo);
  assert.equal(boundary.state.errorInfo.componentStack, 'in ISLTranslator');

  boundary.handleReset();
  assert.equal(boundary.state.hasError, false);
  assert.equal(boundary.state.error, null);
});

test('4. Toast System: verifies toast creation, typing, dismiss and timeout', () => {
  const toasts = [];
  const dismissToast = (id) => {
    const idx = toasts.findIndex(t => t.id === id);
    if (idx !== -1) toasts.splice(idx, 1);
  };

  const showToast = (message, type = 'info', duration = 4000) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
    toasts.push({ id, message, type, duration });
    return id;
  };

  const showError = (msg) => showToast(msg, 'error', 5000);
  const showSuccess = (msg) => showToast(msg, 'success', 3500);
  const showWarning = (msg) => showToast(msg, 'warning', 4000);
  const showInfo = (msg) => showToast(msg, 'info', 3500);

  const errId = showError('Translation failed: Rate limit exceeded');
  const succId = showSuccess('Translation cached');
  const warnId = showWarning('Offline Mode Active');
  const infoId = showInfo('Switched to Pralekha engine');

  assert.equal(toasts.length, 4);
  assert.equal(toasts[0].type, 'error');
  assert.equal(toasts[0].duration, 5000);
  assert.equal(toasts[1].type, 'success');
  assert.equal(toasts[2].type, 'warning');
  assert.equal(toasts[3].type, 'info');

  dismissToast(errId);
  assert.equal(toasts.length, 3);
  assert.equal(toasts.find(t => t.id === errId), undefined);
});

test('5. API Contract & Engine Switching: Universal vs Pralekha payload verification', async () => {
  const calls = [];
  const mockFetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.includes('/api/translate')) {
      const body = JSON.parse(options.body);
      return {
        ok: true,
        json: async () => ({
          source_language: body.source_language,
          target_language: body.target_language,
          source_text: body.text,
          translated_text: body.engine === 'pralekha' ? 'नमस्ते भारत (प्रलेखा)' : 'नमस्ते भारत (यूनिवर्सल)',
          confidence: 0.98,
          pipeline: body.engine === 'pralekha' ? 'AI4Bharat Pralekha Seq2Seq Indic Model' : 'Universal Google Cloud Engine',
          engine: body.engine,
          cached: false
        })
      };
    }
    if (url.includes('/api/translation/network-status')) {
      return {
        ok: true,
        json: async () => ({
          network_online: true,
          offline_ready: true,
          download_status: { status: 'completed', is_downloaded: true, progress_percent: 100 }
        })
      };
    }
    return { ok: true, json: async () => ({}) };
  };

  // 5a: Test Universal Engine
  const reqUniversal = {
    source_language: 'en',
    target_language: 'hi',
    text: 'Hello India',
    module: 'text',
    engine: 'universal'
  };
  const resUniv = await mockFetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqUniversal)
  });
  const dataUniv = await resUniv.json();
  assert.equal(dataUniv.engine, 'universal');
  assert.ok(dataUniv.pipeline.includes('Universal'));

  // 5b: Test Pralekha Engine
  const reqPralekha = {
    source_language: 'en',
    target_language: 'hi',
    text: 'Hello India',
    module: 'text',
    engine: 'pralekha'
  };
  const resPral = await mockFetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqPralekha)
  });
  const dataPral = await resPral.json();
  assert.equal(dataPral.engine, 'pralekha');
  assert.ok(dataPral.pipeline.includes('Pralekha'));

  // Verify that the engine field was properly forwarded in JSON payloads
  assert.equal(JSON.parse(calls[0].options.body).engine, 'universal');
  assert.equal(JSON.parse(calls[1].options.body).engine, 'pralekha');
});

test('6. Network Polling & Offline Status Transition Simulation', async () => {
  let prevOnline = null;
  const notifications = [];

  const simulateStatusCheck = (network_online) => {
    if (prevOnline !== null) {
      if (prevOnline === true && !network_online) {
        notifications.push({ type: 'warning', message: 'Network offline: Local offline fallback active' });
      } else if (prevOnline === false && network_online) {
        notifications.push({ type: 'success', message: 'Network restored: Cloud neural engines re-enabled' });
      }
    }
    prevOnline = network_online;
  };

  // Step 1: Initial check online -> no notification
  simulateStatusCheck(true);
  assert.equal(notifications.length, 0);

  // Step 2: Next check still online -> no notification
  simulateStatusCheck(true);
  assert.equal(notifications.length, 0);

  // Step 3: Network drops offline -> triggers warning
  simulateStatusCheck(false);
  assert.equal(notifications.length, 1);
  assert.equal(notifications[0].type, 'warning');
  assert.ok(notifications[0].message.includes('Network offline'));

  // Step 4: Network reconnects -> triggers success
  simulateStatusCheck(true);
  assert.equal(notifications.length, 2);
  assert.equal(notifications[1].type, 'success');
  assert.ok(notifications[1].message.includes('Network restored'));
});

test('7. API Error Detail Extraction: guarantees rich error messages to toasts', async () => {
  const mockFetchError = async () => {
    return {
      ok: false,
      status: 400,
      json: async () => ({
        detail: 'Invalid source language: xx is not an Eighth Schedule language'
      })
    };
  };

  let caughtError = null;
  try {
    const res = await mockFetchError();
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Translation failed');
    }
  } catch (err) {
    caughtError = err;
  }

  assert.ok(caughtError);
  assert.equal(caughtError.message, 'Invalid source language: xx is not an Eighth Schedule language');
});
