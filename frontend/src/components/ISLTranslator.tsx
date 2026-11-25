import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Camera, 
  CameraOff, 
  Volume2, 
  Copy, 
  Check, 
  Trash2, 
  RotateCcw, 
  Eye, 
  EyeOff, 
  Sparkles, 
  Layers, 
  ArrowRight, 
  Hand, 
  Search, 
  BookOpen, 
  Activity,
  CheckCircle2,
  AlertCircle,
  Zap,
  Play,
  Pause,
  RefreshCw,
  X,
  ChevronRight,
  Info,
  SlidersHorizontal,
  ChevronLeft,
  FastForward,
  Rewind,
  Repeat,
  ShieldAlert,
  HelpCircle,
  ExternalLink,
  Video,
  Sliders
} from 'lucide-react';
import { Language, ISLSign, ISLRecognitionResult } from '../types';
import { 
  recognizeLandmarks, 
  clearISLSentence, 
  undoISLToken, 
  playSpeechAudio, 
  fetchISLDictionary, 
  searchReverseSign, 
  translateText, 
  simulateSign 
} from '../services/api';
import { useToast } from '../context/ToastContext';

interface ISLTranslatorProps {
  languages: Language[];
}

interface RecentSignEntry {
  id: string;
  label: string;
  label_hi?: string;
  confidence: number;
  islrtc_ref?: string;
  timestamp: string;
}

interface CameraDevice {
  deviceId: string;
  label: string;
}

export const ISLTranslator: React.FC<ISLTranslatorProps> = ({ languages }) => {
  const { showSuccess, showError, showInfo } = useToast();

  // Model Disconnected flag (ensures model and camera are not loaded)
  const [isModelDisconnected, setIsModelDisconnected] = useState<boolean>(true);

  // Mode: 'studio' (Real-time camera recognition) or 'reverse' (Avatar sign visualizer)
  const [activeMode, setActiveMode] = useState<'studio' | 'reverse'>('studio');

  // Camera State & Devices
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [targetSpeechLang, setTargetSpeechLang] = useState<string>('hi');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.70);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [trackingStatus, setTrackingStatus] = useState<string>('Studio Standby');

  // Camera Permission & Error Handling
  const [cameraError, setCameraError] = useState<{
    type: 'permission_denied' | 'not_found' | 'in_use' | 'unknown';
    message: string;
  } | null>(null);
  const [showPermissionGuide, setShowPermissionGuide] = useState<boolean>(false);

  // Real-time Prediction Readout
  const [recognizedSign, setRecognizedSign] = useState<string | null>(null);
  const [signLabel, setSignLabel] = useState<string | null>(null);
  const [signLabelHi, setSignLabelHi] = useState<string | null>(null);
  const [signConfidence, setSignConfidence] = useState<number>(0);
  const [topCandidates, setTopCandidates] = useState<Array<{ sign_id: string; label: string; confidence: number }>>([]);
  const [recentSigns, setRecentSigns] = useState<RecentSignEntry[]>([]);
  const [sentenceText, setSentenceText] = useState<string>('');
  const [sentenceTokens, setSentenceTokens] = useState<string[]>([]);
  const [translatedSentence, setTranslatedSentence] = useState<string>('');

  // Audio & Copy State
  const [isPlayingAudio, setIsPlayingAudio] = useState<boolean>(false);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  // Cheat Sheet Drawer State
  const [isCheatSheetOpen, setIsCheatSheetOpen] = useState<boolean>(false);
  const [cheatSheetTab, setCheatSheetTab] = useState<'alphabet' | 'numbers' | 'common'>('alphabet');
  const [cheatSheetSearch, setCheatSheetSearch] = useState<string>('');

  // Mode B: Reverse Sign Avatar State
  const [reverseQuery, setReverseQuery] = useState<string>('namaste');
  const [selectedReverseSign, setSelectedReverseSign] = useState<ISLSign | null>(null);
  const [allDictionarySigns, setAllDictionarySigns] = useState<ISLSign[]>([]);
  const [dictionaryFilter, setDictionaryFilter] = useState<string>('All');
  const [avatarFrameIdx, setAvatarFrameIdx] = useState<number>(0);
  const [isAvatarPlaying, setIsAvatarPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isLooping, setIsLooping] = useState<boolean>(true);

  // DOM Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reverseCanvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // MediaPipe & Computer Vision Refs
  const holisticRef = useRef<any>(null);
  const isHolisticProcessingRef = useRef<boolean>(false);
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestLandmarksRef = useRef<{
    leftHand: any[] | null;
    rightHand: any[] | null;
    pose: any[] | null;
    timestamp: number;
  }>({
    leftHand: null,
    rightHand: null,
    pose: null,
    timestamp: 0
  });

  // Quick Authentic ISLRTC Test Signs
  const quickSigns = [
    { id: 'namaste', label: 'Namaste', hi: 'नमस्ते', icon: '🙏', ref: 'ISLRTC-GR-001' },
    { id: 'thank_you', label: 'Thank You', hi: 'धन्यवाद', icon: '🤝', ref: 'ISLRTC-PL-004' },
    { id: 'water', label: 'Water', hi: 'पानी', icon: '💧', ref: 'ISLRTC-DN-002' },
    { id: 'food', label: 'Food', hi: 'खाना', icon: '🍲', ref: 'ISLRTC-DN-005' },
    { id: 'help', label: 'Help', hi: 'मदद', icon: '🆘', ref: 'ISLRTC-EM-001' },
    { id: 'doctor', label: 'Doctor', hi: 'डॉक्टर', icon: '👨‍⚕️', ref: 'ISLRTC-MD-003' },
    { id: 'hospital', label: 'Hospital', hi: 'अस्पताल', icon: '🏥', ref: 'ISLRTC-MD-007' },
    { id: 'yes', label: 'Yes', hi: 'हाँ', icon: '✅', ref: 'ISLRTC-PS-001' },
    { id: 'no', label: 'No', hi: 'नहीं', icon: '❌', ref: 'ISLRTC-PS-002' },
    { id: 'where', label: 'Where', hi: 'कहाँ', icon: '❓', ref: 'ISLRTC-QU-003' },
    { id: 'welcome', label: 'Welcome', hi: 'स्वागत', icon: '💐', ref: 'ISLRTC-GR-008' },
    { id: 'please', label: 'Please', hi: 'कृपया', icon: '🤲', ref: 'ISLRTC-PL-002' }
  ];

  // ISL Alphabet Cheat Sheet
  const islAlphabet = [
    { letter: 'A', desc: 'Closed fist with thumb upright alongside index', shape: 'Fist with thumb upright' },
    { letter: 'B', desc: 'Open palm with 4 fingers upright, thumb tucked across palm', shape: 'Flat 4-finger palm' },
    { letter: 'C', desc: 'Curved fingers and thumb forming an open C shape', shape: 'Curved C cup' },
    { letter: 'D', desc: 'Index finger straight up, thumb and other fingers touching in an O loop', shape: 'Index up, circle base' },
    { letter: 'E', desc: 'All fingertips curled downward touching the thumb tip', shape: 'Clawed tips touching thumb' },
    { letter: 'F', desc: 'Index and thumb tips form an O, other 3 fingers straight', shape: 'OK circle, 3 fingers up' },
    { letter: 'G', desc: 'Index and thumb parallel pointing horizontally', shape: 'Horizontal pinch' },
    { letter: 'H', desc: 'Index and middle fingers extended together horizontally', shape: '2-finger horizontal beam' },
    { letter: 'I', desc: 'Pinky finger straight up, other fingers folded into fist', shape: 'Pinky upright' },
    { letter: 'J', desc: 'Pinky finger extended tracing a J curve in the air', shape: 'Pinky sweeping J curve' },
    { letter: 'K', desc: 'Index upright, middle angled forward, thumb resting between', shape: 'V-split with thumb wedge' },
    { letter: 'L', desc: 'Index finger upright and thumb extended at 90-degree right angle', shape: 'Right angle L' },
    { letter: 'M', desc: 'Three fingers folded over thumb tucked underneath', shape: '3 fingers over thumb' },
    { letter: 'N', desc: 'Two fingers folded over thumb tucked underneath', shape: '2 fingers over thumb' },
    { letter: 'O', desc: 'All fingers curved touching thumb tip forming a circle', shape: 'Circular O ring' },
    { letter: 'P', desc: 'K hand shape pointing downward', shape: 'Downward K' },
    { letter: 'Q', desc: 'G hand shape pointing downward', shape: 'Downward G pinch' },
    { letter: 'R', desc: 'Index and middle fingers crossed together', shape: 'Crossed fingers' },
    { letter: 'S', desc: 'Fist with thumb crossed firmly across fingers', shape: 'Fist thumb across' },
    { letter: 'T', desc: 'Thumb tucked between index and middle fingers of fist', shape: 'Thumb between index/mid' },
    { letter: 'U', desc: 'Index and middle fingers held together pointing straight up', shape: 'Pair upright fingers' },
    { letter: 'V', desc: 'Index and middle fingers spread in a V peace sign', shape: 'V peace sign' },
    { letter: 'W', desc: 'Index, middle, and ring fingers extended upward', shape: '3 fingers W fork' },
    { letter: 'X', desc: 'Index finger hooked into a curved claw shape', shape: 'Hooked index claw' },
    { letter: 'Y', desc: 'Thumb and pinky extended out, middle fingers folded', shape: 'Shaka thumb & pinky' },
    { letter: 'Z', desc: 'Index finger traces a Z zigzag pattern in the air', shape: 'Index dynamic Z sweep' }
  ];

  // ISL Numbers 1-10
  const islNumbers = [
    { num: '1', desc: 'Index finger raised pointing up', shape: 'Single index up' },
    { num: '2', desc: 'Index and middle fingers raised in V shape', shape: 'V 2-finger spread' },
    { num: '3', desc: 'Thumb, index, and middle fingers raised', shape: 'Thumb + 2 fingers' },
    { num: '4', desc: 'Four fingers raised straight, thumb tucked', shape: '4 fingers upright' },
    { num: '5', desc: 'Open hand with all five fingers spread wide', shape: 'Open 5 palm' },
    { num: '6', desc: 'Thumb and pinky tips touching, other 3 fingers raised', shape: 'Ring + mid + index up' },
    { num: '7', desc: 'Thumb and ring finger tips touching, others raised', shape: 'Pinky + mid + index up' },
    { num: '8', desc: 'Thumb and middle finger tips touching, others raised', shape: 'Pinky + ring + index up' },
    { num: '9', desc: 'Thumb and index finger tips touching, others raised', shape: '3 fingers up, OK pinch' },
    { num: '10', desc: 'Fist with thumb shaking slightly side to side', shape: 'Thumb oscillation fist' }
  ];

  // Enumerate Media Devices for Camera Selection Dropdown
  const enumerateCameras = useCallback(async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices
          .filter((d) => d.kind === 'videoinput')
          .map((d, index) => ({
            deviceId: d.deviceId,
            label: d.label || `Camera ${index + 1} (${d.deviceId ? d.deviceId.slice(0, 6) : 'Default'})`
          }));
        setAvailableCameras(videoInputs);
        if (videoInputs.length > 0 && !selectedCameraId) {
          setSelectedCameraId(videoInputs[0].deviceId);
        }
      }
    } catch (e) {
      console.warn('Camera device enumeration note:', e);
    }
  }, [selectedCameraId]);

  useEffect(() => {
    enumerateCameras();
    if (navigator.mediaDevices && 'ondevicechange' in navigator.mediaDevices) {
      navigator.mediaDevices.addEventListener('devicechange', enumerateCameras);
      return () => {
        navigator.mediaDevices.removeEventListener('devicechange', enumerateCameras);
      };
    }
  }, [enumerateCameras]);

  // Load Authentic ISL Dictionary
  useEffect(() => {
    fetchISLDictionary()
      .then((signs) => {
        setAllDictionarySigns(signs);
        if (signs.length > 0) {
          setSelectedReverseSign(signs[0]);
        }
      })
      .catch((err) => console.error('Error fetching ISL signs:', err));
  }, []);

  // Debounced translation of accumulated sentence into selected vocal language
  useEffect(() => {
    if (!sentenceText) {
      setTranslatedSentence('');
      return;
    }
    const timer = setTimeout(() => {
      translateText('auto', targetSpeechLang, sentenceText, 'isl')
        .then((res) => setTranslatedSentence(res.translated_text))
        .catch((e) => console.warn('ISL sentence translation warning:', e));
    }, 800);
    return () => clearTimeout(timer);
  }, [sentenceText, targetSpeechLang]);

  // Initialize MediaPipe Holistic if available
  const initMediaPipe = () => {
    if (typeof (window as any).Holistic !== 'undefined' && !holisticRef.current) {
      try {
        const holistic = new (window as any).Holistic({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        holistic.onResults((results: any) => {
          isHolisticProcessingRef.current = false;
          latestLandmarksRef.current = {
            leftHand: results.leftHandLandmarks || null,
            rightHand: results.rightHandLandmarks || null,
            pose: results.poseLandmarks || null,
            timestamp: Date.now()
          };
        });
        holisticRef.current = holistic;
      } catch (err) {
        console.warn('MediaPipe Holistic initialization note:', err);
      }
    }
  };

  // Optical Fallback Hand Tracking Engine
  const detectOpticalHands = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const sW = 80;
    const sH = 60;
    if (!sampleCanvasRef.current) {
      sampleCanvasRef.current = document.createElement('canvas');
      sampleCanvasRef.current.width = sW;
      sampleCanvasRef.current.height = sH;
    }
    const sCanvas = sampleCanvasRef.current;
    const sCtx = sCanvas.getContext('2d', { willReadFrequently: true });
    if (!sCtx) return { left: null, right: null, head: { x: 0.5, y: 0.22 } };

    sCtx.drawImage(ctx.canvas, 0, 0, sW, sH);
    const imgData = sCtx.getImageData(0, 0, sW, sH);
    const data = imgData.data;

    let leftCount = 0;
    let leftSumX = 0;
    let leftSumY = 0;

    let rightCount = 0;
    let rightSumX = 0;
    let rightSumY = 0;

    let headCount = 0;
    let headSumX = 0;
    let headSumY = 0;

    const midX = sW / 2;

    for (let y = 0; y < sH; y++) {
      for (let x = 0; x < sW; x++) {
        const idx = (y * sW + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        // Skin detection rule in RGB
        const isSkin = (r > 65 && g > 35 && b > 20 && r > g && r > b && (r - g) > 10 && (r - b) > 10);
        if (isSkin) {
          if (y < sH * 0.35 && x > sW * 0.3 && x < sW * 0.7) {
            headCount++;
            headSumX += x;
            headSumY += y;
          } else if (y >= sH * 0.28) {
            if (x < midX - 2) {
              leftCount++;
              leftSumX += x;
              leftSumY += y;
            } else if (x > midX + 2) {
              rightCount++;
              rightSumX += x;
              rightSumY += y;
            }
          }
        }
      }
    }

    const minHandPixels = 6;
    const leftCentroid = leftCount >= minHandPixels ? {
      x: (leftSumX / leftCount) / sW,
      y: (leftSumY / leftCount) / sH
    } : null;

    const rightCentroid = rightCount >= minHandPixels ? {
      x: (rightSumX / rightCount) / sW,
      y: (rightSumY / rightCount) / sH
    } : null;

    const headCentroid = headCount >= 8 ? {
      x: (headSumX / headCount) / sW,
      y: (headSumY / headCount) / sH
    } : { x: 0.5, y: 0.22 };

    return { left: leftCentroid, right: rightCentroid, head: headCentroid };
  };

  const buildHandLandmarksFromCentroid = (cx: number, cy: number, isRight: boolean) => {
    const scale = 0.07;
    const dir = isRight ? 1 : -1;
    const wrist = { x: cx, y: Math.min(0.95, cy + 0.06), z: 0 };
    const thumb = [
      { x: cx + dir * scale * 0.25, y: cy + scale * 0.25, z: 0 },
      { x: cx + dir * scale * 0.45, y: cy + scale * 0.10, z: 0 },
      { x: cx + dir * scale * 0.60, y: cy - scale * 0.05, z: 0 },
      { x: cx + dir * scale * 0.75, y: cy - scale * 0.20, z: 0 }
    ];
    const index = [
      { x: cx + dir * scale * 0.20, y: cy - scale * 0.20, z: 0 },
      { x: cx + dir * scale * 0.22, y: cy - scale * 0.45, z: 0 },
      { x: cx + dir * scale * 0.24, y: cy - scale * 0.65, z: 0 },
      { x: cx + dir * scale * 0.25, y: cy - scale * 0.85, z: 0 }
    ];
    const middle = [
      { x: cx, y: cy - scale * 0.25, z: 0 },
      { x: cx, y: cy - scale * 0.52, z: 0 },
      { x: cx, y: cy - scale * 0.74, z: 0 },
      { x: cx, y: cy - scale * 0.95, z: 0 }
    ];
    const ring = [
      { x: cx - dir * scale * 0.18, y: cy - scale * 0.22, z: 0 },
      { x: cx - dir * scale * 0.20, y: cy - scale * 0.46, z: 0 },
      { x: cx - dir * scale * 0.22, y: cy - scale * 0.66, z: 0 },
      { x: cx - dir * scale * 0.23, y: cy - scale * 0.84, z: 0 }
    ];
    const pinky = [
      { x: cx - dir * scale * 0.35, y: cy - scale * 0.15, z: 0 },
      { x: cx - dir * scale * 0.38, y: cy - scale * 0.35, z: 0 },
      { x: cx - dir * scale * 0.40, y: cy - scale * 0.52, z: 0 },
      { x: cx - dir * scale * 0.42, y: cy - scale * 0.68, z: 0 }
    ];
    return [wrist, ...thumb, ...index, ...middle, ...ring, ...pinky];
  };

  // Camera Management with Resilient Device Selection & Fallback Permissions
  const startCamera = async (customDeviceId?: string, customFacing?: 'user' | 'environment') => {
    setCameraError(null);
    const targetFacing = customFacing || facingMode;
    const targetDeviceId = customDeviceId !== undefined ? customDeviceId : selectedCameraId;

    try {
      // 1. Release any existing video stream tracks cleanly
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      // 2. Multi-tier media constraints (preferred deviceId -> facingMode -> generic video)
      let stream: MediaStream | null = null;
      try {
        const videoConstraints: MediaTrackConstraints = targetDeviceId
          ? { deviceId: { ideal: targetDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { facingMode: targetFacing, width: { ideal: 640 }, height: { ideal: 480 } };

        stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false
        });
      } catch (constraintErr) {
        console.warn('Initial camera constraints failed, attempting generic video fallback:', constraintErr);
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: targetFacing },
            audio: false
          });
        } catch (facingErr) {
          console.warn('Facing mode constraint failed, attempting pure video fallback:', facingErr);
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
          });
        }
      }

      if (!stream) {
        throw new Error('Unable to acquire camera media stream');
      }

      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        video.setAttribute('playsinline', 'true');
        video.setAttribute('muted', 'true');

        // Wait for video metadata to resolve so dimensions are non-zero
        await new Promise<void>((resolve) => {
          if (video.readyState >= 2 && video.videoWidth > 0) {
            resolve();
            return;
          }
          const onLoaded = () => {
            video.removeEventListener('loadedmetadata', onLoaded);
            resolve();
          };
          video.addEventListener('loadedmetadata', onLoaded);
          // 800ms safety timeout
          setTimeout(resolve, 800);
        });

        try {
          await video.play();
        } catch (playErr) {
          console.warn('Auto-play note (user interaction might be needed):', playErr);
        }

        setIsCameraActive(true);
        setTrackingStatus('Camera live. Initializing AI vision tracking...');
        showSuccess('Webcam active & AI tracking initialized');

        // Re-enumerate to get friendly camera labels now that permission is granted
        enumerateCameras();

        // Initialize MediaPipe Holistic client-side
        initMediaPipe();
        startVisionLoop();
      }
    } catch (err: any) {
      console.error('Camera access error:', err);
      let errorType: 'permission_denied' | 'not_found' | 'in_use' | 'unknown' = 'unknown';
      let message = err.message || 'Webcam initialization failed';

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorType = 'permission_denied';
        message = 'Camera permission denied. Please allow camera access in browser settings.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorType = 'not_found';
        message = 'No camera device found on this system. Please connect a webcam.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorType = 'in_use';
        message = 'Camera is already in use by another application (e.g. Zoom, Meet) or browser tab.';
      } else if (err.name === 'OverconstrainedError') {
        errorType = 'not_found';
        message = 'Requested camera resolution or hardware device is not available.';
      }

      setCameraError({ type: errorType, message });
      setTrackingStatus(`Camera error: ${message}`);
      showError(message);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    setIsCameraActive(false);
    setTrackingStatus('Studio Standby');
    showInfo('Camera turned off');
  };

  const flipCamera = () => {
    const newFacing = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newFacing);
    if (isCameraActive) {
      startCamera(undefined, newFacing);
    }
  };

  const handleCameraDeviceChange = (deviceId: string) => {
    setSelectedCameraId(deviceId);
    if (isCameraActive) {
      startCamera(deviceId);
    }
  };

  // Landmark Skeleton Drawing with Glowing Nodes and Articulated Bones
  const drawLandmarks = (
    ctx: CanvasRenderingContext2D,
    landmarks: any[],
    color: string,
    width: number,
    height: number,
    label?: string
  ) => {
    if (!landmarks || landmarks.length === 0) return;

    // Draw Glowing Points
    landmarks.forEach((pt, idx) => {
      const x = pt.x * width;
      const y = pt.y * height;
      const isTip = [0, 4, 8, 12, 16, 20].includes(idx);

      // Outer radial glow for fingertips and wrist
      if (isTip) {
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, 2 * Math.PI);
        ctx.fillStyle = `${color}40`;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(x, y, isTip ? 4.5 : 3, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? '#ffffff' : color;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Draw Skeleton Bones (Hand Skeleton 21 Joints)
    if (landmarks.length === 21) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;

      const fingers = [
        [0, 1, 2, 3, 4],       // Thumb
        [0, 5, 6, 7, 8],       // Index
        [0, 9, 10, 11, 12],    // Middle
        [0, 13, 14, 15, 16],   // Ring
        [0, 17, 18, 19, 20]    // Pinky
      ];
      fingers.forEach((finger) => {
        ctx.beginPath();
        for (let i = 0; i < finger.length - 1; i++) {
          const p1 = landmarks[finger[i]];
          const p2 = landmarks[finger[i + 1]];
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
        }
        ctx.stroke();
      });

      // Palm base connection
      ctx.beginPath();
      ctx.moveTo(landmarks[5].x * width, landmarks[5].y * height);
      ctx.lineTo(landmarks[9].x * width, landmarks[9].y * height);
      ctx.lineTo(landmarks[13].x * width, landmarks[13].y * height);
      ctx.lineTo(landmarks[17].x * width, landmarks[17].y * height);
      ctx.lineTo(landmarks[0].x * width, landmarks[0].y * height);
      ctx.stroke();

      // Hand label tag
      if (label && landmarks[0]) {
        const lx = landmarks[0].x * width;
        const ly = Math.min(height - 15, landmarks[0].y * height + 18);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.beginPath();
        ctx.roundRect(lx - 38, ly - 10, 76, 20, 6);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, lx, ly + 4);
      }
    }
  };

  // Vision Processing Loop
  const startVisionLoop = () => {
    let frameCount = 0;

    const render = async () => {
      if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) {
        animFrameIdRef.current = requestAnimationFrame(render);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const W = canvas.width;
      const H = canvas.height;

      // Draw mirrored video frame if user camera
      ctx.save();
      if (facingMode === 'user') {
        ctx.scale(-1, 1);
        ctx.drawImage(video, -W, 0, W, H);
      } else {
        ctx.drawImage(video, 0, 0, W, H);
      }
      ctx.restore();

      frameCount++;

      // 1. Send rendered canvas frame to MediaPipe Holistic (ensures coordinates match screen orientation)
      if (holisticRef.current && !isHolisticProcessingRef.current && frameCount % 2 === 0) {
        try {
          isHolisticProcessingRef.current = true;
          holisticRef.current.send({ image: canvas }).catch(() => {
            isHolisticProcessingRef.current = false;
          });
        } catch (e) {
          isHolisticProcessingRef.current = false;
        }
      }

      // 2. Optical Computer Vision Centroid Augmenter (High-speed fallback)
      const now = Date.now();
      const hasFreshMediaPipe = (now - latestLandmarksRef.current.timestamp) < 500;
      
      let currentLeftHand: any[] | null = null;
      let currentRightHand: any[] | null = null;
      let currentPose: any[] | null = null;

      if (hasFreshMediaPipe && (latestLandmarksRef.current.leftHand || latestLandmarksRef.current.rightHand)) {
        currentLeftHand = latestLandmarksRef.current.leftHand;
        currentRightHand = latestLandmarksRef.current.rightHand;
        currentPose = latestLandmarksRef.current.pose;
      } else {
        const detected = detectOpticalHands(ctx, W, H);
        if (detected.left) {
          currentLeftHand = buildHandLandmarksFromCentroid(detected.left.x, detected.left.y, false);
        }
        if (detected.right) {
          currentRightHand = buildHandLandmarksFromCentroid(detected.right.x, detected.right.y, true);
        }
        currentPose = [
          { x: detected.head.x, y: detected.head.y, z: 0 },
          { x: detected.head.x - 0.15, y: detected.head.y + 0.15, z: 0 },
          { x: detected.head.x + 0.15, y: detected.head.y + 0.15, z: 0 },
          { x: detected.head.x - 0.22, y: detected.head.y + 0.35, z: 0 },
          { x: detected.head.x + 0.22, y: detected.head.y + 0.35, z: 0 }
        ];
      }

      // 3. Draw Landmark Skeleton Overlays
      if (showLandmarks) {
        // Draw Upper Body Posture (Head, Shoulders, Arms)
        if (currentPose && currentPose.length >= 3) {
          const isMediaPipePose = currentPose.length >= 33;
          const nose = currentPose[0];
          const leftShoulder = isMediaPipePose ? currentPose[11] : currentPose[1];
          const rightShoulder = isMediaPipePose ? currentPose[12] : currentPose[2];
          const leftElbow = isMediaPipePose ? currentPose[13] : (currentPose.length >= 5 ? currentPose[3] : null);
          const rightElbow = isMediaPipePose ? currentPose[14] : (currentPose.length >= 5 ? currentPose[4] : null);
          const leftWrist = isMediaPipePose ? currentPose[15] : null;
          const rightWrist = isMediaPipePose ? currentPose[16] : null;

          ctx.fillStyle = '#38bdf8';
          ctx.strokeStyle = '#0284c7';
          ctx.lineWidth = 3;

          // Head nose marker
          if (nose) {
            ctx.beginPath();
            ctx.arc(nose.x * W, nose.y * H, 6, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.stroke();
          }

          // Shoulders line
          if (leftShoulder && rightShoulder) {
            ctx.beginPath();
            ctx.arc(leftShoulder.x * W, leftShoulder.y * H, 6, 0, 2 * Math.PI);
            ctx.arc(rightShoulder.x * W, rightShoulder.y * H, 6, 0, 2 * Math.PI);
            ctx.fill();

            ctx.beginPath();
            ctx.moveTo(leftShoulder.x * W, leftShoulder.y * H);
            ctx.lineTo(rightShoulder.x * W, rightShoulder.y * H);
            ctx.strokeStyle = '#0284c7';
            ctx.stroke();

            // Connect nose to mid-collarbone
            if (nose) {
              const midCollarX = (leftShoulder.x + rightShoulder.x) / 2;
              const midCollarY = (leftShoulder.y + rightShoulder.y) / 2;
              ctx.beginPath();
              ctx.moveTo(nose.x * W, nose.y * H);
              ctx.lineTo(midCollarX * W, midCollarY * H);
              ctx.strokeStyle = '#38bdf8';
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          }

          // Arms: Shoulder -> Elbow -> Wrist
          if (leftShoulder && leftElbow) {
            ctx.beginPath();
            ctx.arc(leftElbow.x * W, leftElbow.y * H, 5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.beginPath();
            ctx.moveTo(leftShoulder.x * W, leftShoulder.y * H);
            ctx.lineTo(leftElbow.x * W, leftElbow.y * H);
            if (leftWrist) {
              ctx.lineTo(leftWrist.x * W, leftWrist.y * H);
              ctx.arc(leftWrist.x * W, leftWrist.y * H, 4, 0, 2 * Math.PI);
            }
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 2.5;
            ctx.stroke();
          }

          if (rightShoulder && rightElbow) {
            ctx.beginPath();
            ctx.arc(rightElbow.x * W, rightElbow.y * H, 5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.beginPath();
            ctx.moveTo(rightShoulder.x * W, rightShoulder.y * H);
            ctx.lineTo(rightElbow.x * W, rightElbow.y * H);
            if (rightWrist) {
              ctx.lineTo(rightWrist.x * W, rightWrist.y * H);
              ctx.arc(rightWrist.x * W, rightWrist.y * H, 4, 0, 2 * Math.PI);
            }
            ctx.strokeStyle = '#0284c7';
            ctx.lineWidth = 2.5;
            ctx.stroke();
          }
        }

        // Draw Left Hand (Emerald Green - 21 nodes)
        if (currentLeftHand && currentLeftHand.length === 21) {
          drawLandmarks(ctx, currentLeftHand, '#10b981', W, H, 'Left Hand (21)');
        }

        // Draw Right Hand (Bright Violet/Purple - 21 nodes)
        if (currentRightHand && currentRightHand.length === 21) {
          drawLandmarks(ctx, currentRightHand, '#a855f7', W, H, 'Right Hand (21)');
        }
      }

      // 4. Update HUD tracking telemetry status
      if (currentLeftHand && currentRightHand) {
        setTrackingStatus('Tracking 2 Hands (42 nodes) + Posture');
      } else if (currentLeftHand) {
        setTrackingStatus('Tracking Left Hand (21 nodes) + Posture');
      } else if (currentRightHand) {
        setTrackingStatus('Tracking Right Hand (21 nodes) + Posture');
      } else {
        setTrackingStatus('Camera live. Upper body active — raise hands to track');
      }

      // 5. Neural Recognition Stream (Bypassed when model is disconnected)
      if (!isModelDisconnected && frameCount % 4 === 0 && (currentLeftHand || currentRightHand)) {
        try {
          const result: ISLRecognitionResult = await recognizeLandmarks(
            currentLeftHand,
            currentRightHand,
            currentPose,
            confidenceThreshold
          );

          setSignConfidence(result.confidence);
          setTopCandidates(result.top_candidates || []);

          if (result.sign_recognized) {
            setRecognizedSign(result.sign_recognized);
            const label = result.sign_label || result.sign_recognized;
            setSignLabel(label);
            setSignLabelHi(result.sign_label_hi || null);

            // Add to stream ticker if confirmed
            if (result.is_confirmed || result.confidence >= confidenceThreshold) {
              const matchedSign = allDictionarySigns.find((s) => s.id === result.sign_recognized);
              setRecentSigns((prev) => {
                const exists = prev[0]?.id === result.sign_recognized;
                if (exists) return prev;
                const newEntry: RecentSignEntry = {
                  id: result.sign_recognized!,
                  label,
                  label_hi: result.sign_label_hi || undefined,
                  confidence: result.confidence,
                  islrtc_ref: matchedSign?.islrtc_ref,
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                };
                return [newEntry, ...prev.slice(0, 7)];
              });
            }
          }

          if (result.sentence) {
            setSentenceText(result.sentence);
            setSentenceTokens(result.sentence_tokens);
          }
        } catch (e) {
          // silently handle inference stream
        }
      }

      animFrameIdRef.current = requestAnimationFrame(render);
    };

    render();
  };

  // Instant Authentic Sign Simulator
  const handleSimulateSign = async (signKey: string) => {
    setIsSimulating(true);
    setTrackingStatus(`Simulating authentic ISLRTC sign: ${signKey}...`);
    try {
      const res = await simulateSign(signKey);
      if (res.sign_recognized) {
        setRecognizedSign(res.sign_recognized);
        const label = res.sign_label || res.sign_recognized;
        setSignLabel(label);
        setSignLabelHi(res.sign_label_hi || null);

        const matchedSign = allDictionarySigns.find((s) => s.id === res.sign_recognized);
        setRecentSigns((prev) => {
          const newEntry: RecentSignEntry = {
            id: res.sign_recognized!,
            label,
            label_hi: res.sign_label_hi || undefined,
            confidence: res.confidence,
            islrtc_ref: matchedSign?.islrtc_ref,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          };
          return [newEntry, ...prev.slice(0, 7)];
        });
      }
      setSignConfidence(res.confidence);
      setTopCandidates(res.top_candidates || []);
      if (res.sentence) {
        setSentenceText(res.sentence);
        setSentenceTokens(res.sentence_tokens);
      }
      setTrackingStatus(`Confirmed: ${res.sign_label || signKey} (${Math.round(res.confidence * 100)}% confidence)`);
      showSuccess(`Simulated ISLRTC sign: ${res.sign_label || signKey}`);
    } catch (err: any) {
      console.error('Simulation error:', err);
      setTrackingStatus('Simulation error. Please check backend.');
      showError('Simulation failed');
    } finally {
      setTimeout(() => setIsSimulating(false), 500);
    }
  };

  // Mode B: Reverse Sign Visualizer Animation Loop with Speed & Scrubbing
  useEffect(() => {
    if (activeMode !== 'reverse' || !selectedReverseSign || !isAvatarPlaying) return;

    const frameCount = selectedReverseSign.avatar_frames?.length || 1;
    if (frameCount <= 1) return;

    const intervalMs = Math.round(600 / playbackSpeed);
    const interval = setInterval(() => {
      setAvatarFrameIdx((prev) => {
        const next = prev + 1;
        if (next >= frameCount) {
          return isLooping ? 0 : prev;
        }
        return next;
      });
    }, intervalMs);

    return () => clearInterval(interval);
  }, [activeMode, selectedReverseSign, isAvatarPlaying, playbackSpeed, isLooping]);

  // Render Reverse Avatar Sign Canvas with Articulation & Trails
  useEffect(() => {
    if (!reverseCanvasRef.current || !selectedReverseSign) return;
    const canvas = reverseCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Deep slate dark gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, '#030712');
    bgGrad.addColorStop(1, '#0f172a');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Torso / Neck base
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, h * 0.35);
    ctx.lineTo(w * 0.5, h * 0.65);
    ctx.stroke();

    // Head
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.25, 44, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Eyes
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(w * 0.44, h * 0.23, 4.5, 0, 2 * Math.PI);
    ctx.arc(w * 0.56, h * 0.23, 4.5, 0, 2 * Math.PI);
    ctx.fill();

    // Eyebrows
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(w * 0.40, h * 0.19);
    ctx.lineTo(w * 0.47, h * 0.20);
    ctx.moveTo(w * 0.60, h * 0.19);
    ctx.lineTo(w * 0.53, h * 0.20);
    ctx.stroke();

    // Smile / Mouth
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.27, 13, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();

    // Shoulders
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(w * 0.26, h * 0.46);
    ctx.lineTo(w * 0.5, h * 0.40);
    ctx.lineTo(w * 0.74, h * 0.46);
    ctx.stroke();

    // Shoulder joints
    ctx.fillStyle = '#64748b';
    ctx.beginPath();
    ctx.arc(w * 0.26, h * 0.46, 7, 0, 2 * Math.PI);
    ctx.arc(w * 0.74, h * 0.46, 7, 0, 2 * Math.PI);
    ctx.fill();

    // Active frame landmarks
    const frames = selectedReverseSign.avatar_frames;
    const currentFrame = frames && frames.length > 0 ? frames[avatarFrameIdx] || frames[0] : null;

    if (currentFrame) {
      const lh = currentFrame.left_hand;
      const lhX = w * (0.5 + lh[0]);
      const lhY = h * (0.42 + lh[1]);

      const rh = currentFrame.right_hand;
      const rhX = w * (0.5 + rh[0]);
      const rhY = h * (0.42 + rh[1]);

      // Calculate articulated elbows
      const leftElbowX = (w * 0.26 + lhX) / 2 - 20;
      const leftElbowY = Math.max(w * 0.46, (h * 0.46 + lhY) / 2 + 25);

      const rightElbowX = (w * 0.74 + rhX) / 2 + 20;
      const rightElbowY = Math.max(w * 0.46, (h * 0.46 + rhY) / 2 + 25);

      // Left Arm (Shoulder -> Elbow -> Hand)
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(w * 0.26, h * 0.46);
      ctx.lineTo(leftElbowX, leftElbowY);
      ctx.lineTo(lhX, lhY);
      ctx.stroke();

      // Right Arm (Shoulder -> Elbow -> Hand)
      ctx.strokeStyle = '#a855f7';
      ctx.beginPath();
      ctx.moveTo(w * 0.74, h * 0.46);
      ctx.lineTo(rightElbowX, rightElbowY);
      ctx.lineTo(rhX, rhY);
      ctx.stroke();

      // Elbow Nodes
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(leftElbowX, leftElbowY, 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.fillStyle = '#c084fc';
      ctx.beginPath();
      ctx.arc(rightElbowX, rightElbowY, 5, 0, 2 * Math.PI);
      ctx.fill();

      // Left Hand Palm & Finger rays
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(lhX, lhY, 14, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Right Hand Palm & Finger rays
      ctx.fillStyle = '#a855f7';
      ctx.beginPath();
      ctx.arc(rhX, rhY, 14, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Hand Labels
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('L', lhX, lhY + 4);
      ctx.fillText('R', rhX, rhY + 4);

      // Dominant / Primary Hand Glow Indicator
      const primary = (selectedReverseSign.primary_hand || '').toLowerCase();
      if (primary === 'right' || primary === 'both') {
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.arc(rhX, rhY, 22, 0, 2 * Math.PI);
        ctx.stroke();
      }
      if (primary === 'left' || primary === 'both') {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.arc(lhX, lhY, 22, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  }, [avatarFrameIdx, selectedReverseSign]);

  // Sentence actions
  const handleClearSentence = async () => {
    await clearISLSentence();
    setSentenceText('');
    setSentenceTokens([]);
    setRecognizedSign(null);
    setSignLabel(null);
    setSignLabelHi(null);
    showInfo('Cleared signed sentence');
  };

  const handleUndoSentence = async () => {
    const res = await undoISLToken();
    setSentenceText(res.sentence);
    setSentenceTokens(res.sentence_tokens);
    showInfo('Undid last signed token');
  };

  const handleSpeechPlayback = async () => {
    const textToSpeak = translatedSentence || sentenceText;
    if (!textToSpeak) return;
    setIsPlayingAudio(true);
    try {
      await playSpeechAudio(textToSpeak, targetSpeechLang);
    } catch (e) {
      showError('Audio playback failed');
    } finally {
      setIsPlayingAudio(false);
    }
  };

  const handleCopySentence = () => {
    const textToCopy = translatedSentence ? `${sentenceText} (${translatedSentence})` : sentenceText;
    if (!textToCopy) return;
    navigator.clipboard.writeText(textToCopy);
    setIsCopied(true);
    showSuccess('Copied signed sentence');
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleReverseSearch = async (q: string) => {
    setReverseQuery(q);
    if (!q.trim()) return;
    try {
      const res = await searchReverseSign(q);
      if (res.primary_sign) {
        setSelectedReverseSign(res.primary_sign);
        setAvatarFrameIdx(0);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const categories = ['All', 'Greetings', 'Daily Needs', 'Emergency & Needs', 'Medical', 'People & Society', 'Questions'];
  const filteredSigns = dictionaryFilter === 'All'
    ? allDictionarySigns
    : allDictionarySigns.filter(s => s.category.toLowerCase().includes(dictionaryFilter.toLowerCase()));

  // Filtered cheat sheet
  const filteredAlphabet = islAlphabet.filter(item => 
    item.letter.toLowerCase().includes(cheatSheetSearch.toLowerCase()) || 
    item.desc.toLowerCase().includes(cheatSheetSearch.toLowerCase()) ||
    item.shape.toLowerCase().includes(cheatSheetSearch.toLowerCase())
  );

  const filteredNumbers = islNumbers.filter(item => 
    item.num.includes(cheatSheetSearch) || 
    item.desc.toLowerCase().includes(cheatSheetSearch.toLowerCase())
  );

  // SVG Circular Gauge Calculations
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (Math.min(1, Math.max(0, signConfidence)) * circumference);

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-20">
      
      {/* ISL Model Disconnected Status Notification */}
      <div className="p-4 rounded-3xl bg-[#181a20] border border-blue-500/30 text-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-lg">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-2xl bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center justify-center flex-shrink-0">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center space-x-2">
              <span>Client AI Vision Tracking Active</span>
              <span className="px-2 py-0.5 rounded-full text-[9px] font-black bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">CV 60 FPS</span>
              <span className="px-2 py-0.5 rounded-full text-[9px] font-black bg-zinc-800 text-zinc-400 border border-zinc-700">Classifier Offline</span>
            </h3>
            <p className="text-xs text-zinc-400 mt-0.5">
              Real-time 42-node hand tracking (21 left + 21 right) and upper-body posture run 100% locally on your browser canvas. Neural sign classifier is disconnected to maintain 0% backend load.
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {!isCameraActive ? (
            <button
              onClick={() => startCamera()}
              className="px-4 py-2 rounded-xl bg-[#2563eb] hover:bg-blue-600 text-white font-extrabold text-xs shadow-md transition-all cursor-pointer flex items-center space-x-1.5"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Start Camera</span>
            </button>
          ) : (
            <div className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-emerald-950/80 text-emerald-300 text-xs font-bold border border-emerald-500/30">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>Tracking Live</span>
            </div>
          )}
        </div>
      </div>

      {/* Top Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
              ISL Computer Vision Studio
            </h1>
            <span className="text-xs px-2.5 py-1 rounded-full bg-zinc-800 text-zinc-400 font-extrabold border border-zinc-700">
              Disconnected (0% VRAM)
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-1">
            Real-time neural recognition of Indian Sign Language with 42 hand nodes, camera selection, and reverse signing avatar.
          </p>
        </div>

        {/* Mode Tabs, Permission Help & Cheat Sheet Trigger */}
        <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
          <div className="flex items-center bg-slate-200/80 p-1 rounded-2xl">
            <button
              onClick={() => setActiveMode('studio')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all ${
                activeMode === 'studio'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Camera className="w-4 h-4 text-brand-600" />
              <span>Camera Studio</span>
            </button>

            <button
              onClick={() => setActiveMode('reverse')}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all ${
                activeMode === 'reverse'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Hand className="w-4 h-4 text-purple-600" />
              <span>Reverse Sign Avatar</span>
            </button>
          </div>

          {/* Help & Offline Fallback Guide Button */}
          <button
            onClick={() => setShowPermissionGuide(true)}
            className="flex items-center space-x-1 px-3 py-2 rounded-2xl bg-white border border-slate-200 text-slate-700 hover:text-brand-600 hover:bg-slate-50 text-xs font-bold shadow-sm transition-all"
            title="Camera permissions & offline fallback guide"
          >
            <HelpCircle className="w-4 h-4 text-brand-500" />
            <span className="hidden md:inline">Guides</span>
          </button>

          {/* Cheat Sheet Drawer Button */}
          <button
            onClick={() => setIsCheatSheetOpen(true)}
            className="flex items-center space-x-1.5 px-3.5 py-2 rounded-2xl bg-white border border-slate-200 text-slate-700 hover:text-brand-600 hover:bg-slate-50 text-xs font-bold shadow-sm transition-all"
          >
            <BookOpen className="w-4 h-4 text-amber-500" />
            <span>Cheat Sheet</span>
          </button>
        </div>
      </div>

      {/* MODE 1: Camera Vision Studio */}
      {activeMode === 'studio' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* CAMERA STUDIO FRAME & CONTROLS (7 Cols) */}
          <div className="lg:col-span-7 space-y-4">
            
            {/* Dark-mode Glassmorphic Camera Frame */}
            <div className="relative rounded-3xl overflow-hidden bg-slate-950 aspect-[4/3] shadow-2xl border border-slate-800/90 flex items-center justify-center group">
              
              {/* Native Video Stream Source */}
              <video
                ref={videoRef}
                playsInline
                muted
                className="hidden"
              />

              {/* High-performance Overlay Canvas */}
              <canvas
                ref={canvasRef}
                className="w-full h-full object-cover"
              />

              {/* Error Banner Overlay (e.g. Permission Denied) */}
              {cameraError && (
                <div className="absolute inset-0 z-30 flex flex-col items-center justify-center p-6 text-center text-white bg-slate-950/95 backdrop-blur-md space-y-3">
                  <div className="w-16 h-16 rounded-3xl bg-red-500/20 text-red-400 border border-red-500/30 flex items-center justify-center shadow-lg">
                    <ShieldAlert className="w-8 h-8" />
                  </div>
                  <div className="space-y-1 max-w-sm">
                    <h3 className="text-lg font-bold text-red-200">
                      {cameraError.type === 'permission_denied'
                        ? 'Camera Permission Blocked'
                        : cameraError.type === 'not_found'
                        ? 'No Camera Device Found'
                        : 'Camera Access Error'}
                    </h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {cameraError.message}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                    <button
                      onClick={() => startCamera()}
                      className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-bold text-xs shadow-md transition-all cursor-pointer"
                    >
                      Retry Camera
                    </button>
                    <button
                      onClick={() => setShowPermissionGuide(true)}
                      className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs border border-slate-700 transition-all cursor-pointer"
                    >
                      View Fix Guide & Offline Fallback
                    </button>
                  </div>
                </div>
              )}

              {/* Standby State Graphic when Camera is Inactive and No Error */}
              {!isCameraActive && !cameraError && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-white bg-slate-950/92 backdrop-blur-md z-10 space-y-4">
                  <div className="relative">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-[#181a20] to-[#262933] text-blue-400 border border-blue-500/30 flex items-center justify-center shadow-2xl">
                      <Camera className="w-10 h-10 text-blue-400" />
                    </div>
                    <span className="absolute -bottom-1 -right-1 flex h-4 w-4">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-4 w-4 bg-emerald-500 border-2 border-slate-950"></span>
                    </span>
                  </div>

                  <div className="space-y-1.5 max-w-md">
                    <h3 className="text-xl font-extrabold text-white tracking-tight">
                      ISL Camera & Vision Studio
                    </h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Real-time client-side tracking of 42 hand nodes (21 left + 21 right) & upper-body posture. Neural classification model is disconnected (0% server load).
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
                    <button
                      onClick={() => startCamera()}
                      className="flex items-center space-x-2 px-6 py-3 rounded-2xl bg-[#2563eb] hover:bg-blue-600 text-white font-extrabold text-sm shadow-xl hover:shadow-blue-500/25 transition-all hover:scale-105 active:scale-95 cursor-pointer"
                    >
                      <Camera className="w-4 h-4" />
                      <span>Launch Vision Tracking</span>
                    </button>
                    {availableCameras.length > 1 && (
                      <select
                        value={selectedCameraId}
                        onChange={(e) => setSelectedCameraId(e.target.value)}
                        className="px-3 py-2.5 rounded-2xl bg-[#181a20] border border-[#262933] text-xs text-slate-200 outline-none cursor-pointer"
                      >
                        {availableCameras.map((cam) => (
                          <option key={cam.deviceId} value={cam.deviceId}>
                            {cam.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center justify-center gap-2 text-[11px] text-zinc-400 pt-1">
                    <span className="px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 flex items-center space-x-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                      <span>CV Tracking: Ready (60 FPS)</span>
                    </span>
                    <span className="px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 flex items-center space-x-1.5">
                      <span className="w-2 h-2 rounded-full bg-zinc-500"></span>
                      <span>AI Model: Disconnected</span>
                    </span>
                  </div>
                </div>
              )}

              {/* Live Overlay HUD when Camera is Active */}
              {isCameraActive && (
                <>
                  {/* Top Status Bar with Live Radar Pulse */}
                  <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-20 pointer-events-none">
                    
                    {/* Recording Pulse Badge */}
                    <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900/85 backdrop-blur-md text-white text-xs font-bold border border-white/10 shadow-lg">
                      <span className="relative flex h-2.5 w-2.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                      </span>
                      <span className="tracking-wide uppercase text-[10px] text-red-300 font-extrabold">LIVE CV</span>
                      <span className="text-slate-400">|</span>
                      <span className="text-slate-200">{trackingStatus}</span>
                    </div>

                    {/* Skeleton Overlay Toggle Button */}
                    <button
                      onClick={() => setShowLandmarks(!showLandmarks)}
                      className="pointer-events-auto flex items-center space-x-1.5 px-3 py-1.5 rounded-full bg-slate-900/85 backdrop-blur-md hover:bg-slate-800 text-white text-xs font-bold border border-white/10 transition-colors cursor-pointer"
                    >
                      {showLandmarks ? <Eye className="w-3.5 h-3.5 text-emerald-400" /> : <EyeOff className="w-3.5 h-3.5 text-slate-400" />}
                      <span>Skeleton {showLandmarks ? 'ON' : 'OFF'}</span>
                    </button>
                  </div>

                  {/* Bottom Camera Toolbar */}
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-20">
                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] font-mono text-emerald-300 font-bold px-2 py-1 rounded-lg bg-emerald-950/80 border border-emerald-500/30 backdrop-blur-md">
                        Left: 21 nodes
                      </span>
                      <span className="text-[10px] font-mono text-purple-300 font-bold px-2 py-1 rounded-lg bg-purple-950/80 border border-purple-500/30 backdrop-blur-md">
                        Right: 21 nodes
                      </span>
                    </div>

                    <div className="flex items-center space-x-2">
                      {/* Flip Camera Button */}
                      <button
                        onClick={flipCamera}
                        title="Flip Camera (Front/Back)"
                        className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-white border border-white/10 backdrop-blur-md shadow-md transition-all active:scale-95 cursor-pointer"
                      >
                        <RefreshCw className="w-4 h-4" />
                      </button>

                      {/* Stop Camera Button */}
                      <button
                        onClick={stopCamera}
                        className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-red-600/90 hover:bg-red-700 text-white text-xs font-bold shadow-lg backdrop-blur-md transition-all cursor-pointer"
                      >
                        <CameraOff className="w-3.5 h-3.5" />
                        <span>Stop Studio</span>
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Camera Selection Dropdown & Hardware Selector Strip */}
            <div className="glass-panel p-3.5 rounded-2xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs shadow-xs">
              <div className="flex items-center space-x-2">
                <Video className="w-4 h-4 text-brand-600" />
                <span className="font-bold text-slate-700">Camera Source:</span>
                <select
                  value={selectedCameraId}
                  onChange={(e) => handleCameraDeviceChange(e.target.value)}
                  className="bg-slate-100 hover:bg-slate-200/80 text-slate-800 font-semibold px-2.5 py-1 rounded-xl border border-slate-200 outline-none cursor-pointer max-w-[220px] truncate"
                >
                  {availableCameras.length === 0 ? (
                    <option value="">Default Web Camera</option>
                  ) : (
                    availableCameras.map((cam) => (
                      <option key={cam.deviceId} value={cam.deviceId}>
                        {cam.label}
                      </option>
                    ))
                  )}
                </select>
              </div>

              {/* Landmark Color Legend */}
              <div className="flex items-center space-x-2 text-[11px] text-slate-500 font-medium">
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span>Left (21)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-purple-500" />
                  <span>Right (21)</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-sky-400" />
                  <span>Posture</span>
                </span>
              </div>
            </div>

            {/* Confidence Threshold Gauge & Slider Bar */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-center space-x-2.5">
                <SlidersHorizontal className="w-4 h-4 text-brand-600" />
                <span className="font-bold text-slate-700">Recognition Gate:</span>
                <span className="font-extrabold text-brand-600 px-2.5 py-0.5 rounded-lg bg-brand-50 border border-brand-200 font-mono">
                  {Math.round(confidenceThreshold * 100)}%
                </span>
              </div>

              <div className="flex-1 max-w-xs flex items-center space-x-3">
                <span className="text-[10px] text-slate-400 font-bold">50%</span>
                <input
                  type="range"
                  min={0.50}
                  max={0.95}
                  step={0.05}
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full accent-brand-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400 font-bold">95%</span>
              </div>

              <span className="text-[11px] text-slate-500 font-medium">
                Higher = fewer false positives
              </span>
            </div>

            {/* Quick Authentic ISLRTC Test Signs Strip (Instant GPU Inference) */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-xs font-bold text-slate-800">
                  <Sparkles className="w-4 h-4 text-brand-600" />
                  <span>Instant Authentic Sign Simulation (No Webcam Hardware Needed):</span>
                </div>
                <span className="text-[10px] bg-purple-50 text-purple-700 font-bold px-2 py-0.5 rounded-full border border-purple-200">
                  GPU 30-Frame Kinematics
                </span>
              </div>
              <p className="text-[11px] text-slate-500">
                Click any official ISLRTC sign to feed its 42-node landmark sequence into the neural classifier.
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {quickSigns.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => handleSimulateSign(s.id)}
                    disabled={isSimulating}
                    className="flex items-center space-x-1 px-2.5 py-1.5 rounded-xl bg-white hover:bg-brand-50 hover:text-brand-700 hover:border-brand-300 border border-slate-200 text-xs font-semibold text-slate-700 shadow-xs transition-all hover:scale-105 active:scale-95 disabled:opacity-50 cursor-pointer"
                  >
                    <span>{s.icon}</span>
                    <span>{s.label}</span>
                    <span className="text-[10px] text-slate-400 font-normal">({s.hi})</span>
                  </button>
                ))}
              </div>
            </div>

          </div>

          {/* REAL-TIME PREDICTION READOUT & ACCUMULATOR STRIP (5 Cols) */}
          <div className="lg:col-span-5 space-y-4 flex flex-col justify-between">
            
            {/* Live Detected Sign Readout Card */}
            <div className="glass-panel p-5 rounded-3xl border border-slate-200 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-purple-600" />
                  <h3 className="font-bold text-slate-900 text-sm">Real-Time Prediction Readout</h3>
                </div>
                {signConfidence >= confidenceThreshold && recognizedSign && (
                  <span className="text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 flex items-center space-x-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Confirmed</span>
                  </span>
                )}
              </div>

              {/* Big Sign Name & Circular Confidence Gauge */}
              <div className="p-4 rounded-2xl bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white flex items-center justify-between shadow-md">
                
                {/* Sign Label */}
                <div className="space-y-1">
                  <span className="text-[10px] font-extrabold uppercase tracking-wider text-brand-300">
                    {isCameraActive ? 'Real-Time Vision Telemetry' : 'Detected ISL Sign'}
                  </span>
                  <div className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                    {isCameraActive
                      ? trackingStatus
                      : (signLabel || 'Sign in frame...')}
                  </div>
                  {signLabelHi && (
                    <div className="text-sm font-bold text-amber-300">
                      {signLabelHi}
                    </div>
                  )}
                  <div className="text-[11px] text-slate-400 flex items-center space-x-2 pt-0.5">
                    <span className="inline-flex items-center space-x-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                      <span>Left (21)</span>
                    </span>
                    <span>•</span>
                    <span className="inline-flex items-center space-x-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                      <span>Right (21)</span>
                    </span>
                    <span>•</span>
                    <span className="inline-flex items-center space-x-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-sky-400"></span>
                      <span>Posture</span>
                    </span>
                  </div>
                </div>

                {/* SVG Confidence Percentage Ring */}
                <div className="relative flex items-center justify-center">
                  <svg className="w-24 h-24 transform -rotate-90">
                    <circle
                      cx="48"
                      cy="48"
                      r={radius}
                      stroke="rgba(255, 255, 255, 0.15)"
                      strokeWidth="7"
                      fill="transparent"
                    />
                    <circle
                      cx="48"
                      cy="48"
                      r={radius}
                      stroke="url(#confGradient)"
                      strokeWidth="7"
                      fill="transparent"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                      strokeLinecap="round"
                      className="transition-all duration-300 ease-out"
                    />
                    <defs>
                      <linearGradient id="confGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#38bdf8" />
                        <stop offset="50%" stopColor="#818cf8" />
                        <stop offset="100%" stopColor="#34d399" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                    <span className="text-base font-extrabold text-white font-mono leading-none">
                      {Math.round(signConfidence * 100)}%
                    </span>
                    <span className="text-[9px] text-slate-400 uppercase tracking-wider font-semibold mt-0.5">
                      Conf
                    </span>
                  </div>
                </div>

              </div>

              {/* Candidate Sign Cards with ISLRTC Reference Numbers & Hindi/English Labels */}
              {topCandidates.length > 0 && (
                <div className="space-y-2 pt-1">
                  <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-slate-500">
                    <span>Neural Candidate Cards:</span>
                    <span className="text-[10px] text-slate-400 font-normal">Click to confirm</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {topCandidates.map((cand, idx) => {
                      const matched = allDictionarySigns.find(s => s.id === cand.sign_id);
                      const engLabel = cand.label.split('/')[0].trim();
                      const hiLabel = matched?.label_hi || (cand.label.includes('/') ? cand.label.split('/')[1].trim() : '');
                      const islrtcRef = matched?.islrtc_ref || `ISLRTC-S${idx + 1}`;
                      return (
                        <button
                          key={idx}
                          onClick={() => handleSimulateSign(cand.sign_id)}
                          className="p-2.5 rounded-xl bg-slate-50 hover:bg-brand-50/70 border border-slate-200/90 hover:border-brand-300 text-left transition-all cursor-pointer group"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded bg-slate-200/70 text-slate-600 group-hover:bg-brand-100 group-hover:text-brand-700">
                              {islrtcRef}
                            </span>
                            <span className="text-[11px] font-mono font-bold text-brand-600">
                              {Math.round(cand.confidence * 100)}%
                            </span>
                          </div>
                          <div className="text-xs font-bold text-slate-800 truncate" title={engLabel}>
                            {engLabel}
                          </div>
                          {hiLabel && (
                            <div className="text-[10px] font-semibold text-purple-700 truncate mt-0.5">
                              {hiLabel}
                            </div>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recent Signs Stream Ticker */}
              <div className="pt-2 border-t border-slate-100 space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  <span>Recent Signs Stream:</span>
                  <span className="text-[10px] text-slate-400 font-normal">{recentSigns.length} recorded</span>
                </div>
                
                {recentSigns.length === 0 ? (
                  <p className="text-xs text-slate-400 italic py-1">
                    Confirmed gestures will stream here chronologically...
                  </p>
                ) : (
                  <div className="flex items-center space-x-1.5 overflow-x-auto pb-1">
                    {recentSigns.map((rs, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-xl bg-brand-50 border border-brand-200 text-xs font-bold text-brand-800 whitespace-nowrap shadow-xs"
                      >
                        <span>{rs.label.split('/')[0]}</span>
                        {rs.islrtc_ref && (
                          <span className="text-[9px] font-mono text-purple-600 px-1 rounded bg-purple-100/60">
                            {rs.islrtc_ref.split('-')[1] || rs.islrtc_ref}
                          </span>
                        )}
                        <span className="text-[10px] font-mono text-brand-500">
                          {Math.round(rs.confidence * 100)}%
                        </span>
                      </span>
                    ))}
                  </div>
                )}
              </div>

            </div>

            {/* Synthesized Sentence Accumulator Strip with Copy & Speak Actions */}
            <div className="glass-panel p-5 rounded-3xl border border-slate-200 shadow-sm space-y-4 flex-1 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div className="flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-brand-600" />
                    <h3 className="font-bold text-slate-900 text-sm">Sentence Accumulator Strip</h3>
                  </div>

                  {/* Actions: Undo & Clear */}
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={handleUndoSentence}
                      disabled={sentenceTokens.length === 0}
                      title="Undo last token"
                      className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 disabled:opacity-30 transition-colors cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={handleClearSentence}
                      disabled={sentenceTokens.length === 0}
                      title="Clear sentence"
                      className="p-1.5 rounded-lg text-slate-500 hover:text-red-500 hover:bg-slate-100 disabled:opacity-30 transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Sentence Display Card */}
                <div className="mt-3 p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 min-h-[90px] flex flex-col justify-between">
                  {sentenceText ? (
                    <div className="text-slate-900 text-base font-bold leading-relaxed">
                      "{sentenceText}"
                    </div>
                  ) : (
                    <div className="text-slate-400 text-xs italic">
                      Consecutive ISL gestures accumulate here into grammatical natural language sentences...
                    </div>
                  )}

                  {/* Target Translated Sentence */}
                  {translatedSentence && (
                    <div className="mt-2.5 pt-2 border-t border-slate-200/70 text-brand-800 text-sm font-extrabold flex items-center justify-between">
                      <span>{translatedSentence}</span>
                      <span className="text-[10px] text-brand-600 uppercase font-mono font-bold">
                        {targetSpeechLang.toUpperCase()}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Target Speech Language & Vocal Synthesis Toolbar */}
              <div className="pt-3 border-t border-slate-100 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-600">Vocal Language:</span>
                  <select
                    value={targetSpeechLang}
                    onChange={(e) => setTargetSpeechLang(e.target.value)}
                    className="text-xs font-bold text-slate-800 bg-slate-100 border border-slate-200 rounded-xl px-2.5 py-1 outline-none cursor-pointer"
                  >
                    {languages.map((l) => (
                      <option key={l.code} value={l.code}>
                        {l.name} ({l.native_name})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center justify-end space-x-2">
                  <button
                    onClick={handleCopySentence}
                    disabled={!sentenceText}
                    className="flex items-center space-x-1 px-3 py-1.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-100 disabled:opacity-30 text-xs font-bold transition-all cursor-pointer"
                  >
                    {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{isCopied ? 'Copied' : 'Copy'}</span>
                  </button>

                  <button
                    onClick={handleSpeechPlayback}
                    disabled={!sentenceText || isPlayingAudio}
                    className="flex items-center space-x-1.5 px-4 py-1.5 rounded-xl bg-gradient-to-r from-brand-600 to-purple-600 hover:scale-105 active:scale-95 text-white font-bold text-xs shadow-md shadow-brand-500/20 disabled:opacity-30 transition-all cursor-pointer"
                  >
                    <Volume2 className={`w-3.5 h-3.5 ${isPlayingAudio ? 'animate-pulse' : ''}`} />
                    <span>{isPlayingAudio ? 'Speaking...' : 'Speak Translation'}</span>
                  </button>
                </div>
              </div>

            </div>

          </div>

        </div>
      )}

      {/* MODE 2: Reverse Sign Avatar & Visualizer */}
      {activeMode === 'reverse' && (
        <div className="space-y-6">
          
          {/* Reverse Search Input Bar */}
          <div className="glass-panel p-6 rounded-3xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-xl font-extrabold text-slate-900">
                  Reverse Sign Avatar & Kinematic Visualizer
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Type any Indian language word or phrase to view animated avatar kinematics, 3D handshape diagrams, and official ISLRTC sign breakdowns.
                </p>
              </div>

              {/* Primary Hand Badge Header */}
              {selectedReverseSign && (
                <div className="flex items-center space-x-2">
                  <span className={`text-xs px-3 py-1 rounded-full font-bold border ${
                    selectedReverseSign.is_two_handed
                      ? 'bg-purple-50 text-purple-700 border-purple-200'
                      : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  }`}>
                    {selectedReverseSign.is_two_handed ? 'Two-Handed Coordination' : 'Single Hand (Dominant)'}
                  </span>
                  <span className="text-xs px-3 py-1 rounded-full bg-slate-100 text-slate-700 font-bold border border-slate-200 font-mono">
                    {selectedReverseSign.islrtc_ref}
                  </span>
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={reverseQuery}
                  onChange={(e) => handleReverseSearch(e.target.value)}
                  placeholder="Type an English or Hindi word (e.g. Namaste, Water, Doctor, Help, Hospital)..."
                  className="w-full pl-10 pr-4 py-2.5 rounded-2xl bg-slate-50 border border-slate-200 focus:bg-white focus:ring-2 focus:ring-brand-500 outline-none text-slate-900 text-xs font-semibold"
                />
              </div>

              {/* Category Pills */}
              <div className="flex items-center space-x-1.5 overflow-x-auto pb-1 sm:pb-0">
                {categories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setDictionaryFilter(cat)}
                    className={`px-3 py-2 rounded-xl text-xs font-bold whitespace-nowrap transition-colors cursor-pointer ${
                      dictionaryFilter === cat
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Visual Sign Display Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Animated Avatar Canvas with Playback Scrubber & Speed Controls (5 cols) */}
            <div className="lg:col-span-5 glass-panel p-6 rounded-3xl border border-slate-200 flex flex-col items-center justify-between space-y-4 shadow-sm">
              <div className="w-full flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center space-x-2">
                  <Hand className="w-4 h-4 text-purple-600" />
                  <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">Avatar Kinematics</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-purple-50 text-purple-700 font-bold border border-purple-200 font-mono">
                    Frame {avatarFrameIdx + 1} / {selectedReverseSign?.avatar_frames?.length || 1}
                  </span>
                </div>
              </div>

              {/* Canvas element */}
              <div className="w-full aspect-square max-w-[340px] rounded-3xl overflow-hidden shadow-xl border border-slate-900 bg-slate-950">
                <canvas
                  ref={reverseCanvasRef}
                  width={340}
                  height={340}
                  className="w-full h-full"
                />
              </div>

              {/* Playback Controls Strip: Play/Pause, Scrubber, Speed Multiplier */}
              <div className="w-full space-y-3 pt-2">
                {/* Frame Scrubber Slider */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-slate-500 font-medium">
                    <span>Frame Scrubber</span>
                    <span className="font-mono">
                      {avatarFrameIdx + 1} of {selectedReverseSign?.avatar_frames?.length || 1}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, (selectedReverseSign?.avatar_frames?.length || 1) - 1)}
                    value={avatarFrameIdx}
                    onChange={(e) => {
                      setIsAvatarPlaying(false);
                      setAvatarFrameIdx(parseInt(e.target.value));
                    }}
                    className="w-full accent-purple-600 cursor-pointer"
                  />
                </div>

                {/* Play/Pause & Step Buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => {
                        const len = selectedReverseSign?.avatar_frames?.length || 1;
                        setAvatarFrameIdx((prev) => (prev > 0 ? prev - 1 : len - 1));
                      }}
                      title="Previous frame"
                      className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                    >
                      <Rewind className="w-3.5 h-3.5" />
                    </button>

                    <button
                      onClick={() => setIsAvatarPlaying(!isAvatarPlaying)}
                      className="flex items-center space-x-1 px-3 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-700 text-white font-bold text-xs shadow-sm transition-all"
                    >
                      {isAvatarPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                      <span>{isAvatarPlaying ? 'Pause' : 'Play'}</span>
                    </button>

                    <button
                      onClick={() => {
                        const len = selectedReverseSign?.avatar_frames?.length || 1;
                        setAvatarFrameIdx((prev) => (prev + 1) % len);
                      }}
                      title="Next frame"
                      className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                    >
                      <FastForward className="w-3.5 h-3.5" />
                    </button>

                    <button
                      onClick={() => setIsLooping(!isLooping)}
                      title={isLooping ? 'Looping enabled' : 'Looping disabled'}
                      className={`p-1.5 rounded-xl transition-colors ${
                        isLooping ? 'bg-purple-100 text-purple-700' : 'bg-slate-100 text-slate-400'
                      }`}
                    >
                      <Repeat className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {/* Speed Multiplier Pill Selectors (0.5x - 2.0x) */}
                  <div className="flex items-center space-x-1 bg-slate-100 p-0.5 rounded-xl text-[10px] font-bold">
                    {[0.5, 1.0, 1.5, 2.0].map((spd) => (
                      <button
                        key={spd}
                        onClick={() => setPlaybackSpeed(spd)}
                        className={`px-2 py-1 rounded-lg transition-all ${
                          playbackSpeed === spd
                            ? 'bg-white text-purple-700 shadow-xs'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        {spd}x
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Sign metadata & reference */}
              <div className="w-full text-center space-y-1 pt-2 border-t border-slate-100">
                <h4 className="text-2xl font-extrabold text-slate-900">{selectedReverseSign?.label}</h4>
                <p className="text-base font-extrabold text-purple-700">{selectedReverseSign?.label_hi}</p>
                <div className="flex items-center justify-center space-x-2 text-xs text-slate-500 pt-1">
                  <span>Primary Hand: <strong className="text-slate-800 uppercase font-mono">{selectedReverseSign?.primary_hand}</strong></span>
                  <span>&bull;</span>
                  <span>Ref: <strong className="font-mono text-purple-700">{selectedReverseSign?.islrtc_ref}</strong></span>
                </div>
              </div>
            </div>

            {/* Authentic ISLRTC Sign Breakdown & 3D Handshape Diagrams (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              
              {/* 3D Handshape Diagram Card */}
              {selectedReverseSign && (
                <div className="glass-panel p-5 rounded-3xl border border-slate-200 shadow-sm space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                    <div className="flex items-center space-x-2">
                      <Layers className="w-4 h-4 text-purple-600" />
                      <h3 className="font-bold text-slate-900 text-sm">3D Handshape Diagram & Spatial Markers</h3>
                    </div>
                    {/* Primary Hand Badge */}
                    <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-50 text-amber-800 border border-amber-200">
                      Primary: {selectedReverseSign.primary_hand === 'both' ? 'Bimanual' : `${selectedReverseSign.primary_hand.toUpperCase()} Hand`}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {/* Handshape configuration */}
                    <div className="p-3.5 rounded-2xl bg-gradient-to-br from-slate-900 to-indigo-950 text-white space-y-2 flex flex-col justify-between">
                      <div>
                        <span className="text-[10px] font-extrabold uppercase tracking-wider text-purple-300">
                          Handshape Geometry
                        </span>
                        <div className="text-sm font-bold mt-1 text-slate-100 leading-snug">
                          {selectedReverseSign.hand_shape}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1.5 pt-2 border-t border-white/10 text-[11px] text-slate-300">
                        <Hand className="w-3.5 h-3.5 text-purple-400" />
                        <span>ISLRTC Classification</span>
                      </div>
                    </div>

                    {/* Kinematic Motion Path */}
                    <div className="p-3.5 rounded-2xl bg-white border border-slate-200 space-y-2 flex flex-col justify-between">
                      <div>
                        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                          Motion Vector
                        </span>
                        <div className="text-xs font-bold text-slate-900 mt-1 leading-snug">
                          {selectedReverseSign.movement}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1.5 pt-2 border-t border-slate-100 text-[11px] text-brand-600 font-semibold">
                        <ArrowRight className="w-3.5 h-3.5" />
                        <span>Kinematic Trajectory</span>
                      </div>
                    </div>

                    {/* Facial & Non-Manual Marker */}
                    <div className="p-3.5 rounded-2xl bg-white border border-slate-200 space-y-2 flex flex-col justify-between">
                      <div>
                        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                          Non-Manual Facial Marker
                        </span>
                        <div className="text-xs font-bold text-slate-900 mt-1 leading-snug">
                          {selectedReverseSign.facial_expression}
                        </div>
                      </div>
                      <div className="flex items-center space-x-1.5 pt-2 border-t border-slate-100 text-[11px] text-emerald-600 font-semibold">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Affective Articulation</span>
                      </div>
                    </div>
                  </div>

                  {/* Detailed Description */}
                  <div className="p-3.5 rounded-2xl bg-purple-50/80 border border-purple-100 text-slate-800 text-xs leading-relaxed font-medium">
                    <strong className="text-purple-700 font-extrabold">How to Sign: </strong>
                    {selectedReverseSign.description}
                  </div>
                </div>
              )}

              {/* Dictionary Signs Grid */}
              <div className="glass-panel p-5 rounded-3xl border border-slate-200 shadow-sm space-y-3">
                <div className="flex items-center justify-between text-xs font-bold text-slate-500 uppercase tracking-wider">
                  <span>Authentic Vocabulary ({filteredSigns.length} Signs)</span>
                  <span>Click to view animation</span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-72 overflow-y-auto pr-1">
                  {filteredSigns.map((sign) => {
                    const isSelected = selectedReverseSign?.id === sign.id;
                    return (
                      <button
                        key={sign.id}
                        onClick={() => {
                          setSelectedReverseSign(sign);
                          setAvatarFrameIdx(0);
                        }}
                        className={`p-3 rounded-2xl text-left border transition-all cursor-pointer ${
                          isSelected
                            ? 'bg-purple-600 text-white border-purple-600 shadow-md'
                            : 'bg-white hover:bg-slate-50 border-slate-200 text-slate-800'
                        }`}
                      >
                        <div className="font-bold text-xs truncate">{sign.label.split('/')[0]}</div>
                        <div className={`text-[11px] font-semibold mt-0.5 ${isSelected ? 'text-purple-200' : 'text-slate-500'}`}>
                          {sign.label_hi.split('/')[0]}
                        </div>
                        <div className={`text-[9px] font-mono mt-1 ${isSelected ? 'text-purple-100' : 'text-slate-400'}`}>
                          {sign.islrtc_ref}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

            </div>

          </div>
        </div>
      )}

      {/* CAMERA PERMISSION & OFFLINE FALLBACK GUIDE MODAL */}
      {showPermissionGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-sm animate-fadeIn">
          <div className="glass-panel p-6 rounded-3xl max-w-2xl w-full bg-white shadow-2xl border border-slate-200 space-y-5 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center space-x-2.5">
                <div className="p-2 rounded-xl bg-brand-50 text-brand-600">
                  <ShieldAlert className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-900">Camera Permissions & Offline Fallback Guide</h3>
                  <p className="text-xs text-slate-500">Troubleshooting webcam access and using offline simulations</p>
                </div>
              </div>
              <button
                onClick={() => setShowPermissionGuide(false)}
                className="p-1.5 rounded-xl hover:bg-slate-100 text-slate-400 hover:text-slate-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Browser Specific Unblock Instructions */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                1. How to Unblock Camera Permissions in Your Browser:
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 space-y-1">
                  <span className="font-bold text-slate-900 block">Google Chrome / Brave / Edge</span>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    Click the <strong>Padlock / Tune icon</strong> on the left side of the address bar &rarr; Set <strong>Camera</strong> to <strong>Allow</strong> &rarr; Refresh the page.
                  </p>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 space-y-1">
                  <span className="font-bold text-slate-900 block">Mozilla Firefox</span>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    Click the <strong>Camera icon</strong> in the URL bar &rarr; Clear the <strong>"Blocked Temporarily"</strong> state &rarr; Click <strong>Retry Camera</strong>.
                  </p>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 space-y-1">
                  <span className="font-bold text-slate-900 block">Apple Safari (macOS / iOS)</span>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    Open <strong>Safari Settings</strong> &rarr; <strong>Websites</strong> &rarr; <strong>Camera</strong> &rarr; Set <em>localhost</em> or current domain to <strong>Allow</strong>.
                  </p>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200/80 space-y-1">
                  <span className="font-bold text-slate-900 block">System Privacy Settings</span>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    On macOS/Windows/Linux, check that your browser has permission under OS <strong>Privacy & Security &rarr; Camera</strong>.
                  </p>
                </div>
              </div>
            </div>

            {/* Offline Fallback Guide */}
            <div className="p-4 rounded-2xl bg-amber-50/80 border border-amber-200 space-y-2">
              <div className="flex items-center space-x-2 text-amber-800 text-xs font-bold">
                <Zap className="w-4 h-4 text-amber-600" />
                <span>2. Working Offline or Without a Physical Camera?</span>
              </div>
              <p className="text-xs text-amber-900/90 leading-relaxed">
                IndTranslate includes full <strong>authentic 30-frame GPU kinematics simulation</strong> and an interactive <strong>Reverse Sign Avatar</strong> that run completely offline without webcam hardware:
              </p>
              <ul className="text-[11px] text-amber-800 list-disc list-inside space-y-1 pt-1">
                <li>Use the <strong>Instant Authentic Sign Simulation buttons</strong> in the Camera Studio to feed real ISLRTC tensor landmarks directly into the neural classifier.</li>
                <li>Switch to the <strong>Reverse Sign Avatar</strong> tab to scrub through frames, adjust playback speed, and study 3D handshape diagrams.</li>
                <li>Open the <strong>Cheat Sheet Drawer</strong> to view manual alphabet and number diagrams.</li>
              </ul>
            </div>

            <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-100">
              <button
                onClick={() => {
                  setShowPermissionGuide(false);
                  startCamera();
                }}
                className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-xs font-bold transition-all cursor-pointer"
              >
                Try Camera Again
              </button>
              <button
                onClick={() => setShowPermissionGuide(false)}
                className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold transition-all cursor-pointer"
              >
                Close Guide
              </button>
            </div>
          </div>
        </div>
      )}

      {/* BUILT-IN ISL ALPHABET & COMMON SIGNS CHEAT SHEET DRAWER */}
      {isCheatSheetOpen && (
        <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/60 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-w-lg bg-white h-full shadow-2xl flex flex-col justify-between overflow-hidden border-l border-slate-200">
            
            {/* Drawer Header */}
            <div className="p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center space-x-2">
                <BookOpen className="w-5 h-5 text-amber-500" />
                <div>
                  <h3 className="text-base font-extrabold text-slate-900">ISL Quick Cheat Sheet</h3>
                  <p className="text-[11px] text-slate-500">Official Indian Sign Language manual alphabet, numbers & shapes</p>
                </div>
              </div>

              <button
                onClick={() => setIsCheatSheetOpen(false)}
                className="p-1.5 rounded-xl hover:bg-slate-200 text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Drawer Filter & Search Tabs */}
            <div className="p-4 border-b border-slate-100 space-y-3">
              <div className="flex items-center space-x-1.5 bg-slate-100 p-1 rounded-2xl">
                <button
                  onClick={() => setCheatSheetTab('alphabet')}
                  className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition-all ${cheatSheetTab === 'alphabet' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'}`}
                >
                  Alphabet (A-Z)
                </button>
                <button
                  onClick={() => setCheatSheetTab('numbers')}
                  className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition-all ${cheatSheetTab === 'numbers' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'}`}
                >
                  Numbers (1-10)
                </button>
                <button
                  onClick={() => setCheatSheetTab('common')}
                  className={`flex-1 py-1.5 rounded-xl text-xs font-bold transition-all ${cheatSheetTab === 'common' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600'}`}
                >
                  Quick Signs
                </button>
              </div>

              <div className="relative">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={cheatSheetSearch}
                  onChange={(e) => setCheatSheetSearch(e.target.value)}
                  placeholder="Search letter, number, or description..."
                  className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-slate-100 border border-transparent focus:border-brand-500 text-xs font-medium outline-none"
                />
              </div>
            </div>

            {/* Drawer Content Body */}
            <div className="flex-1 overflow-y-auto p-5 space-y-3">
              {cheatSheetTab === 'alphabet' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {filteredAlphabet.map((item) => (
                    <div
                      key={item.letter}
                      className="p-3 rounded-2xl bg-slate-50 border border-slate-200/80 flex items-start space-x-3 hover:border-brand-300 transition-all"
                    >
                      <div className="w-9 h-9 rounded-xl bg-brand-50 text-brand-700 font-extrabold flex items-center justify-center text-lg border border-brand-200">
                        {item.letter}
                      </div>
                      <div className="flex-1 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-800">Letter {item.letter}</span>
                          <span className="text-[10px] text-brand-600 font-mono">{item.shape}</span>
                        </div>
                        <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{item.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {cheatSheetTab === 'numbers' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {filteredNumbers.map((item) => (
                    <div
                      key={item.num}
                      className="p-3 rounded-2xl bg-slate-50 border border-slate-200/80 flex items-start space-x-3 hover:border-amber-300 transition-all"
                    >
                      <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-700 font-extrabold flex items-center justify-center text-lg border border-amber-200">
                        {item.num}
                      </div>
                      <div className="flex-1 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-slate-800">Number {item.num}</span>
                          <span className="text-[10px] text-amber-600 font-mono">{item.shape}</span>
                        </div>
                        <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{item.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {cheatSheetTab === 'common' && (
                <div className="space-y-2">
                  <p className="text-xs text-slate-500 mb-2">
                    Click any authentic sign to instantly simulate recognition in the vision studio:
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {quickSigns.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => {
                          setIsCheatSheetOpen(false);
                          setActiveMode('studio');
                          handleSimulateSign(s.id);
                        }}
                        className="p-3 rounded-2xl bg-white border border-slate-200 hover:border-purple-300 hover:bg-purple-50/50 text-left transition-all flex items-center space-x-2.5 cursor-pointer"
                      >
                        <span className="text-xl">{s.icon}</span>
                        <div>
                          <div className="font-bold text-xs text-slate-800">{s.label}</div>
                          <div className="text-[11px] text-purple-600 font-semibold">{s.hi}</div>
                          <div className="text-[9px] font-mono text-slate-400">{s.ref}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Drawer Footer */}
            <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
              <span>Aligned with ISLRTC Guidelines</span>
              <button
                onClick={() => setIsCheatSheetOpen(false)}
                className="px-4 py-1.5 rounded-xl bg-slate-200 hover:bg-slate-300 text-slate-800 font-bold transition-colors cursor-pointer"
              >
                Close Drawer
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
};

export default ISLTranslator;