import React, { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, AlertTriangle, Sparkles, Info, X } from 'lucide-react';

export type ToastType = 'error' | 'success' | 'warning' | 'info';

export interface ToastItem {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

interface ToastProps {
  toast: ToastItem;
  onDismiss: (id: string) => void;
}

export const Toast: React.FC<ToastProps> = ({ toast, onDismiss }) => {
  const [progress, setProgress] = useState(100);
  const duration = toast.duration ?? 4000;

  useEffect(() => {
    if (duration <= 0) return;
    const intervalTime = 50;
    const step = (intervalTime / duration) * 100;
    const interval = setInterval(() => {
      setProgress((prev) => Math.max(0, prev - step));
    }, intervalTime);

    return () => clearInterval(interval);
  }, [duration]);

  const getIcon = () => {
    switch (toast.type) {
      case 'error':
        return (
          <div className="w-7 h-7 rounded-xl bg-red-100/90 text-red-600 flex items-center justify-center flex-shrink-0 shadow-xs">
            <AlertCircle className="w-4 h-4" />
          </div>
        );
      case 'success':
        return (
          <div className="w-7 h-7 rounded-xl bg-emerald-100/90 text-emerald-700 flex items-center justify-center flex-shrink-0 shadow-xs">
            <CheckCircle2 className="w-4 h-4" />
          </div>
        );
      case 'warning':
        return (
          <div className="w-7 h-7 rounded-xl bg-saffron-100/90 text-saffron-700 flex items-center justify-center flex-shrink-0 shadow-xs">
            <AlertTriangle className="w-4 h-4" />
          </div>
        );
      case 'info':
      default:
        return (
          <div className="w-7 h-7 rounded-xl bg-chakra-100/90 text-chakra-700 flex items-center justify-center flex-shrink-0 shadow-xs">
            <Sparkles className="w-4 h-4" />
          </div>
        );
    }
  };

  const getStyle = () => {
    switch (toast.type) {
      case 'error':
        return 'bg-white/95 border-red-200/90 text-red-950 shadow-red-500/10 hover:border-red-300';
      case 'success':
        return 'bg-white/95 border-emerald-200/90 text-emerald-950 shadow-emerald-500/10 hover:border-emerald-300';
      case 'warning':
        return 'bg-white/95 border-saffron-200/90 text-saffron-950 shadow-saffron-500/10 hover:border-saffron-300';
      case 'info':
      default:
        return 'bg-white/95 border-chakra-200/90 text-chakra-950 shadow-chakra-500/10 hover:border-chakra-300';
    }
  };

  const getProgressColor = () => {
    switch (toast.type) {
      case 'error':
        return 'bg-red-500';
      case 'success':
        return 'bg-emerald-500';
      case 'warning':
        return 'bg-saffron-500';
      case 'info':
      default:
        return 'bg-chakra-600';
    }
  };

  return (
    <div
      role="alert"
      aria-live={toast.type === 'error' ? 'assertive' : 'polite'}
      className={`relative overflow-hidden flex items-start space-x-3 p-3.5 rounded-2xl border shadow-xl 
        backdrop-blur-xl max-w-sm w-full transition-all duration-300 pointer-events-auto animate-slideUp group ${getStyle()}`}
    >
      {/* Icon */}
      {getIcon()}

      {/* Content */}
      <div className="flex-1 text-xs font-semibold leading-relaxed pt-0.5 pr-1 break-words">
        {toast.message}
      </div>

      {/* Dismiss Button */}
      <button
        onClick={() => onDismiss(toast.id)}
        aria-label="Close notification"
        className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors flex-shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>

      {/* Auto-dismiss countdown bar */}
      {duration > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-100 overflow-hidden">
          <div
            className={`h-full transition-all ease-linear ${getProgressColor()}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
};

export default Toast;
