"use client";

import type { AnalysisError } from "@/types";

interface ErrorBannerProps {
  error: AnalysisError;
  onRetry: () => void;
  onReset: () => void;
}

export default function ErrorBanner({
  error,
  onRetry,
  onReset,
}: ErrorBannerProps) {
  return (
    <div
      role="alert"
      aria-live="assertive"
      className="analyzing-panel"
    >
      <div className="flex items-start gap-3">
        <svg
          className="mt-0.5 h-5 w-5 shrink-0 text-fake"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
          />
        </svg>
        <div className="min-w-0">
          <div className="text-sm font-bold uppercase tracking-widest text-fake">
            Analysis failed
          </div>
          <p className="mt-2 text-sm text-text-bright">{error.message}</p>
          {error.detail && (
            <p className="mt-1 break-words text-xs text-text-dim">
              {error.detail}
            </p>
          )}
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onRetry}
          className="btn text-fake"
          style={{ borderColor: "var(--fake)", color: "var(--fake)" }}
        >
          Retry
        </button>
        <button
          type="button"
          onClick={onReset}
          className="btn btn-ghost"
        >
          Upload different file
        </button>
      </div>
    </div>
  );
}