"use client";

import { useCallback, useId, useRef, useState } from "react";

const ACCEPTED = ["video/mp4", "video/quicktime", "video/webm"];
// Must match the backend's DD_MAX_UPLOAD_MB default.
const MAX_UPLOAD_MB = 200;
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

function formatSize(bytes: number): string {
  if (bytes > 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes > 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

interface UploadZoneProps {
  file: File | null;
  onSelect: (file: File) => void;
  onAnalyze: () => void;
  onClear: () => void;
  disabled: boolean;
}

export default function UploadZone({
  file,
  onSelect,
  onAnalyze,
  onClear,
  disabled,
}: UploadZoneProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [reject, setReject] = useState<string | null>(null);

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      if (!ACCEPTED.includes(f.type)) {
        setReject(
          `Unsupported format "${f.type || "unknown"}". Accepts MP4, MOV, WEBM.`,
        );
        return;
      }
      if (f.size > MAX_UPLOAD_BYTES) {
        setReject(
          `File is ${formatSize(f.size)}; the limit is ${MAX_UPLOAD_MB} MB.`,
        );
        return;
      }
      setReject(null);
      onSelect(f);
    },
    [onSelect],
  );

  return (
    <div>
      {!file && (
        <div
          className={`upload-zone ${dragging ? "drag" : ""}`}
          role="button"
          tabIndex={0}
          aria-label="Upload a video file"
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={(e) => {
            if (!disabled && (e.key === "Enter" || e.key === " ")) {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(e) => {
            if (disabled) return;
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            if (disabled) return;
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files[0]);
          }}
        >
          <svg
            className="upload-icon"
            width="44"
            height="44"
            viewBox="0 0 44 44"
            fill="none"
            aria-hidden="true"
          >
            <path d="M22 30V10" stroke="currentColor" strokeWidth="1.4" />
            <path d="M13 19L22 10L31 19" stroke="currentColor" strokeWidth="1.4" />
            <path d="M8 34H36" stroke="currentColor" strokeWidth="1.4" />
          </svg>
          <div className="upload-title">Drop a video file here</div>
          <div className="upload-sub">
            or <span className="browse">browse from your device</span>
          </div>
          <div className="upload-formats">MP4 · MOV · WEBM — LOCAL ANALYSIS</div>
        </div>
      )}

      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={ACCEPTED.join(",")}
        className="input-sr"
        disabled={disabled}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {reject && <div className="upload-reject">{reject}</div>}

      {file && (
        <>
          <div className="file-chip">
            <div className="file-chip-info">
              <svg
                width="20"
                height="20"
                viewBox="0 0 20 20"
                fill="none"
                aria-hidden="true"
              >
                <rect
                  x="3"
                  y="2"
                  width="14"
                  height="16"
                  rx="1"
                  stroke="#e9ecec"
                  strokeWidth="1.2"
                />
                <path d="M6 7h8M6 10h8M6 13h5" stroke="#e9ecec" strokeWidth="1.2" />
              </svg>
              <div className="min-w-0">
                <div className="file-chip-name">{file.name}</div>
                <div className="file-chip-size">
                  {formatSize(file.size)} · {file.type || "video"}
                </div>
              </div>
            </div>
            <button
              type="button"
              className="file-chip-remove"
              onClick={() => {
                setReject(null);
                onClear();
              }}
              disabled={disabled}
            >
              Remove
            </button>
          </div>

          <button
            type="button"
            className="analyze-btn"
            onClick={onAnalyze}
            disabled={disabled}
          >
            Run analysis →
          </button>
        </>
      )}
    </div>
  );
}