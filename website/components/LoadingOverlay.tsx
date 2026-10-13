"use client";

import { useEffect, useMemo, useRef, useState } from "react";

interface LoadingOverlayProps {
  phase: "uploading" | "analyzing";
  progress: number;
  file?: File | null;
}

const STAGES = [
  { label: "01 · Frame extraction",      status: "Extracting frames…",        from: 0  },
  { label: "02 · Spatial pass",          status: "Running spatial analysis…", from: 3  },
  { label: "03 · Frequency pass",        status: "Running frequency analysis…", from: 8 },
  { label: "04 · Fusion",                status: "Fusing signals into a verdict…", from: 14 },
] as const;

type StageState = "done" | "active" | "pending";

export default function LoadingOverlay({ phase, progress, file }: LoadingOverlayProps) {
  const [elapsed, setElapsed] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Create a FRESH blob URL on every mount/phase change so the effect
  // cleanup (StrictMode double-invoke) never revokes the URL we are playing.
  useEffect(() => {
    if (!file || phase !== "analyzing") return;
    const url = URL.createObjectURL(file);
    const raf = requestAnimationFrame(() => setVideoUrl(url));
    return () => {
      cancelAnimationFrame(raf);
      URL.revokeObjectURL(url);
    };
  }, [phase, file]);

  // Force playback once a valid source is attached (muted autoplay).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !videoUrl || phase !== "analyzing") return;
    const p = v.play();
    if (p) p.catch(() => {});
  }, [videoUrl, phase]);

  useEffect(() => {
    if (phase !== "analyzing") return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [phase]);

  const stages = useMemo<StageState[]>(() => {
    if (phase !== "analyzing") return STAGES.map(() => "pending");

    let activeIdx = STAGES.findLastIndex((s) => elapsed >= s.from);
    if (activeIdx < 0) activeIdx = 0;

    return STAGES.map((_, i) => {
      if (i < activeIdx) return "done";
      if (i === activeIdx) return "active";
      return "pending";
    });
  }, [elapsed, phase]);

  const isUploading = phase === "uploading";
  const statusText = useMemo(() => {
    if (isUploading) {
      return `Uploading file… ${Math.min(100, Math.round(progress))}%`;
    }
    return STAGES.find((_, i) => stages[i] === "active")?.status ?? "";
  }, [isUploading, progress, stages]);

  const fill = isUploading
    ? Math.min(100, Math.round(progress))
    : ((stages.findIndex((s) => s === "active") + 1) / STAGES.length) * 100;

  return (
    <div role="status" aria-live="polite" className="analyzing-panel">
      {!isUploading && videoUrl && (
        <div className="scan-preview">
          <video
            ref={videoRef}
            src={videoUrl}
            autoPlay
            loop
            muted
            playsInline
            disablePictureInPicture
            controls={false}
          />
          <div className="scanner-line" aria-hidden="true" />
          <div className="scan-label">
            <span className="live-dot" />
            Scanning
          </div>
          <div className="scan-elapsed">{elapsed}s</div>
        </div>
      )}

      <div className="analyzing-head">
        <span className="filename">{file?.name ?? "video.mp4"}</span>
        <span className="live">
          <span className="live-dot" />
          {isUploading ? "UPLOADING" : "PROCESSING"}
        </span>
      </div>

      <div className="analyzing-status">{statusText}</div>

      <div className="analyzing-track">
        <div className="analyzing-fill" style={{ width: `${Math.max(fill, 2)}%` }} />
      </div>

      {!isUploading && (
        <div className="analyzing-steps">
          {STAGES.map((stage, i) => (
            <div
              key={stage.label}
              className={`st ${stages[i] === "active" ? "active" : stages[i] === "done" ? "done" : ""}`}
            >
              {stage.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}