"use client";

import type { AnalysisResult } from "@/types";
import { useCountUp, useDecodeText } from "@/lib/anim";
import stats from "@/lib/modelStats.json";

export default function VerdictCard({
  result,
  fileName,
}: {
  result: AnalysisResult;
  fileName: string;
}) {
  const confPct = Math.round(result.confidence * 100);
  const spatialPct = Math.round(result.spatial_weight * 100);
  const freqPct = Math.round(result.freq_weight * 100);

  const variant =
    result.verdict === "FAKE"
      ? "fake"
      : result.verdict === "UNCERTAIN"
        ? "uncertain"
        : "real";
  const verbColor = variant === "fake" ? "text-fake" : variant === "uncertain" ? "text-warn" : "text-real";

  const decodedVerdict = useDecodeText(result.verdict, 800);
  const decodedConf = useCountUp(confPct, 1200);

  const imageSrc =
    /^data:image\/png;base64,/i.test(result.gradcam_base64) ||
    /^https?:\/\//i.test(result.gradcam_base64)
      ? result.gradcam_base64
      : `data:image/png;base64,${result.gradcam_base64}`;

  const faces =
    result.faces_total > 0
      ? `${result.faces_detected}/${result.faces_total} faces scanned`
      : `${result.faces_detected} faces scanned`;

  return (
    <div className="demo-frame">
      <div className="result-summary">
        <div className="min-w-0">
          <div className={`result-verdict ${variant} ${verbColor}`}>
            {decodedVerdict}
          </div>
          <div className="result-conf">
            {fileName || "unknown"} · {faces} · {result.analysis_time.toFixed(1)}s
          </div>
        </div>
        <div className={`result-tag ${variant}`}>
          <span className="conf-num">{decodedConf}</span>% confidence
        </div>
      </div>

      <div className="demo-bar">
        <span>GRAD-CAM — REGIONS THAT DROVE THE VERDICT</span>
        <div className="demo-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </div>

      {result.gradcam_base64 ? (
        <div className="gradcam-wipe">
          <div className="sweep-line" aria-hidden="true" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageSrc}
            alt={`Grad-CAM heatmap showing model focus regions classified as ${result.verdict}`}
            className="gradcam-img"
          />
        </div>
      ) : (
        <div className="result-conf">Heatmap unavailable for this video.</div>
      )}

      <div className="demo-foot">
        <span>low contribution ───────────── high contribution</span>
        <span>
          {stats.architecture.frequency_branch
            ? `spatial ${spatialPct}% / freq ${freqPct}%`
            : "spatial only — the frequency branch is ablated out of this model"}
        </span>
      </div>

      {result.ai_explanation && (
        <div className="ai-explain">
          <div className="demo-bar">
            <span>AI EXPLANATION</span>
            <div className="demo-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
          </div>
          <p className="ai-explain-text">{result.ai_explanation}</p>
        </div>
      )}
    </div>
  );
}