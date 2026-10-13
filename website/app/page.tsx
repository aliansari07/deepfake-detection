"use client";

import { useCallback, useReducer } from "react";
import type { ReactNode } from "react";

import Hero from "@/components/Hero";
import UploadZone from "@/components/UploadZone";
import LoadingOverlay from "@/components/LoadingOverlay";
import VerdictCard from "@/components/VerdictCard";
import ErrorBanner from "@/components/ErrorBanner";
import ArchitectureDiagram from "@/components/ArchitectureDiagram";
import type { AnalysisError, AnalysisResult } from "@/types";
import stats from "@/lib/modelStats.json";

const API_ENDPOINT =
  process.env.NEXT_PUBLIC_API_ENDPOINT ?? "http://localhost:8000/analyze";

const REQUEST_TIMEOUT_MS = 120_000;

type State =
  | { status: "idle"; file: null }
  | { status: "ready"; file: File }
  | { status: "uploading"; file: File; progress: number }
  | { status: "analyzing"; file: File; elapsed: number }
  | { status: "result"; file: File; result: AnalysisResult }
  | { status: "error"; file: File; error: AnalysisError };

type Action =
  | { type: "SELECT_FILE"; file: File }
  | { type: "START_UPLOAD" }
  | { type: "PROGRESS"; pct: number }
  | { type: "START_ANALYZE" }
  | { type: "TICK" }
  | { type: "RESULT"; result: AnalysisResult }
  | { type: "ERROR"; error: AnalysisError }
  | { type: "RESET" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SELECT_FILE":
      return { status: "ready", file: action.file };
    case "START_UPLOAD": {
      if (!state.file) return state;
      return { status: "uploading", file: state.file, progress: 0 };
    }
    case "PROGRESS":
      if (state.status !== "uploading") return state;
      return { ...state, progress: action.pct };
    case "START_ANALYZE": {
      if (state.status !== "uploading" || !state.file) return state;
      return { status: "analyzing", file: state.file, elapsed: 0 };
    }
    case "TICK":
      if (state.status !== "analyzing") return state;
      return { ...state, elapsed: state.elapsed + 1 };
    case "RESULT": {
      if (!state.file) return state;
      return { status: "result", file: state.file, result: action.result };
    }
    case "ERROR": {
      if (!state.file) return state;
      return { status: "error", file: state.file, error: action.error };
    }
    case "RESET":
      return { status: "idle", file: null };
    default:
      return state;
  }
}

const initialState: State = { status: "idle", file: null };

function SpatialIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <rect x="4" y="4" width="32" height="32" stroke="#ff4d4d" strokeWidth="1.4" />
      <circle cx="20" cy="20" r="7" stroke="#ff4d4d" strokeWidth="1.4" />
      <line x1="20" y1="4" x2="20" y2="13" stroke="#ff4d4d" strokeWidth="1" />
      <line x1="20" y1="27" x2="20" y2="36" stroke="#ff4d4d" strokeWidth="1" />
    </svg>
  );
}

function FrequencyIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <path d="M4 26 Q10 10 16 26 T28 26 T40 26" stroke="#35e0c7" strokeWidth="1.4" fill="none" />
      <line x1="4" y1="34" x2="40" y2="34" stroke="#35e0c7" strokeWidth="1" />
    </svg>
  );
}

function FusionIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <circle cx="10" cy="12" r="3" stroke="#e9ecec" strokeWidth="1.3" />
      <circle cx="30" cy="12" r="3" stroke="#e9ecec" strokeWidth="1.3" />
      <circle cx="20" cy="30" r="3" stroke="#e9ecec" strokeWidth="1.3" />
      <line x1="12.5" y1="13.5" x2="18" y2="27.5" stroke="#e9ecec" strokeWidth="1" />
      <line x1="27.5" y1="13.5" x2="22" y2="27.5" stroke="#e9ecec" strokeWidth="1" />
      <line x1="13" y1="12" x2="27" y2="12" stroke="#e9ecec" strokeWidth="1" />
    </svg>
  );
}

function CropIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <path d="M4 13V4h9M27 4h9v9M36 27v9h-9M13 36H4v-9" stroke="#35e0c7" strokeWidth="1.4" />
      <circle cx="20" cy="20" r="6" stroke="#35e0c7" strokeWidth="1.2" />
    </svg>
  );
}

function PoolIcon() {
  return (
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
      <rect x="5" y="6" width="8" height="8" stroke="#e9ecec" strokeWidth="1.2" />
      <rect x="16" y="6" width="8" height="8" stroke="#e9ecec" strokeWidth="1.2" />
      <rect x="27" y="6" width="8" height="8" stroke="#e9ecec" strokeWidth="1.2" />
      <rect x="16" y="26" width="8" height="8" stroke="#e9ecec" strokeWidth="1.2" />
      <line x1="9" y1="15" x2="18" y2="25" stroke="#e9ecec" strokeWidth="1" />
      <line x1="20" y1="15" x2="20" y2="25" stroke="#e9ecec" strokeWidth="1" />
      <line x1="31" y1="15" x2="22" y2="25" stroke="#e9ecec" strokeWidth="1" />
    </svg>
  );
}

const FREQUENCY_BRANCH = stats.architecture.frequency_branch;
// An R5 checkpoint takes no Fourier transform, so the three places that name the transform read
// it from the arch block instead of stating it. See F2a: copy that describes a component the
// served model does not run is the defect this whole generated-stats path exists to prevent.
const FREQ_IS_RESIDUAL =
  (stats.architecture as { frequency_input?: string }).frequency_input === "residual";
const FREQ_STAGE_TITLE = FREQ_IS_RESIDUAL
  ? "High-frequency residual"
  : "Frequency analysis";
const FREQ_STAGE_BODY = FREQ_IS_RESIDUAL
  ? "Subtracting a blurred copy of each frame leaves only its finest detail, " +
    "where the resampling and blending seams of a generative pipeline survive " +
    "even after the frame is re-encoded."
  : "A frequency-domain transform surfaces the periodic residue GAN " +
    "and diffusion pipelines stamp into a frame — patterns invisible " +
    "in the spatial domain alone.";
const FREQ_MODULE_DESC = FREQ_IS_RESIDUAL
  ? "High-pass spatial residual isolating the fine detail that generative " +
    "upsampling and reconstruction disturb."
  : "FFT-based transform isolating high-frequency artifacts " +
    "characteristic of generative upsampling and reconstruction.";
const TRANSFORMER = stats.architecture.transformer;
const FRAMES = stats.architecture.frames_per_clip;

const SPATIAL_BODY =
  "An EfficientNet-B0 backbone inspects texture, lighting continuity, and " +
  "blending seams around the face — the artifacts generative models leave " +
  "behind at the pixel level.";

const CROP_BODY = `SCRFD locates faces across the clip; ${FRAMES} frames are ` +
  "sampled and cropped to 224×224. Every later stage sees the face crop, " +
  "not the whole frame.";

// "How it works" describes the pipeline the served checkpoint actually runs, so the stages are
// built from the same arch block the diagram reads. Listing a stage the model does not have is
// how this page came to describe a three-branch model while serving a spatial-only one.
const STAGE_SPECS: { icon: () => ReactNode; title: string; body: string }[] =
  FREQUENCY_BRANCH
    ? [
        { icon: SpatialIcon, title: "Spatial analysis", body: SPATIAL_BODY },
        {
          icon: FrequencyIcon,
          title: FREQ_STAGE_TITLE,
          body: FREQ_STAGE_BODY,
        },
        {
          icon: FusionIcon,
          title: TRANSFORMER ? "Transformer fusion" : "Gated fusion, then pooling",
          body: TRANSFORMER
            ? "A learned gate blends spatial and frequency features for each " +
              "frame, then a transformer reasons across all sampled frames to " +
              "produce one verdict."
            : "A learned gate blends spatial and frequency features for each " +
              "frame, then the frames are averaged into one clip-level vector " +
              "before a small MLP turns it into a verdict.",
        },
      ]
    : TRANSFORMER
      ? [
          { icon: CropIcon, title: "Face detection & cropping", body: CROP_BODY },
          { icon: SpatialIcon, title: "Spatial analysis", body: SPATIAL_BODY },
          {
            icon: FusionIcon,
            title: "Transformer over frames",
            body:
              "Per-frame features are projected into a 512-d sequence and a " +
              "two-layer transformer reasons across the sampled frames to " +
              "produce one verdict.",
          },
        ]
      : [
          { icon: CropIcon, title: "Face detection & cropping", body: CROP_BODY },
          { icon: SpatialIcon, title: "Spatial analysis", body: SPATIAL_BODY },
          {
            icon: PoolIcon,
            title: "Frame aggregation",
            body:
              "Per-frame features are averaged into one clip-level vector and a " +
              "small MLP turns it into a single verdict, so no ambiguous frame " +
              "decides the answer on its own.",
          },
        ];

const PIPELINE_TITLE = FREQUENCY_BRANCH
  ? "Three passes over every frame"
  : "From face crop to verdict";

const PIPELINE_SUB = FREQUENCY_BRANCH
  ? "No single signal is proof. Veritas cross-checks spatial texture, " +
    "frequency residue, and temporal continuity before it commits to a verdict."
  : "The served model is the spatial backbone on its own — the frequency " +
    "branch and the learned gate measured no better than it, so they were left " +
    "out. Veritas reads the face crop, then aggregates the sampled frames " +
    "before it commits to a verdict.";

const DEMO_SUB = FREQUENCY_BRANCH
  ? "Drop in a clip and Veritas runs it through spatial, frequency, and " +
    "fusion analysis — then shows you exactly what it found."
  : "Drop in a clip and Veritas crops every face, scores it on the spatial " +
    "backbone, and shows you exactly what it found.";

// The ablation finding is only the reason for the served model when the served model is the
// ablated one; promoting a full model makes this sentence the wrong explanation for it.
const ARCHITECTURE_SUB = FREQUENCY_BRANCH && TRANSFORMER
  ? "The full architecture fuses a spatial and a frequency stream through a " +
    "learned gate, then reasons across frames with a Transformer. Every " +
    "figure on this page comes from that run on an identity-disjoint split."
  : "The full architecture fuses a spatial and a frequency stream through a " +
    "learned gate, then reasons across frames with a Transformer. Controlled " +
    "ablations on an identity-disjoint split found neither extra stream " +
    "measurably improves on the spatial backbone alone, so the model served " +
    "here is the simplest one that matches them.";

export default function Home() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const busy = state.status === "uploading" || state.status === "analyzing";

  const analyze = useCallback((file: File) => {
    dispatch({ type: "START_UPLOAD" });

    const xhr = new XMLHttpRequest();
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      xhr.abort();
      dispatch({
        type: "ERROR",
        error: {
          message: "Request timed out.",
          detail:
            "The analysis took longer than 120 seconds. Check that the backend is responding and try again.",
        },
      });
    }, REQUEST_TIMEOUT_MS);

    xhr.open("POST", API_ENDPOINT);

    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      dispatch({ type: "PROGRESS", pct });
    };

    xhr.upload.onload = () => {
      if (!timedOut) dispatch({ type: "START_ANALYZE" });
    };

    xhr.onload = () => {
      clearTimeout(timer);
      if (timedOut) return;

      if (xhr.status < 200 || xhr.status >= 300) {
        // FastAPI errors carry a human-readable `detail` (e.g. no faces found).
        let detail: string | undefined;
        try {
          const parsed = JSON.parse(xhr.responseText);
          if (typeof parsed?.detail === "string") detail = parsed.detail;
        } catch {
          // non-JSON body — fall back to the generic message
        }
        dispatch({
          type: "ERROR",
          error: {
            message: `Backend returned HTTP ${xhr.status}.`,
            detail: detail ?? `Unexpected response from ${API_ENDPOINT}.`,
          },
        });
        return;
      }

      try {
        const body = JSON.parse(xhr.responseText);
        if (
          body &&
          (body.verdict === "REAL" ||
            body.verdict === "FAKE" ||
            body.verdict === "UNCERTAIN") &&
          typeof body.confidence === "number"
        ) {
          dispatch({
            type: "RESULT",
            result: {
              verdict: body.verdict,
              confidence: body.confidence,
              prob: typeof body.prob === "number" ? body.prob : body.confidence,
              logit: typeof body.logit === "number" ? body.logit : 0,
              spatial_weight:
                typeof body.spatial_weight === "number"
                  ? body.spatial_weight
                  : 0.5,
              freq_weight:
                typeof body.freq_weight === "number"
                  ? body.freq_weight
                  : typeof body.spatial_weight === "number"
                    ? 1 - body.spatial_weight
                    : 0.5,
              gradcam_base64:
                typeof body.gradcam_base64 === "string"
                  ? body.gradcam_base64
                  : "",
              faces_detected:
                typeof body.faces_detected === "number"
                  ? body.faces_detected
                  : 0,
              faces_total:
                typeof body.faces_total === "number" ? body.faces_total : 0,
              analysis_time:
                typeof body.analysis_time === "number" ? body.analysis_time : 0,
              ai_explanation:
                typeof body.ai_explanation === "string" ? body.ai_explanation : "",
            },
          });
        } else {
          throw new Error("Malformed response");
        }
      } catch {
        dispatch({
          type: "ERROR",
          error: {
            message: "Could not parse the response.",
            detail:
              "The backend returned data that is not a valid analysis result.",
          },
        });
      }
    };

    xhr.onerror = () => {
      clearTimeout(timer);
      if (timedOut) return;
      dispatch({
        type: "ERROR",
        error: {
          message: "Network request failed.",
          detail: `Could not reach ${API_ENDPOINT}. Check that the backend server is running.`,
        },
      });
    };

    const form = new FormData();
    form.append("video", file);
    xhr.send(form);
  }, []);

  const handleSelect = useCallback((file: File) => {
    dispatch({ type: "SELECT_FILE", file });
  }, []);

  const handleAnalyze = useCallback(() => {
    if (state.status === "ready") analyze(state.file);
  }, [state.status, state.file, analyze]);

  const handleRetry = useCallback(() => {
    if (state.status === "error") analyze(state.file);
  }, [state.status, state.file, analyze]);

  const handleReset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  const analyzing =
    state.status === "uploading" || state.status === "analyzing";

  return (
    <main className="page-root">
      <div className="grid-field" aria-hidden="true" />
      <div className="scan-sweep" aria-hidden="true" />

      <nav className="nav">
        <div className="wrap nav-wrap">
          <div className="brand">
            <span className="brand-mark" />
            VERITAS
          </div>
          <div className="nav-links">
            <a href="#how">How it works</a>
            <a href="#demo">Analysis</a>
            <a href="#architecture">Architecture</a>
          </div>
          <a href="#demo" className="nav-cta">
            Try a video →
          </a>
        </div>
      </nav>

      <Hero />

      {/* How it works */}
      <section className="section" id="how">
        <div className="wrap">
          <div className="section-head">
            <div className="kicker">HOW IT WORKS</div>
            <h2 className="section-title">{PIPELINE_TITLE}</h2>
            <p className="section-sub">{PIPELINE_SUB}</p>
          </div>

          <div className="pipeline">
            {STAGE_SPECS.map((stage, i) => (
              <div className="stage" key={stage.title}>
                <div className="stage-index">
                  {String(i + 1).padStart(2, "0")}
                </div>
                {stage.icon()}
                <h3>{stage.title}</h3>
                <p>{stage.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Analyze a video */}
      <section className="section" id="demo">
        <div className="wrap">
          <div className="section-head">
            <div className="kicker">ANALYZE A VIDEO</div>
            <h2 className="section-title">Upload footage, get a verdict</h2>
            <p className="section-sub">{DEMO_SUB}</p>
          </div>

          {state.status === "result" && (
            <>
              <VerdictCard
                key={`${state.file.name}-${state.result.analysis_time}`}
                result={state.result}
                fileName={state.file.name}
              />
              <button
                type="button"
                onClick={handleReset}
                className="reset-link"
              >
                ↺ Analyze another video
              </button>
            </>
          )}

          {state.status === "error" && (
            <ErrorBanner
              error={state.error}
              onRetry={handleRetry}
              onReset={handleReset}
            />
          )}

          {(state.status === "idle" || state.status === "ready") && (
            <UploadZone
              file={state.status === "ready" ? state.file : null}
              onSelect={handleSelect}
              onAnalyze={handleAnalyze}
              onClear={handleReset}
              disabled={busy}
            />
          )}

          {analyzing && (
            <LoadingOverlay
              key={state.status}
              phase={state.status as "uploading" | "analyzing"}
              progress={state.status === "uploading" ? state.progress : 100}
              file={state.file}
            />
          )}
        </div>
      </section>

      {/* Stats */}
      <section className="section">
        <div className="wrap">
          <div className="stats">
            <div className="stat">
              <div className="stat-num cyan">
                {(stats.test.balanced_accuracy * 100).toFixed(1)}%
              </div>
              <div className="stat-label">BALANCED ACCURACY (FF++ HELD-OUT TEST)</div>
            </div>
            <div className="stat">
              <div className="stat-num">{stats.test.auc.toFixed(3)}</div>
              <div className="stat-label">ROC-AUC (FF++ TEST)</div>
            </div>
            <div className="stat">
              <div className="stat-num cyan">{stats.cross_dataset.auc.toFixed(3)}</div>
              <div className="stat-label">ROC-AUC (CELEB-DF, UNSEEN DATASET)</div>
            </div>
            <div className="stat">
              <div className="stat-num">{stats.architecture.frames_per_clip}</div>
              <div className="stat-label">FACES SAMPLED / CLIP</div>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="section" id="architecture">
        <div className="wrap">
          <div className="section-head">
            <div className="kicker">ARCHITECTURE</div>
            <h2 className="section-title">What&apos;s under the verdict</h2>
            <p className="section-sub">{ARCHITECTURE_SUB}</p>
          </div>

          <div className="diagram-shell">
            <ArchitectureDiagram />
          </div>

          <div className="arch-list">
            <div className="arch-row">
              <div className="arch-name">EfficientNet-B0</div>
              <div className="arch-desc">
                Convolutional backbone for per-frame spatial feature extraction
                — compact enough to run per-face, per-frame, at scale.
              </div>
              <div className="arch-weight">
                {FREQUENCY_BRANCH ? "fused" : "served"}
              </div>
            </div>
            <div className="arch-row">
              <div className="arch-name">Frequency module</div>
              <div className="arch-desc">{FREQ_MODULE_DESC}</div>
              <div className="arch-weight">
                {FREQUENCY_BRANCH ? "fused" : "ablated out"}
              </div>
            </div>
            <div className="arch-row">
              <div className="arch-name">Transformer fusion head</div>
              <div className="arch-desc">
                Self-attention across the sampled faces after gated fusion,
                producing one logit per clip.
                {!TRANSFORMER &&
                  " The served model pools the frames directly instead, which " +
                    "measured the same."}
              </div>
              <div className="arch-weight">
                {TRANSFORMER ? "served" : "ablated out"}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta section">
        <div className="wrap">
          <h2>Don&apos;t trust the frame. Trust the signal.</h2>
          <p>
            Upload a clip and get a verdict, a confidence score, and the
            evidence behind it.
          </p>
          <div className="hero-actions">
            <a href="#demo" className="btn btn-primary">
              Analyze a video
            </a>
          </div>
        </div>
      </section>

      <footer className="site-footer">
        <div className="wrap">
          <span>VERITAS — FORENSIC SIGNAL ANALYSIS</span>
          <span>Results are probabilistic. Treat accordingly.</span>
        </div>
      </footer>
    </main>
  );
}