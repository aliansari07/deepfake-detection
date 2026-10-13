"use client";

import { useDecodeText, useCountUp } from "@/lib/anim";
import stats from "@/lib/modelStats.json";

const HERO_CONF = 94;

export default function Hero() {
  const word = useDecodeText("FAKE", 700);
  const conf = useCountUp(HERO_CONF, 1400);

  return (
    <header className="hero">
      <div className="wrap">
        <div className="eyebrow">
          <span className="dot" />
          Forensic signal analysis, not gut feeling
        </div>

        <div className="hero-grid">
          <div>
            <h1 className="headline">
              What the camera
              <br />
              can&apos;t tell you,
              <br />
              the signal <span className="decode">{word}</span>.
            </h1>
            <p className="sub">
              Veritas reads video the way a forensic lab does — pixel artifacts,
              frequency residue, and temporal drift the eye was never built to
              catch. One verdict, backed by the evidence behind it.
            </p>
            <div className="hero-actions">
              <a href="#demo" className="btn btn-primary">
                Analyze a video
              </a>
              <a href="#architecture" className="btn btn-ghost">
                See the model
              </a>
            </div>
          </div>

          <div className="verdict-card">
            <div className="verdict-top">
              <span className="verdict-label">SAMPLE_042.mp4</span>
              <span className="verdict-tag text-fake">FAKE</span>
            </div>
            <div className="verdict-body">
              <div className="verdict-big text-fake">
                <span className="conf-num">{conf}</span>%
              </div>
              <div className="verdict-conf">confidence · high</div>
              <div className="bar-track">
                <div
                  className="bar-fill bg-fake"
                  style={{ width: `${conf}%` }}
                />
              </div>
              <div className="verdict-meta">
                <div>
                  <span className="k">MODEL</span>
                  <span className="v">{stats.model.run}</span>
                </div>
                <div>
                  <span className="k">TEST AUC</span>
                  <span className="v">{stats.test.auc.toFixed(3)}</span>
                </div>
                <div>
                  <span className="k">FACES SCANNED</span>
                  <span className="v">10 / 10</span>
                </div>
                <div>
                  <span className="k">ANALYSIS TIME</span>
                  <span className="v">4.9s</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}