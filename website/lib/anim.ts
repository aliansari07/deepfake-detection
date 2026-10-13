import { useEffect, useState } from "react";

const SCRAMBLE_CHARS = "!<>-_\\/[]{}=+*^?#";
const FRAME_MS = 45;

function motionAllowed(): boolean {
  if (typeof window === "undefined") return true;
  return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Ease-out cubic count-up. Returns the animated integer. */
export function useCountUp(
  target: number,
  duration = 1200,
  run = true,
): number {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let start: number | null = null;
    let raf = 0;

    const tick = (ts: number) => {
      if (!run || !motionAllowed()) {
        setValue(target);
        return;
      }
      if (start === null) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(eased * target));
      if (progress < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, run]);

  return value;
}

/**
 * Decode-scramble text effect from the blueprint: characters resolve
 * one-by-one, left to right, with random placeholder glyphs in between.
 */
export function useDecodeText(
  target: string,
  duration = 700,
  run = true,
): string {
  const [out, setOut] = useState("");

  useEffect(() => {
    const revealEvery = 3;
    const holdFrames = 4;
    const totalFrames = target.length * revealEvery + holdFrames;
    let start: number | null = null;
    let raf = 0;

    const tick = (ts: number) => {
      if (!run || !motionAllowed()) {
        setOut(target);
        return;
      }
      if (start === null) start = ts;
      const frame = Math.floor((ts - start) / FRAME_MS);
      if (frame > totalFrames) {
        setOut(target);
        return;
      }

      let s = "";
      for (let i = 0; i < target.length; i++) {
        const revealAt = i * revealEvery;
        if (frame >= revealAt + holdFrames) s += target[i];
        else if (frame >= revealAt)
          s += SCRAMBLE_CHARS[Math.floor(Math.random() * SCRAMBLE_CHARS.length)];
        else s += "\u00A0";
      }
      setOut(s);
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, run]);

  return out;
}