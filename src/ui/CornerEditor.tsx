import { useRef, useState } from 'react';
import type { Point, Quad } from '../pipeline/types';
import { polygonPoints } from './geometry';

const NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];

/** Four draggable corner handles (mouse + touch + keyboard arrows) over the photo. */
export function CornerEditor({ url, width, height, initial, onSubmit, onCancel }: {
  url: string;
  width: number;
  height: number;
  initial: Quad;
  onSubmit(q: Quad): void;
  onCancel?(): void;
}) {
  const [q, setQ] = useState<Quad>(initial);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const r = Math.min(width, height) * 0.03;
  // Pad the view by more than a handle radius so handles on the image edge (a tightly
  // cropped photo puts the mosaic corners there) stay fully visible and grabbable.
  const pad = r * 1.4;

  const toImage = (clientX: number, clientY: number): Point => {
    const ctm = svg.current!.getScreenCTM()!.inverse();
    const pt = new DOMPoint(clientX, clientY).matrixTransform(ctm);
    return [Math.min(width, Math.max(0, pt.x)), Math.min(height, Math.max(0, pt.y))];
  };
  const setCorner = (i: number, p: Point) => setQ((old) => old.map((c, j) => (j === i ? p : c)) as Quad);

  return (
    <div className="corner-editor">
      <p className="hint">Drag the four handles onto the outer corners of the mosaic's studs, then re-scan.</p>
      <svg
        ref={svg}
        className="corner-svg"
        viewBox={`${-pad} ${-pad} ${width + 2 * pad} ${height + 2 * pad}`}
        onPointerMove={(e) => {
          if (drag.current !== null) setCorner(drag.current, toImage(e.clientX, e.clientY));
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
      >
        <image href={url} width={width} height={height} />
        <polygon className="quad" points={polygonPoints(q)} style={{ strokeWidth: r * 0.25 }} />
        {q.map(([x, y], i) => (
          <circle key={`dot${i}`} className="handle-dot" cx={x} cy={y} r={r * 0.18} pointerEvents="none" />
        ))}
        {q.map(([x, y], i) => (
          <circle
            key={i}
            className="handle"
            cx={x}
            cy={y}
            r={r}
            tabIndex={0}
            role="slider"
            aria-label={`${NAMES[i]} corner`}
            aria-valuetext={`${Math.round(x)}, ${Math.round(y)}`}
            style={{ strokeWidth: r * 0.25 }}
            onPointerDown={(e) => {
              e.preventDefault();
              drag.current = i;
              (e.currentTarget.ownerSVGElement ?? e.currentTarget).setPointerCapture?.(e.pointerId);
            }}
            onKeyDown={(e) => {
              const step = (e.shiftKey ? 10 : 2) * (width / 400);
              const d: Record<string, Point> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
              const m = d[e.key];
              if (!m) return;
              e.preventDefault();
              setCorner(i, [Math.min(width, Math.max(0, x + m[0])), Math.min(height, Math.max(0, y + m[1]))]);
            }}
          />
        ))}
      </svg>
      <div className="row">
        <button type="button" className="btn primary" onClick={() => onSubmit(q)}>Re-scan with these corners</button>
        {onCancel && <button type="button" className="btn ghost" onClick={onCancel}>Cancel</button>}
      </div>
    </div>
  );
}
