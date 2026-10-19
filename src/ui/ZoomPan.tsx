import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent as RPointerEvent, type ReactNode } from 'react';

interface View {
  s: number;
  x: number;
  y: number;
}

const MAX = 8;

/**
 * Pinch / wheel / drag zoom-pan container with a zoom slider. Children are laid out at
 * container width; clicks that end a drag are swallowed so taps stay taps.
 */
export function ZoomPan({ children, label }: { children: ReactNode; label: string }) {
  const box = useRef<HTMLDivElement>(null);
  const [v, setV] = useState<View>({ s: 1, x: 0, y: 0 });
  const vRef = useRef(v);
  useLayoutEffect(() => {
    vRef.current = v;
  }, [v]);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{ dist: number; mid: { x: number; y: number }; start: View; moved: boolean; origin: { x: number; y: number } } | null>(null);
  const suppressClick = useRef(false);

  const clamp = useCallback((n: View): View => {
    const el = box.current;
    const s = Math.min(MAX, Math.max(1, n.s));
    if (!el) return { s, x: n.x, y: n.y };
    const w = el.clientWidth, h = el.clientHeight;
    return { s, x: Math.min(0, Math.max(w - w * s, n.x)), y: Math.min(0, Math.max(h - h * s, n.y)) };
  }, []);

  const zoomAt = useCallback(
    (s2: number, px: number, py: number, from = vRef.current) => {
      const k = Math.min(MAX, Math.max(1, s2)) / from.s;
      setV(clamp({ s: from.s * k, x: px - (px - from.x) * k, y: py - (py - from.y) * k }));
    },
    [clamp],
  );

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(vRef.current.s * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  const local = (e: RPointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const begin = () => {
    const pts = [...pointers.current.values()];
    const mid = pts.length > 1 ? { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 } : pts[0];
    const dist = pts.length > 1 ? Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) : 0;
    gesture.current = { dist, mid, start: vRef.current, moved: gesture.current?.moved ?? false, origin: gesture.current?.origin ?? mid };
  };

  return (
    <div className="zoompan">
      <div
        ref={box}
        className="zoompan-box"
        role="group"
        aria-label={label}
        onPointerDown={(e) => {
          if (e.pointerType === 'mouse' && e.button !== 0) return;
          pointers.current.set(e.pointerId, local(e));
          if (pointers.current.size === 1) gesture.current = null;
          begin();
        }}
        onPointerMove={(e) => {
          if (!pointers.current.has(e.pointerId)) return;
          pointers.current.set(e.pointerId, local(e));
          const g = gesture.current;
          if (!g) return;
          const pts = [...pointers.current.values()];
          if (pts.length >= 2) {
            const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
            const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
            const s2 = g.start.s * (dist / (g.dist || 1));
            const k = Math.min(MAX, Math.max(1, s2)) / g.start.s;
            g.moved = true;
            setV(clamp({ s: g.start.s * k, x: mid.x - (g.mid.x - g.start.x) * k, y: mid.y - (g.mid.y - g.start.y) * k }));
          } else {
            const p = pts[0];
            if (Math.hypot(p.x - g.origin.x, p.y - g.origin.y) > 6) g.moved = true;
            if (g.moved && g.start.s > 1) {
              if (!box.current!.hasPointerCapture(e.pointerId)) box.current!.setPointerCapture(e.pointerId);
              setV(clamp({ ...g.start, x: g.start.x + p.x - g.mid.x, y: g.start.y + p.y - g.mid.y }));
            }
          }
        }}
        onPointerUp={(e) => {
          pointers.current.delete(e.pointerId);
          if (gesture.current?.moved) suppressClick.current = true;
          if (pointers.current.size > 0) begin();
          else setTimeout(() => (suppressClick.current = false), 0);
        }}
        onPointerCancel={(e) => {
          pointers.current.delete(e.pointerId);
          gesture.current = null;
        }}
        onClickCapture={(e) => {
          if (suppressClick.current) {
            e.stopPropagation();
            e.preventDefault();
            suppressClick.current = false;
          }
        }}
      >
        <div className="zoompan-content" style={{ transform: `translate(${v.x}px, ${v.y}px) scale(${v.s})` }}>
          {children}
        </div>
      </div>
      <div className="zoom-controls">
        <button type="button" className="icon-btn" aria-label="Zoom out" onClick={() => zoomCenter(v.s / 1.5)}>−</button>
        <input
          type="range"
          min={1}
          max={MAX}
          step={0.1}
          value={v.s}
          aria-label="Zoom"
          onChange={(e) => zoomCenter(Number(e.target.value))}
        />
        <button type="button" className="icon-btn" aria-label="Zoom in" onClick={() => zoomCenter(v.s * 1.5)}>+</button>
        <button type="button" className="btn small ghost" onClick={() => setV({ s: 1, x: 0, y: 0 })}>Fit</button>
      </div>
    </div>
  );

  function zoomCenter(s: number) {
    const el = box.current;
    if (!el) return;
    zoomAt(s, el.clientWidth / 2, el.clientHeight / 2);
  }
}
