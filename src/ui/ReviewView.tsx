import { memo, useMemo } from 'react';
import type { PairDecision, Piece, ScanResult } from '../pipeline/types';
import { gridMapper, polygonPoints } from './geometry';
import { ZoomPan } from './ZoomPan';

const LOW = 0.6;

interface Props {
  url: string;
  width: number;
  height: number;
  result: ScanResult;
  onToggle(col: number, row: number, dir: 'h' | 'v'): void;
}

/** Photo with every piece outlined and numbered; tap a seam to split/merge. */
export function ReviewView(props: Props) {
  return (
    <ZoomPan label="Review: pinch or scroll to zoom, tap a seam between studs to split or merge pieces">
      <Overlay {...props} />
    </ZoomPan>
  );
}

const Overlay = memo(function Overlay({ url, width, height, result, onToggle }: Props) {
  const grid = result.grid!;
  const map = useMemo(() => gridMapper(grid.corners, grid.cols, grid.rows), [grid]);
  const pitch = useMemo(() => {
    const [a, b] = [map(0, 0), map(grid.cols, 0)];
    return Math.hypot(b[0] - a[0], b[1] - a[1]) / grid.cols;
  }, [map, grid.cols]);

  return (
    <svg
      className="review-svg"
      viewBox={`0 0 ${width} ${height}`}
      style={{ ['--pitch' as string]: `${pitch}px` }}
      onClick={(e) => {
        const t = (e.target as Element).closest('[data-pair]');
        if (!t) return;
        const [c, r, d] = t.getAttribute('data-pair')!.split(',');
        onToggle(Number(c), Number(r), d as 'h' | 'v');
      }}
    >
      <image href={url} width={width} height={height} />
      <Pieces pieces={result.pieces ?? []} map={map} pitch={pitch} />
      <Seams pairs={result.pairs ?? []} map={map} pitch={pitch} />
    </svg>
  );
});

type Mapper = ReturnType<typeof gridMapper>;

function Pieces({ pieces, map, pitch }: { pieces: Piece[]; map: Mapper; pitch: number }) {
  return (
    <g className="pieces" style={{ strokeWidth: Math.max(1, pitch * 0.07) }}>
      {pieces.map((p) => (
        <polygon
          key={`${p.x},${p.y},${p.w},${p.h}`}
          className={p.confidence < LOW ? 'piece low' : 'piece'}
          points={polygonPoints([map(p.x, p.y), map(p.x + p.w, p.y), map(p.x + p.w, p.y + p.h), map(p.x, p.y + p.h)])}
        />
      ))}
      <g className="piece-labels" style={{ fontSize: Math.max(6, pitch * 0.4), strokeWidth: Math.max(1, pitch * 0.09) }}>
        {pieces.map((p) => {
          const [x, y] = map(p.x + p.w / 2, p.y + p.h / 2);
          return (
            <text key={p.id} x={x} y={y}>
              {p.id}
            </text>
          );
        })}
      </g>
    </g>
  );
}

function Seams({ pairs, map, pitch }: { pairs: PairDecision[]; map: Mapper; pitch: number }) {
  return (
    <g className="seams" style={{ strokeWidth: Math.max(4, pitch * 0.38) }}>
      {pairs.map((p) => {
        // boundary between the two studs, in grid units
        const [a, b] = p.dir === 'h' ? [map(p.col + 1, p.row + 0.12), map(p.col + 1, p.row + 0.88)] : [map(p.col + 0.12, p.row + 1), map(p.col + 0.88, p.row + 1)];
        const cls = ['seam', p.seam ? 'on' : 'off', p.manual ? 'manual' : p.confidence < LOW ? 'low' : ''].join(' ');
        return (
          <line
            key={`${p.col},${p.row},${p.dir}`}
            data-pair={`${p.col},${p.row},${p.dir}`}
            data-seam={p.seam ? '1' : '0'}
            className={cls}
            x1={a[0]}
            y1={a[1]}
            x2={b[0]}
            y2={b[1]}
          />
        );
      })}
    </g>
  );
}
