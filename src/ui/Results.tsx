import { useMemo } from 'react';
import type { Breakdown, Piece, ScanResult } from '../pipeline/types';
import { PALETTE_BY_ID } from '../lib/palette';
import { compareSize, computeBreakdown } from '../lib/breakdown';
import { partsList } from '../lib/export';

export function swatch(color: string): string {
  const c = PALETTE_BY_ID[color];
  return c ? `rgb(${c.rgb.join(',')})` : '#888';
}
export const colorName = (id: string) => PALETTE_BY_ID[id]?.name ?? id;

export function Results({ result }: { result: ScanResult }) {
  const b = useMemo(() => computeBreakdown(result.pieces ?? []), [result.pieces]);
  return (
    <section className="results" aria-label="Results">
      <div className="stats">
        <div className="stat big">
          <span className="num" data-testid="piece-count" aria-live="polite">{result.pieceCount}</span>
          <span className="lbl">pieces</span>
        </div>
        <div className="stat">
          <span className="num">{result.studCount}</span>
          <span className="lbl">studs</span>
        </div>
        <div className={result.lowConfidenceCount ? 'stat warn' : 'stat'}>
          <span className="num">{result.lowConfidenceCount}</span>
          <span className="lbl">to check</span>
        </div>
        {result.grid && (
          <div className="stat">
            <span className="num">{result.grid.cols}×{result.grid.rows}</span>
            <span className="lbl">grid</span>
          </div>
        )}
      </div>
      <div className="breakdowns">
        <ColorBreakdown b={b} />
        <SizeBreakdown b={b} />
        {result.grid && result.pieces && <GridPreview cols={result.grid.cols} rows={result.grid.rows} pieces={result.pieces} />}
      </div>
    </section>
  );
}

export function ColorBreakdown({ b }: { b: Pick<Breakdown, 'byColor'> }) {
  const rows = Object.entries(b.byColor).sort((x, y) => y[1] - x[1]);
  return (
    <div className="card-sub">
      <h3>By colour</h3>
      <ul className="bars">
        {rows.map(([id, n]) => (
          <li key={id}>
            <span className="sw" style={{ background: swatch(id) }} aria-hidden />
            <span className="name">{colorName(id)}</span>
            <span className="qty">{n}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SizeBreakdown({ b }: { b: Pick<Breakdown, 'bySize'> }) {
  const rows = Object.entries(b.bySize).sort((x, y) => compareSize(x[0], y[0]));
  return (
    <div className="card-sub">
      <h3>By size</h3>
      <ul className="bars">
        {rows.map(([size, n]) => (
          <li key={size}>
            <span className="name">{size}</span>
            <span className="qty">{n}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PartsTable({ byColorSize }: { byColorSize: Record<string, number> }) {
  const rows = partsList(byColorSize);
  return (
    <div className="table-wrap">
      <table className="parts">
        <thead>
          <tr><th scope="col">Colour</th><th scope="col">Size</th><th scope="col" className="r">Qty</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.color}|${r.size}`}>
              <td><span className="sw" style={{ background: swatch(r.color) }} aria-hidden /> {r.colorName}</td>
              <td>{r.size}</td>
              <td className="r">{r.qty}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Rectified schematic: pieces drawn on the stud grid in palette colours. */
export function GridPreview({ cols, rows, pieces }: { cols: number; rows: number; pieces: Piece[] }) {
  return (
    <div className="card-sub">
      <h3>Grid preview</h3>
      <svg className="grid-preview" viewBox={`0 0 ${cols} ${rows}`} role="img" aria-label={`${cols} by ${rows} stud grid with ${pieces.length} pieces`}>
        {pieces.map((p) => (
          <rect key={`${p.x},${p.y}`} x={p.x + 0.04} y={p.y + 0.04} width={p.w - 0.08} height={p.h - 0.08} rx={0.12} fill={swatch(p.color)} className={p.confidence < 0.6 ? 'low' : undefined} />
        ))}
      </svg>
    </div>
  );
}
