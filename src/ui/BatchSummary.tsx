import type { PhotoItem } from '../lib/store';
import { computeBreakdown, mergeCounts } from '../lib/breakdown';
import { shareOrDownload, toCsv, toJson } from '../lib/export';
import { PartsTable } from './Results';

export function BatchSummary({ photos }: { photos: PhotoItem[] }) {
  const done = photos.filter((p) => p.state === 'done' && p.result);
  const totals = (() => {
    const byColorSize: Record<string, number> = {};
    let pieces = 0, studs = 0, low = 0;
    for (const p of done) {
      const r = p.result!;
      pieces += r.pieceCount;
      studs += r.studCount;
      low += r.lowConfidenceCount;
      mergeCounts(byColorSize, computeBreakdown(r.pieces ?? []).byColorSize);
    }
    return { pieces, studs, low, byColorSize };
  })();

  if (!done.length) return null;
  const data = done.map((p) => ({ name: p.name, result: p.result! }));
  const pending = photos.length - done.length;

  return (
    <section className="batch card" aria-labelledby="batch-title">
      <h2 id="batch-title">Batch total</h2>
      <div className="stats">
        <div className="stat big">
          <span className="num" data-testid="batch-count">{totals.pieces}</span>
          <span className="lbl">pieces in {done.length} photo{done.length > 1 ? 's' : ''}</span>
        </div>
        <div className="stat"><span className="num">{totals.studs}</span><span className="lbl">studs</span></div>
        <div className={totals.low ? 'stat warn' : 'stat'}><span className="num">{totals.low}</span><span className="lbl">to check</span></div>
      </div>
      {pending > 0 && <p className="hint">{pending} photo{pending > 1 ? 's' : ''} not included (still scanning or need attention).</p>}
      <details>
        <summary>Combined parts list</summary>
        <PartsTable byColorSize={totals.byColorSize} />
      </details>
      <div className="row wrap">
        <button type="button" className="btn" data-testid="batch-csv" onClick={() => void shareOrDownload(new Blob([toCsv(data)], { type: 'text/csv' }), 'brickcount-batch.csv')}>
          Export batch CSV
        </button>
        <button type="button" className="btn" onClick={() => void shareOrDownload(new Blob([toJson(data)], { type: 'application/json' }), 'brickcount-batch.json')}>
          Export batch JSON
        </button>
      </div>
    </section>
  );
}
