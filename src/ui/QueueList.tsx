import type { PhotoItem } from '../lib/store';

export function StatusChip({ p }: { p: PhotoItem }) {
  switch (p.state) {
    case 'queued':
      return <span className="chip">Queued</span>;
    case 'processing':
      return (
        <span className="chip busy">
          {stageLabel(p.stage)}
          {p.frac ? ` ${Math.round(p.frac * 100)}%` : ''}
        </span>
      );
    case 'done':
      return <span className="chip ok">Done</span>;
    case 'grid_failed':
      return <span className="chip warn">Grid not found</span>;
    case 'error':
      return <span className="chip err">Error</span>;
  }
}

function stageLabel(s?: string): string {
  if (!s) return 'Working';
  return s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ');
}

export function QueueList({ photos, onOpen }: { photos: PhotoItem[]; onOpen(id: string): void }) {
  return (
    <ul className="queue" aria-label="Photos">
      {photos.map((p) => (
        <li key={p.id}>
          <button type="button" className="qcard" onClick={() => onOpen(p.id)} data-testid="photo-card" aria-label={`${p.name}: ${summary(p)}`}>
            <img src={p.url} alt="" width={64} height={64} />
            <span className="qmeta">
              <span className="qname">{p.name}</span>
              <span aria-live="polite">
                <StatusChip p={p} />
              </span>
              {p.state === 'processing' && <progress max={1} value={p.frac ?? 0} />}
            </span>
            {p.state === 'done' && p.result && (
              <span className="qcount">
                <strong data-testid="card-count">{p.result.pieceCount}</strong>
                <small>pieces</small>
              </span>
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}

function summary(p: PhotoItem): string {
  if (p.state === 'done' && p.result) return `${p.result.pieceCount} pieces`;
  if (p.state === 'grid_failed') return 'grid not found, tap to adjust corners';
  if (p.state === 'error') return 'error';
  return p.state;
}
