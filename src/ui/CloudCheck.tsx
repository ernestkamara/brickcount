import { useState } from 'react';
import { requestVisionCheck, type VisionCount } from '../lib/cloudCheck';

type State = { kind: 'idle' } | { kind: 'busy' } | { kind: 'done'; v: VisionCount } | { kind: 'error'; message: string };

/** On-demand second opinion from the cloud vision check; never replaces the on-device count. */
export function CloudCheck({ source, pieceCount }: { source: Blob; pieceCount: number }) {
  const [state, setState] = useState<State>({ kind: 'idle' });

  const run = async () => {
    setState({ kind: 'busy' });
    try {
      setState({ kind: 'done', v: await requestVisionCheck(source) });
    } catch (e) {
      setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  };

  return (
    <section className="cloud-check" aria-label="Cloud double-check">
      {state.kind === 'idle' && (
        <button type="button" className="btn ghost" onClick={run}>
          Get a cloud second opinion
        </button>
      )}
      {state.kind === 'busy' && <p className="hint" role="status">Asking the cloud model… this can take a minute.</p>}
      {state.kind === 'error' && (
        <p className="alert err" role="alert">
          Cloud check failed: {state.message}{' '}
          <button type="button" className="btn ghost small" onClick={run}>Retry</button>
        </p>
      )}
      {state.kind === 'done' && <Verdict v={state.v} pieceCount={pieceCount} />}
    </section>
  );
}

function Verdict({ v, pieceCount }: { v: VisionCount; pieceCount: number }) {
  const diff = v.pieceCount - pieceCount;
  const agrees = Math.abs(diff) <= Math.max(1, Math.round(pieceCount * 0.03));
  return (
    <div className={agrees ? 'alert ok' : 'alert warn'} role="status">
      <p>
        Cloud model counted <strong>{v.pieceCount}</strong> pieces ({v.gridCols}×{v.gridRows} grid, {v.studCount} studs)
        {agrees ? ' — agrees with this device.' : ` — ${diff > 0 ? `${diff} more` : `${-diff} fewer`} than this device. Worth reviewing the amber seams.`}
      </p>
      {v.notes && <p className="hint">{v.notes}</p>}
      <p className="hint">Vision models are a cross-check only; they are less reliable than the on-device count for exact totals.</p>
    </div>
  );
}
