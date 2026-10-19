// Sequential processing queue: one photo in the worker at a time so the UI stays responsive.
import type { PairDecision, Quad } from '../pipeline/types';
import { scanOptions } from '../lib/settings';
import { addPhotos, getPhoto, getState, updatePhoto, type PhotoItem } from '../lib/store';
import { decodeScaled, isAcceptedImage, prepareFile } from './decode';
import { getBrickWorker } from './workerClient';

let seq = 0;
let running = false;

export async function addFiles(files: Iterable<File>): Promise<number> {
  let added = 0;
  for (const f of files) {
    if (!isAcceptedImage(f)) continue;
    try {
      const prep = await prepareFile(f);
      const item: PhotoItem = { id: `p${Date.now().toString(36)}${seq++}`, name: f.name || `Photo ${seq}`, ...prep, state: 'queued', rev: 0 };
      addPhotos([item]);
      added++;
      void pump();
    } catch (e) {
      console.error('Could not read image', f.name, e);
      alert(`Couldn't read "${f.name}". Try a JPEG, PNG or HEIC photo.`);
    }
  }
  return added;
}

export function rescan(id: string, corners?: Quad) {
  updatePhoto(id, { state: 'queued', stage: undefined, frac: undefined, message: undefined, result: undefined, ...(corners ? { corners } : {}) });
  void pump();
}

async function pump() {
  if (running) return;
  running = true;
  try {
    for (;;) {
      const next = getState().photos.find((p) => p.state === 'queued');
      if (!next) break;
      await processOne(next.id);
    }
  } finally {
    running = false;
  }
}

async function processOne(id: string) {
  const item = getPhoto(id);
  if (!item) return;
  updatePhoto(id, { state: 'processing', stage: 'loading', frac: 0 });
  try {
    const worker = await getBrickWorker();
    const bmp = await decodeScaled(item.source);
    const opts = { ...scanOptions(getState().settings), ...(item.corners ? { corners: item.corners } : {}) };
    const result = await worker.scan(bmp, opts, (stage, frac) => {
      if (getPhoto(id)?.state === 'processing') updatePhoto(id, { stage, frac });
    });
    if (!getPhoto(id)) return; // removed meanwhile
    if (result.status === 'ok') {
      updatePhoto(id, (p) => ({ state: 'done', result, stage: undefined, frac: 1, rev: p.rev + 1 }));
    } else if (result.status === 'grid_failed') {
      updatePhoto(id, (p) => ({ state: 'grid_failed', result, message: result.message, rev: p.rev + 1 }));
    } else {
      updatePhoto(id, { state: 'error', result: undefined, message: result.message ?? 'Scan failed' });
    }
  } catch (e) {
    updatePhoto(id, { state: 'error', message: e instanceof Error ? e.message : String(e) });
  }
}

const editSeq = new Map<string, number>();

/** Flip one seam decision (manual) and regroup in the worker; updates counts live. */
export async function toggleSeam(id: string, col: number, row: number, dir: 'h' | 'v') {
  const item = getPhoto(id);
  const r = item?.result;
  if (!item || !r?.pairs || !r.cells || !r.grid) return;
  const pairs: PairDecision[] = r.pairs.map((p) =>
    p.col === col && p.row === row && p.dir === dir ? { ...p, seam: !p.seam, manual: true, confidence: 1 } : p,
  );
  const n = (editSeq.get(id) ?? 0) + 1;
  editSeq.set(id, n);
  updatePhoto(id, (p) => ({ result: { ...r, pairs }, rev: p.rev + 1 }));
  const worker = await getBrickWorker();
  const g = await worker.regroup(r.cells, { cols: r.grid.cols, rows: r.grid.rows }, pairs);
  if (editSeq.get(id) !== n) return; // a newer edit superseded this one
  updatePhoto(id, (p) => (p.result ? { result: { ...p.result, ...g, breakdown: undefined }, rev: p.rev + 1 } : {}));
}
