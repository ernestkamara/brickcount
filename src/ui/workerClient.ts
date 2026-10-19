// Single switch point between the real pipeline worker and the mock.
// Mock when `?mock=1` is in the URL or VITE_MOCK_WORKER is set; otherwise the
// real worker from src/worker/client.ts (loaded through import.meta.glob so the
// build does not break while that file does not exist yet).
import * as Comlink from 'comlink';
import type { WorkerApi } from '../worker/api';
import type { GridSize, PairDecision, ScanOptions, ScanResult, StudCell } from '../pipeline/types';

type ClientModule = { getWorker(): Comlink.Remote<WorkerApi> };

/** Uniform API for the UI; transfer/proxy handling lives here. */
export interface BrickWorker {
  readonly mock: boolean;
  scan(image: ImageBitmap, opts: ScanOptions, onProgress?: (stage: string, frac: number) => void): Promise<ScanResult>;
  regroup: WorkerApi['regroup'];
}

export function wantsMock(): boolean {
  const env = import.meta.env.VITE_MOCK_WORKER;
  if (env && env !== '0' && env !== 'false') return true;
  try {
    return new URLSearchParams(location.search).get('mock') === '1';
  } catch {
    return false;
  }
}

let pending: Promise<BrickWorker> | null = null;

export function getBrickWorker(): Promise<BrickWorker> {
  pending ??= load().catch((e) => {
    pending = null;
    throw e;
  });
  return pending;
}

async function load(): Promise<BrickWorker> {
  if (wantsMock()) {
    const { createMockWorker } = await import('../worker/mock');
    const api = createMockWorker();
    await api.init();
    return {
      mock: true,
      scan: (img, opts, cb) => api.scan(img, opts, cb),
      regroup: (cells, grid, pairs) => api.regroup(cells, grid, pairs),
    };
  }
  const mods = import.meta.glob<ClientModule>('../worker/client.ts');
  const loader = mods['../worker/client.ts'];
  if (!loader) throw new Error('Pipeline worker is not available in this build (src/worker/client.ts missing). Open with ?mock=1 for demo mode.');
  const remote = (await loader()).getWorker();
  await remote.init();
  return {
    mock: false,
    scan: (img, opts, cb) =>
      remote.scan(Comlink.transfer(img, [img]), opts, cb ? Comlink.proxy(cb) : undefined) as Promise<ScanResult>,
    regroup: (cells: StudCell[], grid: GridSize, pairs: PairDecision[]) => remote.regroup(cells, grid, pairs),
  };
}
