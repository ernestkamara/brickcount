// Main-thread handle to the pipeline worker (singleton).
import * as Comlink from 'comlink';
import type { WorkerApi } from './api';

let remote: Comlink.Remote<WorkerApi> | null = null;

export function getWorker(): Comlink.Remote<WorkerApi> {
  if (!remote) {
    const worker = new Worker(new URL('./pipeline.worker.ts', import.meta.url), { type: 'module' });
    remote = Comlink.wrap<WorkerApi>(worker);
  }
  return remote;
}
