import { useSyncExternalStore } from 'react';
import type { Quad, ScanResult } from '../pipeline/types';
import { loadSettings, saveSettings, type Settings } from './settings';

export type PhotoState = 'queued' | 'processing' | 'done' | 'grid_failed' | 'error';

export interface PhotoItem {
  id: string;
  name: string;
  /** Decodable source (HEIC already converted to JPEG). Kept for re-scans. */
  source: Blob;
  /** Object URL of a ≤1600 px JPEG matching the worker's image coordinates. */
  url: string;
  width: number;
  height: number;
  state: PhotoState;
  stage?: string;
  frac?: number;
  message?: string;
  /** Corners from the corner editor for the next scan. */
  corners?: Quad;
  result?: ScanResult;
  /** Bumped on every result change (seam edits) to memoise derived data. */
  rev: number;
}

export interface AppState {
  photos: PhotoItem[];
  settings: Settings;
}

let state: AppState = { photos: [], settings: loadSettings() };
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export function getState(): AppState {
  return state;
}

export function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useAppState<T>(select: (s: AppState) => T): T {
  return useSyncExternalStore(subscribe, () => select(state));
}

export function addPhotos(items: PhotoItem[]) {
  state = { ...state, photos: [...state.photos, ...items] };
  emit();
}

export function updatePhoto(id: string, patch: Partial<PhotoItem> | ((p: PhotoItem) => Partial<PhotoItem>)) {
  state = {
    ...state,
    photos: state.photos.map((p) => (p.id === id ? { ...p, ...(typeof patch === 'function' ? patch(p) : patch) } : p)),
  };
  emit();
}

export function removePhoto(id: string) {
  const p = state.photos.find((x) => x.id === id);
  if (p) URL.revokeObjectURL(p.url);
  state = { ...state, photos: state.photos.filter((x) => x.id !== id) };
  emit();
}

export function clearPhotos() {
  for (const p of state.photos) URL.revokeObjectURL(p.url);
  state = { ...state, photos: [] };
  emit();
}

export function setSettings(s: Settings) {
  saveSettings(s);
  state = { ...state, settings: s };
  emit();
}

export function getPhoto(id: string): PhotoItem | undefined {
  return state.photos.find((p) => p.id === id);
}
