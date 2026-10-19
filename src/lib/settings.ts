import type { GridSize, ScanOptions, SeamClassifierName } from '../pipeline/types';

export type GridChoice = 'auto' | '16' | '32' | '48' | 'custom';

export interface Settings {
  grid: GridChoice;
  customCols: number;
  customRows: number;
  countBaseplate: boolean;
  classifier: SeamClassifierName;
  cloudCheck: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  grid: 'auto',
  customCols: 32,
  customRows: 32,
  countBaseplate: true,
  classifier: 'cnn',
  cloudCheck: false,
};

const KEY = 'brickcount.settings.v1';

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable (private mode, quota) — keep in memory only */
  }
}

export function clearSettings(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export function gridOverride(s: Settings): GridSize | undefined {
  if (s.grid === 'auto') return undefined;
  if (s.grid === 'custom') {
    const cols = Math.max(1, Math.min(256, Math.round(s.customCols) || 1));
    const rows = Math.max(1, Math.min(256, Math.round(s.customRows) || 1));
    return { cols, rows };
  }
  const n = Number(s.grid);
  return { cols: n, rows: n };
}

export function scanOptions(s: Settings): ScanOptions {
  const o = gridOverride(s);
  return {
    classifier: s.classifier,
    countBaseplate: s.countBaseplate,
    maxSide: 1600,
    ...(o ? { gridOverride: o } : {}),
  };
}
