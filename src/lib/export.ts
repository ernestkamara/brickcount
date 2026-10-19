import type { Piece, ScanResult } from '../pipeline/types';
import { PALETTE_BY_ID } from './palette';
import { compareSize, computeBreakdown, mergeCounts } from './breakdown';
import { gridMapper } from '../ui/geometry';

export interface PhotoExport {
  name: string;
  result: ScanResult;
}

export interface PartRow {
  color: string;
  colorName: string;
  size: string;
  qty: number;
}

export function partsList(byColorSize: Record<string, number>): PartRow[] {
  return Object.entries(byColorSize)
    .map(([k, qty]) => {
      const [color, size] = k.split('|');
      return { color, colorName: PALETTE_BY_ID[color]?.name ?? color, size, qty };
    })
    .sort((a, b) => a.colorName.localeCompare(b.colorName) || compareSize(a.size, b.size));
}

function csvCell(v: string | number): string {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** CSV with one row per (photo, colour, size) plus BATCH TOTAL rows when more than one photo. */
export function toCsv(photos: PhotoExport[]): string {
  const lines = [['photo', 'color_id', 'color_name', 'size', 'quantity'].join(',')];
  const total: Record<string, number> = {};
  for (const p of photos) {
    const b = computeBreakdown(p.result.pieces ?? []);
    mergeCounts(total, b.byColorSize);
    for (const r of partsList(b.byColorSize)) lines.push([p.name, r.color, r.colorName, r.size, r.qty].map(csvCell).join(','));
  }
  if (photos.length > 1) {
    for (const r of partsList(total)) lines.push(['BATCH TOTAL', r.color, r.colorName, r.size, r.qty].map(csvCell).join(','));
  }
  return lines.join('\n') + '\n';
}

export function toJson(photos: PhotoExport[]): string {
  const total: Record<string, number> = {};
  const out = photos.map((p) => {
    const pieces = p.result.pieces ?? [];
    const b = computeBreakdown(pieces);
    mergeCounts(total, b.byColorSize);
    return {
      photo: p.name,
      pieceCount: p.result.pieceCount,
      studCount: p.result.studCount,
      lowConfidenceCount: p.result.lowConfidenceCount,
      grid: p.result.grid ? { cols: p.result.grid.cols, rows: p.result.grid.rows } : null,
      classifier: p.result.classifier,
      parts: partsList(b.byColorSize),
      pieces: pieces.map(({ id, x, y, w, h, color, size, confidence }) => ({ id, x, y, w, h, color, size, confidence: +confidence.toFixed(3) })),
    };
  });
  const batch = {
    photoCount: photos.length,
    pieceCount: photos.reduce((s, p) => s + p.result.pieceCount, 0),
    studCount: photos.reduce((s, p) => s + p.result.studCount, 0),
    parts: partsList(total),
  };
  return JSON.stringify({ app: 'BrickCount', exportedAt: new Date().toISOString(), batch, photos: out }, null, 2);
}

/** Photo with piece outlines and numbers, as PNG. */
export async function annotatedPng(imageUrl: string, result: ScanResult): Promise<Blob> {
  const img = new Image();
  img.src = imageUrl;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const grid = result.grid;
  if (grid) drawPieces(ctx, grid.corners, grid.cols, grid.rows, result.pieces ?? []);
  return new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('PNG encode failed'))), 'image/png'));
}

function drawPieces(ctx: CanvasRenderingContext2D, corners: NonNullable<ScanResult['grid']>['corners'], cols: number, rows: number, pieces: Piece[]) {
  const map = gridMapper(corners, cols, rows);
  const [a, b] = [map(0, 0), map(1, 0)];
  const pitch = Math.hypot(b[0] - a[0], b[1] - a[1]);
  ctx.lineJoin = 'round';
  for (const p of pieces) {
    const pts = [map(p.x, p.y), map(p.x + p.w, p.y), map(p.x + p.w, p.y + p.h), map(p.x, p.y + p.h)];
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    if (p.confidence < 0.6) {
      ctx.fillStyle = 'rgba(255,176,0,0.35)';
      ctx.fill();
    }
    ctx.strokeStyle = 'rgba(0,0,0,0.75)';
    ctx.lineWidth = Math.max(2, pitch * 0.12);
    ctx.stroke();
    ctx.strokeStyle = p.confidence < 0.6 ? '#ffb000' : '#ffffff';
    ctx.lineWidth = Math.max(1, pitch * 0.05);
    ctx.stroke();
  }
  ctx.font = `600 ${Math.max(8, pitch * 0.42)}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = Math.max(2, pitch * 0.1);
  for (const p of pieces) {
    const [x, y] = map(p.x + p.w / 2, p.y + p.h / 2);
    ctx.strokeStyle = 'rgba(0,0,0,0.8)';
    ctx.strokeText(String(p.id), x, y);
    ctx.fillStyle = '#fff';
    ctx.fillText(String(p.id), x, y);
  }
}

/** Share via Web Share API with files when available, otherwise download. */
export async function shareOrDownload(blob: Blob, filename: string): Promise<'shared' | 'downloaded' | 'cancelled'> {
  const file = new File([blob], filename, { type: blob.type });
  const nav = navigator as Navigator & { canShare?: (d: ShareData) => boolean };
  if (typeof nav.share === 'function' && nav.canShare?.({ files: [file] })) {
    try {
      await nav.share({ files: [file], title: filename });
      return 'shared';
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return 'cancelled';
      /* fall back to download */
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return 'downloaded';
}

export function safeName(name: string): string {
  return name.replace(/\.[a-z0-9]+$/i, '').replace(/[^\w.-]+/g, '_').slice(0, 60) || 'photo';
}
