// Generates PWA icons (public/icons/*) and the e2e fixture mosaic PNG.
// Pure Node (zlib) PNG encoder, no deps. Run: node tests/e2e/fixtures/make-images.mjs
import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '../../..');

const crcTable = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function png(w, h, px /* (x,y)=>[r,g,b,a] */) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const [r, g, b, a] = px(x, y);
      const o = y * (w * 4 + 1) + 1 + x * 4;
      raw[o] = r; raw[o + 1] = g; raw[o + 2] = b; raw[o + 3] = a;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0)),
  ]);
}

// --- Icon: red 2x2 brick (top view) on dark rounded tile -------------------
const RED = [201, 26, 9], RED_HI = [232, 70, 50], BG = [24, 24, 27];
function inRoundRect(x, y, x0, y0, x1, y1, r) {
  if (x < x0 || x > x1 || y < y0 || y > y1) return false;
  const cx = Math.min(Math.max(x, x0 + r), x1 - r), cy = Math.min(Math.max(y, y0 + r), y1 - r);
  return (x - cx) ** 2 + (y - cy) ** 2 <= r * r;
}
function icon(size, maskable) {
  // maskable: full-bleed background, content inside the 80% safe zone.
  const s = size;
  const pad = maskable ? 0.22 * s : 0.14 * s;
  return png(s, s, (x0, y0) => {
    const x = x0 + 0.5, y = y0 + 0.5;
    const bgOn = maskable || inRoundRect(x, y, 0, 0, s, s, s * 0.2);
    if (!bgOn) return [0, 0, 0, 0];
    const b0 = pad, b1 = s - pad;
    if (inRoundRect(x, y, b0, b0, b1, b1, s * 0.06)) {
      const step = (b1 - b0) / 2, sr = step * 0.3;
      for (const i of [0, 1]) for (const j of [0, 1]) {
        const cx = b0 + step * (i + 0.5), cy = b0 + step * (j + 0.5);
        const d2 = (x - cx) ** 2 + (y - cy) ** 2;
        if (d2 <= sr * sr) return [...((x - cx) + (y - cy) < 0 ? RED_HI : RED), 255];
        if (d2 <= (sr * 1.15) ** 2) return [150, 18, 6, 255];
      }
      return [...RED, 255];
    }
    return [...BG, 255];
  });
}
const iconDir = join(root, 'public/icons');
mkdirSync(iconDir, { recursive: true });
writeFileSync(join(iconDir, 'icon-192.png'), icon(192, false));
writeFileSync(join(iconDir, 'icon-512.png'), icon(512, false));
writeFileSync(join(iconDir, 'maskable-512.png'), icon(512, true));
writeFileSync(join(iconDir, 'apple-touch-icon.png'), icon(180, true));

// --- Fixture: 16x16 stud mosaic, 480x480 px, few colours in blocks ---------
const cols = 16, pitch = 28, border = 16, W = cols * pitch + border * 2;
const colors = [[201, 26, 9], [0, 85, 191], [242, 205, 55], [244, 244, 244], [27, 42, 52]];
const cellColor = (c, r) => colors[(Math.floor(c / 4) + Math.floor(r / 3) * 2) % colors.length];
const mosaic = png(W, W, (x, y) => {
  const gx = x - border, gy = y - border;
  if (gx < 0 || gy < 0 || gx >= cols * pitch || gy >= cols * pitch) return [40, 40, 40, 255];
  const c = Math.floor(gx / pitch), r = Math.floor(gy / pitch);
  const [R, G, B] = cellColor(c, r);
  const lx = gx % pitch - pitch / 2, ly = gy % pitch - pitch / 2;
  const d = Math.hypot(lx, ly);
  const shade = d < pitch * 0.32 && d > pitch * 0.26 ? 0.75 : 1;
  return [Math.round(R * shade), Math.round(G * shade), Math.round(B * shade), 255];
});
writeFileSync(join(here, 'mosaic16.png'), mosaic);
console.log('icons + fixture written');

// --- Fixture: very wide photo → mock reports grid_failed (corner editor flow) ---
const wide = png(900, 300, (x, y) => {
  if (x < 300 || x >= 600) return [70, 70, 70, 255];
  const c = Math.floor((x - 300) / 18.75), r = Math.floor(y / 18.75);
  return [...colors[(c + r) % colors.length], 255];
});
writeFileSync(join(here, 'wide.png'), wide);
