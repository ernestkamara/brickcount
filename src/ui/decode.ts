import { heicToJpeg, isHeic } from '../lib/heic';

export const MAX_SIDE = 4096;

export function fitSize(w: number, h: number, max = MAX_SIDE): { width: number; height: number } {
  const s = Math.min(1, max / Math.max(w, h));
  return { width: Math.max(1, Math.round(w * s)), height: Math.max(1, Math.round(h * s)) };
}

/** Decode a blob to an ImageBitmap downscaled to ≤ MAX_SIDE (EXIF orientation applied). */
export async function decodeScaled(blob: Blob): Promise<ImageBitmap> {
  const full = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const { width, height } = fitSize(full.width, full.height);
  if (width === full.width && height === full.height) return full;
  try {
    return await createImageBitmap(full, { resizeWidth: width, resizeHeight: height, resizeQuality: 'high' });
  } finally {
    full.close();
  }
}

export function isAcceptedImage(f: File): boolean {
  return f.type.startsWith('image/') || isHeic(f) || /\.(jpe?g|png|webp|heic|heif)$/i.test(f.name);
}

/** Prepare an added file: HEIC→JPEG, display JPEG at worker resolution. */
export async function prepareFile(file: File): Promise<{ source: Blob; url: string; width: number; height: number }> {
  const source = isHeic(file) ? await heicToJpeg(file) : file;
  const bmp = await decodeScaled(source);
  const { width, height } = bmp;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d')!.drawImage(bmp, 0, 0);
  bmp.close();
  const blob = await new Promise<Blob>((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error('Could not encode preview'))), 'image/jpeg', 0.9),
  );
  return { source, url: URL.createObjectURL(blob), width, height };
}
