/** HEIC/HEIF detection and lazy conversion to JPEG (heic2any is ~1 MB, loaded on demand). */
export function isHeic(file: File | Blob): boolean {
  const type = file.type.toLowerCase();
  if (type === 'image/heic' || type === 'image/heif' || type === 'image/heic-sequence' || type === 'image/heif-sequence') return true;
  const name = 'name' in file ? (file as File).name.toLowerCase() : '';
  return name.endsWith('.heic') || name.endsWith('.heif');
}

export async function heicToJpeg(file: Blob): Promise<Blob> {
  // Safari decodes HEIC natively; try that first to skip the converter.
  try {
    const bmp = await createImageBitmap(file);
    bmp.close();
    return file;
  } catch {
    /* fall through to heic2any */
  }
  const { default: heic2any } = await import('heic2any');
  const out = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.95 });
  return Array.isArray(out) ? out[0] : out;
}
