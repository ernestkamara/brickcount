// Load OpenCV.js in Node tests via require (vitest's ESM interop turns the module's
// exported Promise into a broken thenable namespace).
import { createRequire } from 'node:module';
import { resolveCv, type CV } from '../../src/pipeline/cv';

export async function loadOpenCvNode(): Promise<CV> {
  const require = createRequire(import.meta.url);
  return resolveCv(require('@techstark/opencv-js'));
}
