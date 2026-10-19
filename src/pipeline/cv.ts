// OpenCV.js typing + loader shared by the worker, Node eval and tests.
// The @techstark/opencv-js typings are incomplete/unshipped, so we use a loose type.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type CV = any;

/**
 * Resolve the various shapes the opencv.js module can take (Promise, ready object,
 * or Emscripten module awaiting onRuntimeInitialized).
 */
export async function resolveCv(mod: unknown): Promise<CV> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let m: any = mod;
  if (m && typeof m === 'object' && 'default' in m && !m.Mat) m = m.default;
  if (m instanceof Promise) m = await m;
  if (m && m.Mat) return m;
  if (m && typeof m.then === 'function') {
    // Emscripten modules can be self-resolving thenables: awaiting them directly loops.
    return new Promise((resolve) => {
      m.then((ready: CV) => {
        const r = ready ?? m;
        if (r && typeof r.then === 'function') delete r.then;
        resolve(r);
      });
    });
  }
  if (!m) throw new Error('OpenCV.js module not found');
  await new Promise<void>((resolve) => {
    m.onRuntimeInitialized = () => resolve();
  });
  return m;
}

/** Dynamically import and initialise OpenCV.js. */
export async function loadOpenCv(): Promise<CV> {
  const mod = await import('@techstark/opencv-js');
  return resolveCv(mod);
}
