// Optional cloud double-check: sends one photo to the visionCheck Firebase Function for an
// independent piece count from a vision LLM. Firebase is imported lazily so it costs nothing
// unless the user turns the feature on and asks for a check.
import type { FirebaseApp } from 'firebase/app';

export interface VisionCount {
  pieceCount: number;
  studCount: number;
  gridCols: number;
  gridRows: number;
  notes: string;
  model: string;
}

const env = import.meta.env;
const firebaseConfig = {
  apiKey: env.VITE_FIREBASE_API_KEY,
  authDomain: env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: env.VITE_FIREBASE_PROJECT_ID,
  appId: env.VITE_FIREBASE_APP_ID,
};
const recaptchaSiteKey = env.VITE_RECAPTCHA_ENTERPRISE_SITE_KEY;

/** True when this build was configured with a Firebase project and an App Check site key. */
export const cloudCheckAvailable = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId && firebaseConfig.appId && recaptchaSiteKey);

let appPromise: Promise<FirebaseApp> | null = null;

function getApp(): Promise<FirebaseApp> {
  appPromise ??= (async () => {
    const [{ initializeApp }, { initializeAppCheck, ReCaptchaEnterpriseProvider }] = await Promise.all([
      import('firebase/app'),
      import('firebase/app-check'),
    ]);
    const app = initializeApp(firebaseConfig);
    initializeAppCheck(app, { provider: new ReCaptchaEnterpriseProvider(recaptchaSiteKey!), isTokenAutoRefreshEnabled: true });
    return app;
  })().catch((e) => {
    appPromise = null;
    throw e;
  });
  return appPromise;
}

/** Longest side sent to the server; enough for a vision model to see seams, small enough to upload fast. */
const UPLOAD_MAX_SIDE = 1568;

async function toJpegBase64(source: Blob): Promise<string> {
  const bmp = await createImageBitmap(source);
  const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(bmp.width, bmp.height));
  const canvas = new OffscreenCanvas(Math.round(bmp.width * scale), Math.round(bmp.height * scale));
  canvas.getContext('2d')!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

export async function requestVisionCheck(source: Blob): Promise<VisionCount> {
  if (!cloudCheckAvailable) throw new Error('Cloud check is not configured for this build');
  const [app, { getFunctions, httpsCallable }, imageBase64] = await Promise.all([
    getApp(),
    import('firebase/functions'),
    toJpegBase64(source),
  ]);
  // Single-use App Check tokens: the function consumes each token, so a captured request can't be replayed.
  const call = httpsCallable<{ imageBase64: string; mediaType: string }, VisionCount>(getFunctions(app), 'visionCheck', {
    limitedUseAppCheckTokens: true,
    timeout: 300_000,
  });
  const { data } = await call({ imageBase64, mediaType: 'image/jpeg' });
  return data;
}
