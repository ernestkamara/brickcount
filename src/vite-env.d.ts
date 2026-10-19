/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

interface ImportMetaEnv {
  /** Set to 1/true to use the mock worker (src/worker/mock.ts). */
  readonly VITE_MOCK_WORKER?: string;
  /** Firebase web config for the optional cloud double-check (public identifiers, not secrets). */
  readonly VITE_FIREBASE_API_KEY?: string;
  readonly VITE_FIREBASE_AUTH_DOMAIN?: string;
  readonly VITE_FIREBASE_PROJECT_ID?: string;
  readonly VITE_FIREBASE_APP_ID?: string;
  /** reCAPTCHA Enterprise site key used by Firebase App Check. */
  readonly VITE_RECAPTCHA_ENTERPRISE_SITE_KEY?: string;
}
