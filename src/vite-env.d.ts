/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

interface ImportMetaEnv {
  /** Set to 1/true to use the mock worker (src/worker/mock.ts). */
  readonly VITE_MOCK_WORKER?: string;
}
