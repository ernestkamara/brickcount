import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: false, // registered in src/main.tsx via virtual:pwa-register
      includeAssets: ['favicon.svg', 'icons/apple-touch-icon.png'],
      manifest: {
        name: 'BrickCount',
        short_name: 'BrickCount',
        description: 'Count the LEGO pieces in photos of flat mosaics, on-device.',
        theme_color: '#c91a09',
        background_color: '#111113',
        display: 'standalone',
        start_url: '/',
        scope: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: '/icons/maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
          { src: '/favicon.svg', sizes: 'any', type: 'image/svg+xml' },
        ],
      },
      workbox: {
        // App shell + worker chunk + OpenCV/ONNX wasm + models, so it works offline after first load.
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,ico,webmanifest,wasm,onnx,json}'],
        maximumFileSizeToCacheInBytes: 20 * 1024 * 1024,
        navigateFallback: '/index.html',
        cleanupOutdatedCaches: true,
      },
    }),
  ],
  worker: {
    format: 'es',
  },
  assetsInclude: ['**/*.onnx'],
  build: { chunkSizeWarningLimit: 2000 },
});
