import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import { resolve } from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [tailwindcss(), react()],
  build: {
    rollupOptions: {
      // The hosted Vercel build is a product site. The local Windows bundle
      // also includes the diagnostic UI, which needs the on-device API.
      input: process.env.VERCEL
        ? { site: resolve(import.meta.dirname, 'index.html') }
        : {
            site: resolve(import.meta.dirname, 'index.html'),
            app: resolve(import.meta.dirname, 'app/index.html'),
          },
    },
  },
  server: {
    host: '127.0.0.1',
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.CURIO_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
