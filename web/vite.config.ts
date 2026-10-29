import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 3000, host: '127.0.0.1' },
  preview: { port: 3000, host: '127.0.0.1' },
  test: { environment: 'node' },
});
