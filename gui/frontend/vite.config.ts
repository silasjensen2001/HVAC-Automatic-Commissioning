import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({ include: ['buffer', 'process'] }),
  ],
  server: { port: 5173 },
  optimizeDeps: {
    include: ['react-plotly.js', 'plotly.js'],
  },
})
