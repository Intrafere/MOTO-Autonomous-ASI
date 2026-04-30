import { defineConfig, loadEnv, createLogger } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendPort = env.MOTO_BACKEND_PORT || env.PORT || '8000'
  const backendUrl = env.VITE_MOTO_BACKEND_URL || `http://localhost:${backendPort}`
  const backendWsUrl = env.VITE_MOTO_BACKEND_WS_URL || backendUrl.replace(/^http/i, 'ws')
  const frontendPort = Number(env.VITE_MOTO_FRONTEND_PORT || env.MOTO_FRONTEND_PORT || env.FRONTEND_PORT || 5173)
  const frontendHost = env.VITE_MOTO_FRONTEND_HOST || '0.0.0.0'

  const logger = createLogger()
  const originalError = logger.error.bind(logger)
  logger.error = (msg, options) => {
    if (typeof msg === 'string' && /proxy error/i.test(msg) && /ECONNREFUSED|ECONNRESET|ETIMEDOUT/i.test(msg)) {
      return
    }
    originalError(msg, options)
  }

  return {
    customLogger: logger,
    plugins: [react()],
    server: {
      host: frontendHost,
      port: frontendPort,
      strictPort: true,
      open: false,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          configure: (proxy) => {
            proxy.on('error', (err, _req, res) => {
              const code = err && err.code
              if (code === 'ECONNREFUSED' || code === 'ECONNRESET' || code === 'ETIMEDOUT') {
                if (res && !res.headersSent && typeof res.writeHead === 'function') {
                  try {
                    res.writeHead(503, { 'Content-Type': 'application/json' })
                    res.end('{"error":"backend_unavailable"}')
                  } catch (_) {}
                }
              }
            })
          },
        },
        '/ws': {
          target: backendWsUrl,
          ws: true,
          configure: (proxy) => {
            proxy.on('error', () => {})
          },
        },
      },
    },
  }
})

