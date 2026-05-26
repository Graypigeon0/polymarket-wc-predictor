import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'WC 2026 Edge Engine',
  description: 'Live Polymarket edges for the 2026 World Cup',
  manifest: '/manifest.json'
}

export const viewport: Viewport = {
  themeColor: '#0a0a0a',
  width: 'device-width',
  initialScale: 1
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main className="max-w-md mx-auto p-4 min-h-screen">{children}</main>
        <script
          dangerouslySetInnerHTML={{
            __html: `if ('serviceWorker' in navigator) {
              window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js'))
            }`
          }}
        />
      </body>
    </html>
  )
}
