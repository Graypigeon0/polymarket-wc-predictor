// Minimal service worker — caches the shell for offline opening.
// On install, no-op. On fetch, network-first with cache fallback for navigation.
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))
self.addEventListener('fetch', (event) => {
  if (event.request.mode !== 'navigate') return
  event.respondWith(
    fetch(event.request).catch(() => caches.match('/') || new Response('Offline'))
  )
})
