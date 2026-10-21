// oral-XAI Service Worker :: Production Offline Cache & PWA WebAPK Support
const CACHE_NAME = 'oral-xai-pwa-v2';

const PRECACHE_ASSETS = [
    './',
    './index.html',
    './style.css',
    './main.js',
    './manifest.json',
    './assets/icons/icon-192.png',
    './assets/icons/icon-512.png',
    './assets/icons/icon-maskable-192.png',
    './assets/icons/icon-maskable-512.png',
    './assets/icons/apple-touch-icon.png',
    './assets/presets/opmd.jpg',
    './assets/presets/cancer.jpg',
    './assets/presets/normal.jpg',
    './assets/presets/variation.jpg'
];

// Install: Cache critical assets individually with Promise.allSettled so no single asset causes failure
self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return Promise.allSettled(
                PRECACHE_ASSETS.map((asset) => {
                    return fetch(asset).then((res) => {
                        if (res.ok) return cache.put(asset, res);
                    }).catch((e) => {
                        console.warn('[SW] Precache skipped:', asset, e);
                    });
                })
            );
        })
    );
});

// Activate: Purge old cache stores immediately
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME) {
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch: Stale-While-Revalidate with strict non-undefined Response fallback
self.addEventListener('fetch', (event) => {
    const req = event.request;

    // Only intercept GET requests
    if (req.method !== 'GET') {
        return;
    }

    // Skip Chrome extension requests or analytics
    if (!req.url.startsWith('http')) {
        return;
    }

    event.respondWith(
        caches.match(req).then((cachedResponse) => {
            if (cachedResponse) {
                // Fetch updated version in background
                fetch(req).then((networkRes) => {
                    if (networkRes && networkRes.status === 200) {
                        const copy = networkRes.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
                    }
                }).catch(() => {});
                return cachedResponse;
            }

            return fetch(req).then((networkRes) => {
                if (networkRes && networkRes.status === 200) {
                    const copy = networkRes.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
                }
                return networkRes;
            }).catch(() => {
                // Offline fallback: for navigations, return cached index.html
                if (req.mode === 'navigate' || (req.headers.get('accept') && req.headers.get('accept').includes('text/html'))) {
                    return caches.match('./index.html').then((indexCached) => {
                        return indexCached || caches.match('./');
                    });
                }
                // Never return undefined to event.respondWith
                return new Response('Network error occurred', {
                    status: 408,
                    headers: { 'Content-Type': 'text/plain' }
                });
            });
        })
    );
});
