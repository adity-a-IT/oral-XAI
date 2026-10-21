// oral-XAI Service Worker :: Offline Cache & PWA Support
const CACHE_NAME = 'oral-xai-pwa-v1';

const PRECACHE_ASSETS = [
    './',
    './index.html',
    './style.css',
    './main.js',
    './manifest.json',
    './assets/icons/icon-192.png',
    './assets/icons/icon-512.png',
    './assets/icons/apple-touch-icon.png',
    './assets/presets/opmd.jpg',
    './assets/presets/cancer.jpg',
    './assets/presets/normal.jpg',
    './assets/presets/variation.jpg'
];

// Install: Cache initial assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(PRECACHE_ASSETS).catch((err) => {
                console.warn('[SW] Precache partial fallback:', err);
            });
        }).then(() => self.skipWaiting())
    );
});

// Activate: Clean up old caches
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

// Fetch: Network-first for dynamic content, Cache-first for shell assets
self.addEventListener('fetch', (event) => {
    const req = event.request;

    // Never cache non-GET requests (e.g. POST /api/analyze)
    if (req.method !== 'GET') {
        return;
    }

    // Ignore cross-origin non-essential requests
    const url = new URL(req.url);
    if (url.origin !== self.location.origin && !url.hostname.includes('fonts.googleapis.com') && !url.hostname.includes('fonts.gstatic.com')) {
        return;
    }

    event.respondWith(
        fetch(req).then((networkRes) => {
            // Cache valid responses
            if (networkRes && networkRes.status === 200) {
                const resClone = networkRes.clone();
                caches.open(CACHE_NAME).then((cache) => {
                    cache.put(req, resClone);
                });
            }
            return networkRes;
        }).catch(() => {
            // Network failed: Serve from cache
            return caches.match(req).then((cachedRes) => {
                if (cachedRes) return cachedRes;
                if (req.headers.get('accept') && req.headers.get('accept').includes('text/html')) {
                    return caches.match('./index.html');
                }
            });
        })
    );
});
