// Offline support. App shell: cache first, refreshed in the background. Data: network first, so a new
// export shows up on the next load, with the cached copy as the offline fallback.
const CACHE = "tendency-board-v5";
const SHELL = ["./", "index.html", "styles.css", "app.js", "validate.js", "manifest.webmanifest", "icons/icon.svg", "icons/icon-192.png"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim())));

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;
  const put = (res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); } return res; };
  if (url.pathname.includes("/data/")) {
    e.respondWith(fetch(e.request).then(put).catch(() => caches.match(e.request).then((r) => r || Response.error())));
  } else {
    e.respondWith(caches.match(e.request).then((hit) => { const net = fetch(e.request).then(put).catch(() => hit || Response.error()); return hit || net; }));
  }
});
