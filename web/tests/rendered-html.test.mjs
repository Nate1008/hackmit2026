import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the HonkPack application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>HonkPack — Video to editable 3D<\/title>/i);
  assert.match(html, /Turn a quick video of your packed items into an editable 3D scene/);
  assert.match(html, /honkpack-mark\.png/);
  assert.match(html, /mobile-capture-shell/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("keeps Vercel, backend, and Google Maps configuration explicit", async () => {
  const [packageText, vercelText, envExample, nextConfig, truckSource] = await Promise.all([
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../vercel.json", import.meta.url), "utf8"),
    readFile(new URL("../.env.example", import.meta.url), "utf8"),
    readFile(new URL("../next.config.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/truck-fitting.tsx", import.meta.url), "utf8"),
  ]);
  const packageJson = JSON.parse(packageText);
  const vercel = JSON.parse(vercelText);

  assert.equal(packageJson.scripts["build:vercel"], "next build");
  assert.equal(vercel.framework, "nextjs");
  assert.equal(vercel.buildCommand, "npm run build:vercel");
  assert.match(envExample, /NEXT_PUBLIC_SHAPER_API_URL=https:\/\/api\.example\.com/);
  assert.match(envExample, /NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=replace_with_browser_key/);
  assert.match(nextConfig, /Missing required Vercel environment variables/);
  assert.match(truckSource, /fetch\(`\$\{API_BASE\}\/api\/maps\/autocomplete/);
  assert.match(truckSource, /fetch\(`\$\{API_BASE\}\/api\/maps\/route/);
  assert.doesNotMatch(envExample + nextConfig + truckSource, /AIza[0-9A-Za-z_-]{30,}/);
});
