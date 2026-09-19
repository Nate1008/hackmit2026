import http from "node:http";
import { createRequire } from "node:module";

const require = createRequire(new URL("../web/package.json", import.meta.url));
const { WebSocket, WebSocketServer } = require("ws");

const listenHost = process.env.SHAPER_PROXY_HOST || "127.0.0.1";
const listenPort = Number(process.env.SHAPER_PROXY_PORT || 3200);
const backendPort = Number(process.env.SHAPER_BACKEND_PORT || 8000);
const frontendPort = Number(process.env.SHAPER_FRONTEND_PORT || 3100);
const apiPrefix = "/api/shaper";
const eventHistoryPattern = /^\/api\/shaper\/api\/jobs\/([a-zA-Z0-9_-]+)\/events\/history$/;
const eventSocketPattern = /^\/api\/shaper\/api\/jobs\/([a-zA-Z0-9_-]+)\/events\/ws$/;
const pairingSocketPattern = /^\/api\/shaper\/api\/pairings\/([a-zA-Z0-9_-]+)\/events\/ws$/;
const webSockets = new WebSocketServer({ noServer: true });

function targetFor(requestUrl = "/") {
  if (requestUrl === apiPrefix || requestUrl.startsWith(`${apiPrefix}/`)) {
    return {
      hostname: "127.0.0.1",
      port: backendPort,
      path: requestUrl.slice(apiPrefix.length) || "/",
      service: "backend",
    };
  }
  return {
    hostname: "127.0.0.1",
    port: frontendPort,
    path: requestUrl,
    service: "frontend",
  };
}

function sendEventHistory(request, response, jobId, after) {
  const upstream = http.get(
    {
      hostname: "127.0.0.1",
      port: backendPort,
      path: "/api/jobs/" + encodeURIComponent(jobId) + "/events",
      headers: { accept: "text/event-stream", "last-event-id": after },
    },
    (upstreamResponse) => {
      if (upstreamResponse.statusCode !== 200) {
        response.writeHead(upstreamResponse.statusCode || 502, {
          "content-type": "application/json",
          "cache-control": "no-store",
        });
        response.end(JSON.stringify({ events: [] }));
        upstreamResponse.resume();
        return;
      }

      upstreamResponse.setEncoding("utf8");
      const events = [];
      let buffer = "";
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        upstream.destroy();
        response.writeHead(200, {
          "content-type": "application/json; charset=utf-8",
          "cache-control": "no-store",
        });
        response.end(JSON.stringify({ events }));
      };
      const timeout = setTimeout(finish, 3000);

      upstreamResponse.on("data", (chunk) => {
        buffer += chunk.replaceAll("\r\n", "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          if (block.startsWith(": keepalive")) {
            finish();
            return;
          }
          const payload = block
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (payload) {
            try {
              events.push(JSON.parse(payload));
            } catch {
              // Ignore a malformed event and keep the remaining history usable.
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      });
      upstreamResponse.on("end", finish);
    },
  );

  upstream.on("error", (error) => {
    console.error("[public-proxy] event history failed", {
      jobId,
      message: error.message,
    });
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "application/json" });
    }
    response.end(JSON.stringify({ events: [] }));
  });
  request.on("aborted", () => upstream.destroy());
}

const server = http.createServer((request, response) => {
  const requestUrl = new URL(request.url || "/", "http://localhost");
  const historyMatch = requestUrl.pathname.match(eventHistoryPattern);
  if (request.method === "GET" && historyMatch) {
    sendEventHistory(
      request,
      response,
      historyMatch[1],
      requestUrl.searchParams.get("after") || "0",
    );
    return;
  }

  const target = targetFor(request.url);
  const headers = { ...request.headers };
  headers.host = `${target.hostname}:${target.port}`;
  headers["x-forwarded-host"] = request.headers.host || "";
  headers["x-forwarded-proto"] = request.headers["x-forwarded-proto"] || "https";

  const upstream = http.request(
    {
      hostname: target.hostname,
      port: target.port,
      path: target.path,
      method: request.method,
      headers,
    },
    (upstreamResponse) => {
      response.writeHead(
        upstreamResponse.statusCode || 502,
        upstreamResponse.statusMessage,
        upstreamResponse.headers,
      );
      response.flushHeaders();
      upstreamResponse.pipe(response);
    },
  );

  upstream.on("error", (error) => {
    console.error("[public-proxy] upstream failed", {
      service: target.service,
      message: error.message,
    });
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "application/json" });
    }
    response.end(JSON.stringify({ detail: `${target.service} unavailable` }));
  });

  request.on("aborted", () => upstream.destroy());
  request.pipe(upstream);
});

function bridgeEvents(client, streamPath, streamLabel) {
  const upstream = http.get(
    {
      hostname: "127.0.0.1",
      port: backendPort,
      path: streamPath,
      headers: { accept: "text/event-stream" },
    },
    (upstreamResponse) => {
      if (upstreamResponse.statusCode !== 200) {
        client.send(JSON.stringify({
          id: -1,
          type: "error",
          stage: "error",
          progress: 0,
          title: "Reconstruction unavailable",
          detail: "Event stream returned " + (upstreamResponse.statusCode || 502) + ".",
        }));
        client.close(1011);
        upstreamResponse.resume();
        return;
      }

      upstreamResponse.setEncoding("utf8");
      let buffer = "";
      upstreamResponse.on("data", (chunk) => {
        buffer += chunk.replaceAll("\r\n", "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const payload = block
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (payload && client.readyState === WebSocket.OPEN) client.send(payload);
          boundary = buffer.indexOf("\n\n");
        }
      });
      upstreamResponse.on("end", () => {
        if (client.readyState === WebSocket.OPEN) client.close(1000);
      });
    },
  );

  upstream.on("error", (error) => {
    console.error("[public-proxy] event bridge failed", {
      streamLabel,
      message: error.message,
    });
    if (client.readyState === WebSocket.OPEN) client.close(1011);
  });
  client.on("close", () => upstream.destroy());
  client.on("error", () => upstream.destroy());
}

server.on("upgrade", (request, socket, head) => {
  const pathname = new URL(request.url || "/", "http://localhost").pathname;
  const jobMatch = pathname.match(eventSocketPattern);
  const pairingMatch = pathname.match(pairingSocketPattern);
  if (!jobMatch && !pairingMatch) {
    socket.destroy();
    return;
  }
  const streamPath = jobMatch
    ? `/api/jobs/${encodeURIComponent(jobMatch[1])}/events`
    : `/api/pairings/${encodeURIComponent(pairingMatch[1])}/events`;
  const streamLabel = jobMatch ? `job:${jobMatch[1]}` : `pairing:${pairingMatch[1]}`;
  webSockets.handleUpgrade(request, socket, head, (client) => {
    bridgeEvents(client, streamPath, streamLabel);
  });
});

const heartbeat = setInterval(() => {
  for (const client of webSockets.clients) {
    if (client.readyState === WebSocket.OPEN) client.ping();
  }
}, 20_000);
heartbeat.unref();

server.requestTimeout = 0;
server.timeout = 0;
server.keepAliveTimeout = 65_000;
server.headersTimeout = 70_000;

server.listen(listenPort, listenHost, () => {
  console.log(`[public-proxy] listening on http://${listenHost}:${listenPort}`);
});
