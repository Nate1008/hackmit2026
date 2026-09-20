import type { NextConfig } from "next";

const requiredVercelEnvironment = [
  "NEXT_PUBLIC_SHAPER_API_URL",
  "NEXT_PUBLIC_GOOGLE_MAPS_API_KEY",
] as const;

if (process.env.VERCEL === "1") {
  const missing = requiredVercelEnvironment.filter((name) => !process.env[name]?.trim());
  if (missing.length > 0) {
    throw new Error(`Missing required Vercel environment variables: ${missing.join(", ")}`);
  }
}

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: "1gb",
    },
  },
};

export default nextConfig;
