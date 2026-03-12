/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
  async headers() {
    const isDev = process.env.NODE_ENV === "development";
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              isDev
                ? "script-src 'self' 'unsafe-eval' 'unsafe-inline'"
                : "script-src 'self' 'unsafe-inline'",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "connect-src 'self' ws: wss:",
              "img-src 'self' data: blob: https://img.shields.io",
            ].join("; "),
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
        ],
      },
    ];
  },
  async rewrites() {
    return {
      // "beforeFiles" rewrites run before pages, so API calls to /chat/conversations
      // are proxied to the backend even though /chat/[id] is a page route.
      beforeFiles: [
        { source: '/chat/conversations', destination: 'http://localhost:8080/chat/conversations' },
        { source: '/chat/conversations/:path*', destination: 'http://localhost:8080/chat/conversations/:path*' },
      ],
      afterFiles: [
        { source: '/auth/:path*', destination: 'http://localhost:8080/auth/:path*' },
        { source: '/credentials/:path*', destination: 'http://localhost:8080/credentials/:path*' },
        { source: '/storage/:path*', destination: 'http://localhost:8080/storage/:path*' },
        { source: '/training/:path*', destination: 'http://localhost:8080/training/:path*' },
        { source: '/datasets/:path*', destination: 'http://localhost:8080/datasets/:path*' },
        { source: '/models/:path*', destination: 'http://localhost:8080/models/:path*' },
        { source: '/train/:path*', destination: 'http://localhost:8080/train/:path*' },
        { source: '/predict/:path*', destination: 'http://localhost:8080/predict/:path*' },
        { source: '/design/:path*', destination: 'http://localhost:8080/design/:path*' },
        { source: '/config/:path*', destination: 'http://localhost:8080/config/:path*' },
        { source: '/health', destination: 'http://localhost:8080/health' },
        { source: '/workers/:path*', destination: 'http://localhost:8080/workers/:path*' },
        { source: '/api/:path*', destination: 'http://localhost:8080/api/:path*' },
        { source: '/ws/:path*', destination: 'http://localhost:8080/ws/:path*' },
        { source: '/billing/:path*', destination: 'http://localhost:8080/billing/:path*' },
      ],
      fallback: [],
    };
  },
};

export default nextConfig;
