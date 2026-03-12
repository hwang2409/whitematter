import type { Metadata } from "next";
import { Providers } from "./providers";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "whitematter — Train neural networks from your browser",
  description:
    "Design architectures with AI, train models in real-time, and deploy with one click.",
  openGraph: {
    title: "whitematter",
    description:
      "Train neural networks from your browser. Design architectures with AI. Deploy with one click.",
    images: [{ url: "/og-image.svg", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "whitematter",
    description: "Train neural networks from your browser.",
    images: ["/og-image.svg"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
