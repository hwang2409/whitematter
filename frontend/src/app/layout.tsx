import type { Metadata } from "next";
import { Providers } from "./providers";
import "@/app/globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000"),
  title: "WhiteMatter — Build ML Models Through Conversation",
  description:
    "Describe what you want to build. We design the architecture, train the model, and give you an API. No ML expertise required.",
  openGraph: {
    title: "WhiteMatter — Build ML Models Through Conversation",
    description:
      "Describe what you want to build. We design the architecture, train the model, and give you an API.",
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
