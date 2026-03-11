import type { Metadata } from "next";
import { Providers } from "./providers";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "whitematter",
  description: "Whitematter ML platform",
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
