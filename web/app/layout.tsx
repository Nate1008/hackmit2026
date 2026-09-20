import type { Metadata } from "next";
import "./globals.css";
import "./studio.css";

const siteUrl = new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: siteUrl,
  title: {
    default: "HonkPack — Video to editable 3D",
    template: "%s · HonkPack",
  },
  description: "Turn a quick video of your packed items into an editable 3D scene.",
  icons: {
    icon: "/brand/honkpack-mark.png",
    shortcut: "/brand/honkpack-mark.png",
    apple: "/brand/honkpack-mark.png",
  },
  openGraph: {
    title: "HonkPack — Video to editable 3D",
    description: "Turn a quick video of your packed items into an editable 3D scene.",
    images: [{ url: "/brand/honkpack-mark.png", width: 1254, height: 1254 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "HonkPack — Video to editable 3D",
    description: "Turn a quick video of your packed items into an editable 3D scene.",
    images: ["/brand/honkpack-mark.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
