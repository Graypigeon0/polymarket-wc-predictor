import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

// Overrides the root manifest for the /bets route only. Because Next resolves
// `metadata.manifest` per route segment (deepest wins), the /bets page emits a
// <link rel="manifest" href="/bets.webmanifest"> whose start_url is /bets — so
// an icon added from this page launches here instead of the edges dashboard.
export const metadata: Metadata = {
  title: "My Bets",
  manifest: "/bets.webmanifest",
  appleWebApp: {
    capable: true,
    title: "My Bets",
    statusBarStyle: "black-translucent",
  },
};

export const viewport: Viewport = {
  themeColor: "#0b0e14",
  width: "device-width",
  initialScale: 1,
};

export default function BetsLayout({ children }: { children: ReactNode }) {
  return children;
}
