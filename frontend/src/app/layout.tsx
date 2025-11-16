import type { Metadata } from 'next';
import { Inter, Orbitron } from 'next/font/google';
import '../styles/globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const orbitron = Orbitron({
  subsets: ['latin'],
  variable: '--font-orbitron',
  weight: ['400', '500', '700', '900'],
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'ZELDA | RF Signal Intelligence Platform',
  description: 'Advanced TDOA Geolocation, ML Signal Detection, and Defensive Electronic Warfare',
  keywords: ['RF', 'Signal Intelligence', 'TDOA', 'Electronic Warfare', 'SIGINT'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${orbitron.variable} font-inter antialiased`}>
        <div className="relative min-h-screen bg-background grid-bg">
          {/* Background Effects */}
          <div className="fixed inset-0 bg-gradient-to-br from-neon-cyan/5 via-transparent to-neon-purple/5 pointer-events-none" />
          <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-neon-cyan/10 via-transparent to-transparent pointer-events-none" />

          {/* Content */}
          <div className="relative z-10">
            {children}
          </div>

          {/* Scan Line Effect */}
          <div className="fixed inset-0 scanlines pointer-events-none opacity-20" />
        </div>
      </body>
    </html>
  );
}
