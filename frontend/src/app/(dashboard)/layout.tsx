'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  Radio,
  Target,
  Shield,
  BarChart3,
  Settings,
  LogOut,
  Menu,
  X,
  Zap,
  Activity,
  Bell,
  User,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ZeldaLogoAnimated } from '@/components/ZeldaLogo';
import { ThreatProvider } from '@/contexts/ThreatContext';
import { CountermeasureProvider } from '@/contexts/CountermeasureContext';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Spectrum', href: '/dashboard/spectrum', icon: Activity },
  { name: 'Missions', href: '/dashboard/missions', icon: Target },
  { name: 'Receivers', href: '/dashboard/receivers', icon: Radio },
  { name: 'Threats', href: '/dashboard/threats', icon: Shield },
  { name: 'Analytics', href: '/dashboard/analytics', icon: BarChart3 },
  { name: 'Settings', href: '/dashboard/settings', icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();

  return (
    <ThreatProvider>
      <CountermeasureProvider>
        <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 ease-in-out',
          'bg-card/95 backdrop-blur-xl border-r border-neon-cyan/30',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between p-6 border-b border-neon-cyan/30">
          <Link href="/dashboard" className="flex items-center gap-3 group">
            <div className="relative">
              <ZeldaLogoAnimated className="w-10 h-10 drop-shadow-[0_0_10px_rgba(255,17,102,0.8)]" />
            </div>
            <div>
              <h1 className="font-orbitron font-black text-xl text-glow-pink tracking-wider">
                ZELDA
              </h1>
              <p className="text-[10px] text-neon-purple uppercase tracking-widest">
                EW SYSTEM v2.0
              </p>
            </div>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigation.map((item) => {
            const isActive = pathname === item.href || pathname?.startsWith(item.href + '/');
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 px-4 py-3 rounded-lg transition-all font-orbitron text-sm uppercase tracking-wider',
                  isActive
                    ? 'bg-neon-cyan/20 text-neon-cyan border border-neon-cyan/50 glow-cyan'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                )}
              >
                <item.icon className="w-5 h-5" />
                <span>{item.name}</span>
                {isActive && (
                  <div className="ml-auto w-2 h-2 rounded-full bg-neon-cyan animate-pulse" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* User Section */}
        <div className="p-4 border-t border-neon-cyan/30">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 mb-2">
            <div className="w-10 h-10 rounded-lg bg-neon-purple/20 flex items-center justify-center">
              <User className="w-5 h-5 text-neon-purple" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">Operator</p>
              <p className="text-xs text-muted-foreground truncate">admin@zelda.rf</p>
            </div>
          </div>
          <Button variant="outline" size="sm" className="w-full justify-start gap-2">
            <LogOut className="w-4 h-4" />
            Sign Out
          </Button>
        </div>

        {/* Status Bar */}
        <div className="p-4 border-t border-neon-cyan/30 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">System Status</span>
            <Badge variant="success" className="text-[10px]">
              Online
            </Badge>
          </div>
          <div className="text-center text-xs py-2">
            <div className="text-muted-foreground">Version 2.0.0</div>
            <div className="text-neon-cyan font-orbitron text-sm mt-1">Ready</div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 lg:ml-64">
        {/* Top Bar */}
        <header className="sticky top-0 z-40 bg-card/95 backdrop-blur-xl border-b border-neon-cyan/30">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                className="lg:hidden"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu className="w-5 h-5" />
              </Button>
              <div>
                <h2 className="font-orbitron font-bold text-lg text-glow-cyan uppercase tracking-wider">
                  {navigation.find((item) => pathname === item.href)?.name || 'Dashboard'}
                </h2>
                <p className="text-xs text-muted-foreground">
                  Real-time RF Signal Intelligence
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Notifications */}
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="w-5 h-5" />
                <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-neon-red animate-pulse" />
              </Button>

              {/* Status Indicator */}
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neon-green/20 border border-neon-green/50">
                <div className="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
                <span className="text-xs font-orbitron text-neon-green uppercase tracking-wider">
                  Live
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-6">
          {children}
        </main>
      </div>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
        </div>
      </CountermeasureProvider>
    </ThreatProvider>
  );
}
