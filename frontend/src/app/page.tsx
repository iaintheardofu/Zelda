'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Activity, Shield, Radio, Zap, Target, Eye } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-20">
        <div className="max-w-6xl mx-auto text-center space-y-8 animate-fade-in">
          {/* Logo/Title */}
          <div className="space-y-4">
            <div className="inline-block">
              <h1 className="text-8xl font-orbitron font-black text-glow-cyan tracking-wider glitch">
                ZELDA
              </h1>
              <div className="h-1 w-full bg-gradient-to-r from-transparent via-neon-cyan to-transparent glow-cyan mt-4" />
            </div>
            <p className="text-2xl font-orbitron text-neon-purple text-glow-purple uppercase tracking-widest">
              RF Signal Intelligence Platform
            </p>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto font-inter">
              Advanced TDOA Geolocation • ML Signal Detection • Defensive Electronic Warfare
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex gap-4 justify-center pt-8">
            <Link href="/dashboard">
              <Button size="xl" variant="neon" className="group">
                <Zap className="w-5 h-5 group-hover:animate-pulse" />
                Enter Dashboard
              </Button>
            </Link>
            <Link href="/login">
              <Button size="xl" variant="outline">
                <Shield className="w-5 h-5" />
                Sign In
              </Button>
            </Link>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-16 max-w-5xl mx-auto">
            <Card className="group hover:glow-cyan transition-all duration-300">
              <CardContent className="p-6 space-y-4">
                <div className="w-12 h-12 rounded-lg bg-neon-cyan/20 flex items-center justify-center group-hover:glow-cyan transition-all">
                  <Radio className="w-6 h-6 text-neon-cyan" />
                </div>
                <h3 className="text-xl font-orbitron text-neon-cyan">TDOA GEOLOCATION</h3>
                <p className="text-muted-foreground text-sm">
                  Precision emitter location tracking with &lt;10m CEP accuracy at 1km range
                </p>
                <div className="pt-2 space-y-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-cyan" />
                    Multi-receiver correlation
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-cyan" />
                    Real-time tracking
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="group hover:glow-purple transition-all duration-300">
              <CardContent className="p-6 space-y-4">
                <div className="w-12 h-12 rounded-lg bg-neon-purple/20 flex items-center justify-center group-hover:glow-purple transition-all">
                  <Activity className="w-6 h-6 text-neon-purple" />
                </div>
                <h3 className="text-xl font-orbitron text-neon-purple">ML DETECTION</h3>
                <p className="text-muted-foreground text-sm">
                  Ultra YOLO ensemble with 97%+ accuracy across 878K+ training samples
                </p>
                <div className="pt-2 space-y-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-purple" />
                    6 neural networks
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-purple" />
                    47.7M parameters
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="group hover:glow-pink transition-all duration-300">
              <CardContent className="p-6 space-y-4">
                <div className="w-12 h-12 rounded-lg bg-neon-pink/20 flex items-center justify-center group-hover:glow-pink transition-all">
                  <Shield className="w-6 h-6 text-neon-pink" />
                </div>
                <h3 className="text-xl font-orbitron text-neon-pink">DEFENSIVE EW</h3>
                <p className="text-muted-foreground text-sm">
                  Comprehensive jamming/spoofing detection with 10-30 dB SNR improvement
                </p>
                <div className="pt-2 space-y-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-pink" />
                    6 jamming types
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-1 h-1 rounded-full bg-neon-pink" />
                    Anti-jam processing
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Stats Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-16 max-w-4xl mx-auto">
            <div className="cyber-panel p-4 text-center">
              <div className="text-3xl font-orbitron font-bold text-neon-cyan text-glow-cyan">
                &lt;10m
              </div>
              <div className="text-xs text-muted-foreground uppercase tracking-wider">
                TDOA Accuracy
              </div>
            </div>
            <div className="cyber-panel p-4 text-center">
              <div className="text-3xl font-orbitron font-bold text-neon-purple text-glow-purple">
                97%+
              </div>
              <div className="text-xs text-muted-foreground uppercase tracking-wider">
                ML Detection
              </div>
            </div>
            <div className="cyber-panel p-4 text-center">
              <div className="text-3xl font-orbitron font-bold text-neon-pink text-glow-pink">
                95-99%
              </div>
              <div className="text-xs text-muted-foreground uppercase tracking-wider">
                Threat Detection
              </div>
            </div>
            <div className="cyber-panel p-4 text-center">
              <div className="text-3xl font-orbitron font-bold text-neon-green glow-green">
                &lt;500ms
              </div>
              <div className="text-xs text-muted-foreground uppercase tracking-wider">
                Latency
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="pt-16 text-center space-y-2">
            <p className="text-sm text-muted-foreground">
              Making the Invisible, Visible
            </p>
            <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
              <Eye className="w-3 h-3" />
              <span>100% Defensive • Legal & Compliant • Production Ready</span>
            </div>
          </div>
        </div>
      </main>

      {/* Animated Background Elements */}
      <div className="fixed top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-neon-cyan to-transparent animate-pulse" />
      <div className="fixed bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-neon-pink to-transparent animate-pulse" />
    </div>
  );
}
