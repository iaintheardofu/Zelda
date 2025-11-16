'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Activity, Play, Pause, Settings, Download, Maximize2 } from 'lucide-react';

export default function SpectrumPage() {
  const [isRunning, setIsRunning] = useState(true);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-cyan uppercase tracking-wider">
            SPECTRUM ANALYZER
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time RF spectrum waterfall visualization
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant={isRunning ? 'success' : 'warning'} className="animate-pulse">
            {isRunning ? 'Live' : 'Paused'}
          </Badge>
          <Button
            variant="neon"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Resume
              </>
            )}
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4" />
            Config
          </Button>
        </div>
      </div>

      {/* Main Spectrum Display */}
      <Card className="h-[600px]">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-neon-cyan animate-pulse" />
                Waterfall Display
              </CardTitle>
              <CardDescription>915 MHz ISM Band</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon">
                <Download className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon">
                <Maximize2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative h-[480px] bg-black rounded-lg border border-neon-cyan/30 overflow-hidden">
            {/* Spectrum Waterfall Placeholder */}
            <div className="absolute inset-0">
              {/* Grid overlay */}
              <div className="absolute inset-0 grid grid-cols-10 grid-rows-10">
                {Array.from({ length: 100 }).map((_, i) => (
                  <div key={i} className="border border-neon-cyan/5" />
                ))}
              </div>

              {/* Center message */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center space-y-4">
                  <Activity className="w-16 h-16 text-neon-cyan mx-auto animate-glow-pulse" />
                  <div>
                    <p className="text-lg font-orbitron text-neon-cyan text-glow-cyan">
                      SPECTRUM ANALYZER
                    </p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Connect WebSocket to view live RF data
                    </p>
                    <p className="text-xs text-muted-foreground mt-1 font-mono">
                      ws://localhost:8000/ws
                    </p>
                  </div>
                  <Button variant="neon" size="sm">
                    <Play className="w-4 h-4" />
                    Connect to Backend
                  </Button>
                </div>
              </div>

              {/* Frequency scale */}
              <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-black/80 to-transparent flex items-end justify-between px-4 pb-2 text-xs font-mono text-neon-cyan">
                <span>900 MHz</span>
                <span>905 MHz</span>
                <span>910 MHz</span>
                <span>915 MHz</span>
                <span>920 MHz</span>
                <span>925 MHz</span>
                <span>930 MHz</span>
              </div>

              {/* Power scale */}
              <div className="absolute left-0 top-0 bottom-0 w-12 bg-gradient-to-r from-black/80 to-transparent flex flex-col justify-between py-4 pl-2 text-xs font-mono text-neon-cyan">
                <span>0</span>
                <span>-20</span>
                <span>-40</span>
                <span>-60</span>
                <span>-80</span>
                <span>-100</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detection Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Signal Detections</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
                <div>
                  <p className="text-xs text-muted-foreground">Frequency</p>
                  <p className="text-sm font-orbitron text-neon-cyan font-bold">915.25 MHz</p>
                </div>
                <Badge variant="success">Active</Badge>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-neon-purple/10 border border-neon-purple/30">
                <div>
                  <p className="text-xs text-muted-foreground">Frequency</p>
                  <p className="text-sm font-orbitron text-neon-purple font-bold">920.10 MHz</p>
                </div>
                <Badge variant="info">Detected</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Parameters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Center Freq</span>
                <span className="font-orbitron text-neon-cyan">915.0 MHz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Bandwidth</span>
                <span className="font-orbitron text-neon-cyan">40 MHz</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sample Rate</span>
                <span className="font-orbitron text-neon-cyan">40 MS/s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Resolution</span>
                <span className="font-orbitron text-neon-cyan">10 kHz</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Signals</span>
                <span className="font-orbitron text-neon-green">23</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Power</span>
                <span className="font-orbitron text-neon-cyan">-45 dBm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Peak Power</span>
                <span className="font-orbitron text-neon-purple">-12 dBm</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Occupancy</span>
                <span className="font-orbitron text-neon-pink">67%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
