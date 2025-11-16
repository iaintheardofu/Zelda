'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Activity, Play, Pause, Settings, Download, Maximize2 } from 'lucide-react';
import { RealTimeSpectrum } from '@/components/spectrum/RealTimeSpectrum';
import { ThreatClassifier } from '@/components/threats/ThreatClassifier';
import { useDetections } from '@/hooks/useRealTimeData';
import { formatFrequency, formatPower, formatRelativeTime } from '@/lib/utils';

export default function SpectrumPage() {
  const [isRunning, setIsRunning] = useState(true);
  const { detections, latestDetection, isConnected } = useDetections();

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
          <Badge variant={isConnected ? 'success' : 'danger'} className="animate-pulse">
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4" />
            Config
          </Button>
        </div>
      </div>

      {/* Real-Time Spectrum Display */}
      <RealTimeSpectrum />

      {/* Threat Classifier */}
      <ThreatClassifier maxItems={5} compact={false} />

      {/* Detection Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Live Detections ({detections.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-[400px] overflow-y-auto">
              {detections.slice(0, 10).map((detection) => (
                <div
                  key={detection.id}
                  className="flex items-center justify-between p-3 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30 hover:glow-cyan transition-all"
                >
                  <div>
                    <p className="text-xs text-muted-foreground">{detection.signal_type}</p>
                    <p className="text-sm font-orbitron text-neon-cyan font-bold">
                      {formatFrequency(detection.frequency)}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatPower(detection.power)} • {(detection.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <Badge variant="success" className="text-[10px] mb-1">Active</Badge>
                    <p className="text-[10px] text-muted-foreground">
                      {formatRelativeTime(detection.timestamp)}
                    </p>
                  </div>
                </div>
              ))}

              {detections.length === 0 && (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No signals detected</p>
                </div>
              )}
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
