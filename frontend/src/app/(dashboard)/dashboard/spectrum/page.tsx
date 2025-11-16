'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Activity, Play, Pause, Settings, Download, Maximize2, Wifi, WifiOff } from 'lucide-react';
import { WebGLWaterfall, WebGLWaterfallRef } from '@/components/spectrum/WebGLWaterfall';
import { ThreatClassifier } from '@/components/threats/ThreatClassifier';
import { useDetections, useSpectrumData } from '@/hooks/useRealTimeData';
import { formatFrequency, formatPower, formatRelativeTime } from '@/lib/utils';

export default function SpectrumPage() {
  const [isRunning, setIsRunning] = useState(true);
  const [centerFreq, setCenterFreq] = useState(915e6); // 915 MHz
  const [span, setSpan] = useState(40e6); // 40 MHz
  const waterfallRef = useRef<WebGLWaterfallRef>(null);

  const { detections, latestDetection, isConnected: detectionsConnected } = useDetections();
  const { spectrumData, history, isConnected: spectrumConnected } = useSpectrumData();

  const isConnected = detectionsConnected || spectrumConnected;

  // Feed spectrum data to WebGL waterfall
  useEffect(() => {
    if (spectrumData && spectrumData.powers && waterfallRef.current) {
      waterfallRef.current.addFFTData(spectrumData.powers);
    }
  }, [spectrumData]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-cyan uppercase tracking-wider">
            SPECTRUM ANALYZER
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            WebGL-accelerated RF spectrum waterfall - Real-time FFT visualization
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Wifi className="w-4 h-4 text-neon-green animate-pulse" />
            ) : (
              <WifiOff className="w-4 h-4 text-neon-red animate-pulse" />
            )}
            <Badge variant={isConnected ? 'success' : 'danger'} className="animate-pulse">
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </Badge>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? (
              <>
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-1" />
                Resume
              </>
            )}
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4 mr-1" />
            Config
          </Button>
        </div>
      </div>

      {/* WebGL Waterfall Display */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-neon-cyan animate-pulse" />
                WATERFALL DISPLAY - WEBSOCKET READY
              </CardTitle>
              <CardDescription>
                {spectrumData
                  ? `${formatFrequency(centerFreq - span / 2)} - ${formatFrequency(centerFreq + span / 2)}`
                  : 'WebGL waterfall ready for spectrum data'}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-xs text-muted-foreground">
                <span className="font-orbitron">CENTER:</span>{' '}
                <span className="text-neon-cyan">{formatFrequency(centerFreq)}</span>
              </div>
              <div className="text-xs text-muted-foreground">
                <span className="font-orbitron">SPAN:</span>{' '}
                <span className="text-neon-purple">{formatFrequency(span)}</span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => waterfallRef.current?.clear()}
              >
                Clear
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <WebGLWaterfall
              ref={waterfallRef}
              width={1024}
              height={512}
              fftSize={1024}
              historySize={200}
              className="w-full h-auto rounded-lg border border-neon-cyan/30 bg-black"
            />

            {/* Frequency labels */}
            <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 text-xs font-mono text-neon-cyan">
              <span>{formatFrequency(centerFreq - span / 2)}</span>
              <span>{formatFrequency(centerFreq)}</span>
              <span>{formatFrequency(centerFreq + span / 2)}</span>
            </div>

            {/* Power scale */}
            <div className="absolute left-2 top-0 bottom-0 flex flex-col justify-between py-4 text-xs font-mono text-neon-cyan">
              <span>-20 dBm</span>
              <span>-60 dBm</span>
              <span>-100 dBm</span>
            </div>

            {/* Connection status overlay */}
            {!isConnected && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/80 rounded-lg">
                <div className="text-center space-y-2">
                  <WifiOff className="w-12 h-12 text-neon-red mx-auto animate-pulse" />
                  <p className="text-sm text-neon-red font-orbitron">WebSocket Disconnected</p>
                  <p className="text-xs text-muted-foreground">Waiting for spectrum data stream...</p>
                  <p className="text-xs text-neon-cyan mt-2">
                    Connect Python backend or use mock data generator
                  </p>
                </div>
              </div>
            )}

            {/* Stats */}
            {spectrumData && (
              <div className="mt-4 grid grid-cols-4 gap-4 text-center text-sm">
                <div>
                  <div className="text-xs text-muted-foreground">Min Power</div>
                  <div className="text-lg font-orbitron text-neon-cyan">
                    {formatPower(Math.min(...spectrumData.powers))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Avg Power</div>
                  <div className="text-lg font-orbitron text-neon-purple">
                    {formatPower(spectrumData.powers.reduce((a, b) => a + b, 0) / spectrumData.powers.length)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Peak Power</div>
                  <div className="text-lg font-orbitron text-neon-pink">
                    {formatPower(Math.max(...spectrumData.powers))}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Update Rate</div>
                  <div className="text-lg font-orbitron text-neon-green">
                    {isConnected ? '60 FPS' : '--'}
                  </div>
                </div>
              </div>
            )}

            {/* Presets */}
            <div className="mt-4 flex gap-2">
              <span className="text-xs text-muted-foreground self-center font-orbitron">PRESETS:</span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => { setCenterFreq(915e6); setSpan(50e6); }}
              >
                915 MHz
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => { setCenterFreq(2.45e9); setSpan(100e6); }}
              >
                WiFi
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => { setCenterFreq(1.5755e9); setSpan(20e6); }}
              >
                GPS
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

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
