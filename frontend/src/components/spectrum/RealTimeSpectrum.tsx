'use client';

import { useEffect, useRef } from 'react';
import { useSpectrumData } from '@/hooks/useRealTimeData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity } from 'lucide-react';
import { formatFrequency, formatPower } from '@/lib/utils';

export function RealTimeSpectrum() {
  const { spectrumData, history, isConnected } = useSpectrumData();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !history.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Draw waterfall
    const numBins = history[0]?.powers?.length || 512;
    const binWidth = width / numBins;
    const lineHeight = height / history.length;

    history.forEach((spectrum, timeIndex) => {
      if (!spectrum.powers) return;

      spectrum.powers.forEach((power, freqIndex) => {
        // Convert power to color (dBm to heatmap)
        const normalizedPower = (power + 100) / 50; // -100 to -50 dBm range
        const intensity = Math.max(0, Math.min(1, normalizedPower));

        // Cyberpunk color gradient
        let r, g, b;
        if (intensity < 0.5) {
          // Dark blue to cyan
          r = 0;
          g = Math.floor(intensity * 2 * 255);
          b = 255;
        } else {
          // Cyan to pink
          const t = (intensity - 0.5) * 2;
          r = Math.floor(t * 255);
          g = Math.floor((1 - t) * 255);
          b = Math.floor((1 - t) * 255);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(
          freqIndex * binWidth,
          height - (timeIndex + 1) * lineHeight,
          binWidth,
          lineHeight
        );
      });
    });

    // Draw grid overlay
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Vertical lines
    for (let i = 0; i < 10; i++) {
      const x = (i / 10) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let i = 0; i < 10; i++) {
      const y = (i / 10) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

  }, [history]);

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-neon-cyan animate-pulse" />
              Real-Time Spectrum Waterfall
            </CardTitle>
            <CardDescription>
              {spectrumData
                ? `${formatFrequency(spectrumData.frequencies[0])} - ${formatFrequency(spectrumData.frequencies[spectrumData.frequencies.length - 1])}`
                : 'Waiting for data...'}
            </CardDescription>
          </div>
          <Badge variant={isConnected ? 'success' : 'danger'} className="animate-pulse">
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={1024}
            height={512}
            className="w-full h-auto rounded-lg border border-neon-cyan/30 bg-black"
          />

          {/* Frequency labels */}
          {spectrumData && (
            <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 text-xs font-mono text-neon-cyan">
              <span>{formatFrequency(spectrumData.frequencies[0])}</span>
              <span>{formatFrequency(spectrumData.frequencies[Math.floor(spectrumData.frequencies.length / 2)])}</span>
              <span>{formatFrequency(spectrumData.frequencies[spectrumData.frequencies.length - 1])}</span>
            </div>
          )}

          {/* Power scale */}
          <div className="absolute left-2 top-0 bottom-0 flex flex-col justify-between py-4 text-xs font-mono text-neon-cyan">
            <span>0 dBm</span>
            <span>-50 dBm</span>
            <span>-100 dBm</span>
          </div>

          {/* Current power indicator */}
          {spectrumData && (
            <div className="mt-4 grid grid-cols-3 gap-4 text-center text-sm">
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
            </div>
          )}

          {!isConnected && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80 rounded-lg">
              <div className="text-center space-y-2">
                <Activity className="w-12 h-12 text-neon-red mx-auto animate-pulse" />
                <p className="text-sm text-neon-red font-orbitron">WebSocket Disconnected</p>
                <p className="text-xs text-muted-foreground">Attempting to reconnect...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
