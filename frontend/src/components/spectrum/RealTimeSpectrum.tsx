'use client';

import { useEffect, useRef } from 'react';
import { useSpectrumData, useDetections } from '@/hooks/useRealTimeData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, Radio } from 'lucide-react';
import { formatFrequency, formatPower } from '@/lib/utils';
import { useGlobalThreats } from '@/contexts/ThreatContext';

export function RealTimeSpectrum() {
  const { spectrumData, history, isConnected } = useSpectrumData();
  const { detections } = useDetections();
  const { classifySignal, createThreat } = useGlobalThreats();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

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

  // Draw signal detection overlays
  useEffect(() => {
    if (!overlayCanvasRef.current || !spectrumData || detections.length === 0) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear overlay
    ctx.clearRect(0, 0, width, height);

    const minFreq = spectrumData.frequencies[0];
    const maxFreq = spectrumData.frequencies[spectrumData.frequencies.length - 1];
    const freqRange = maxFreq - minFreq;

    // Draw each detection
    detections.forEach((detection) => {
      const freq = detection.frequency;
      if (freq < minFreq || freq > maxFreq) return;

      // Calculate position
      const x = ((freq - minFreq) / freqRange) * width;

      // Classify threat level
      const classification = classifySignal(
        detection.frequency,
        detection.power,
        detection.bandwidth || 0,
        detection.modulation
      );

      // Choose color based on severity
      let color;
      switch (classification.severity) {
        case 'critical':
          color = '#ff1166'; // neon-red
          break;
        case 'high':
          color = '#ff6b35'; // neon-orange
          break;
        case 'medium':
          color = '#ffd700'; // neon-yellow
          break;
        default:
          color = '#00ffff'; // neon-cyan
      }

      // Draw vertical line at signal frequency
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw signal marker at top
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, 10, 6, 0, Math.PI * 2);
      ctx.fill();

      // Add glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 15;
      ctx.beginPath();
      ctx.arc(x, 10, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Draw label
      ctx.fillStyle = color;
      ctx.font = '10px "Orbitron", monospace';
      ctx.textAlign = 'center';
      ctx.fillText(
        detection.signal_type,
        x,
        30
      );

      // Draw confidence
      ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.font = '8px monospace';
      ctx.fillText(
        `${(detection.confidence * 100).toFixed(0)}%`,
        x,
        42
      );
    });
  }, [spectrumData, detections, classifySignal]);

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
          {/* Overlay canvas for signal markers */}
          <canvas
            ref={overlayCanvasRef}
            width={1024}
            height={512}
            className="absolute top-0 left-0 w-full h-auto rounded-lg pointer-events-none"
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

          {/* Signal classification legend */}
          {detections.length > 0 && (
            <div className="mt-4 p-3 bg-background/50 rounded-lg border border-border">
              <p className="text-xs font-orbitron text-muted-foreground mb-2">Signal Classification:</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-neon-red" />
                  <span>Critical</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-neon-orange" />
                  <span>High</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-neon-yellow" />
                  <span>Medium</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-neon-cyan" />
                  <span>Low/Normal</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
