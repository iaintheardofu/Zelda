'use client';

import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Box, RotateCcw, ZoomIn, ZoomOut, Move } from 'lucide-react';

interface Spectrogram3DProps {
  width?: number;
  height?: number;
  timeResolution?: number;
  frequencyBins?: number;
}

export function Spectrogram3D({
  width = 800,
  height = 600,
  timeResolution = 100,
  frequencyBins = 64,
}: Spectrogram3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ x: 45, y: 45 });
  const [zoom, setZoom] = useState(1.0);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [autoRotate, setAutoRotate] = useState(false);
  const [colorScale, setColorScale] = useState<'viridis' | 'plasma' | 'inferno'>('viridis');

  // Generate sample spectrogram data
  const generateSpectrogramData = (): number[][] => {
    const data: number[][] = [];
    for (let t = 0; t < timeResolution; t++) {
      const timeSlice: number[] = [];
      for (let f = 0; f < frequencyBins; f++) {
        // Simulate multiple signals with varying intensity
        let power = 0;

        // Background noise
        power += Math.random() * 10;

        // Strong carrier at f=20
        if (Math.abs(f - 20) < 3) {
          power += 70 + Math.sin(t * 0.1) * 10;
        }

        // Sweeping signal
        const sweepFreq = (t / timeResolution) * frequencyBins;
        if (Math.abs(f - sweepFreq) < 2) {
          power += 50;
        }

        // Intermittent burst
        if (t > 40 && t < 60 && f > 45 && f < 52) {
          power += 60 * Math.sin((t - 40) * 0.3);
        }

        timeSlice.push(Math.max(0, power));
      }
      data.push(timeSlice);
    }
    return data;
  };

  const [spectrogramData] = useState(generateSpectrogramData());

  // Color mapping
  const getColor = (value: number, max: number): string => {
    const normalized = Math.min(value / max, 1);

    if (colorScale === 'viridis') {
      // Viridis colormap approximation
      const r = Math.floor(68 + normalized * (253 - 68));
      const g = Math.floor(1 + normalized * (231 - 1));
      const b = Math.floor(84 + normalized * (37 - 84));
      return `rgb(${r}, ${g}, ${b})`;
    } else if (colorScale === 'plasma') {
      // Plasma colormap approximation
      const r = Math.floor(13 + normalized * (240 - 13));
      const g = Math.floor(8 + normalized * (249 - 8));
      const b = Math.floor(135 + normalized * (33 - 135));
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Inferno colormap
      const r = Math.floor(0 + normalized * 252);
      const g = Math.floor(0 + normalized * 255);
      const b = Math.floor(4 + normalized * (164 - 4));
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // 3D projection
  const project3D = (
    x: number,
    y: number,
    z: number,
    rotX: number,
    rotY: number,
    zoomFactor: number
  ): { x: number; y: number } => {
    // Convert rotation to radians
    const angleX = (rotX * Math.PI) / 180;
    const angleY = (rotY * Math.PI) / 180;

    // Rotate around X axis
    let y1 = y * Math.cos(angleX) - z * Math.sin(angleX);
    let z1 = y * Math.sin(angleX) + z * Math.cos(angleX);

    // Rotate around Y axis
    let x2 = x * Math.cos(angleY) + z1 * Math.sin(angleY);
    let z2 = -x * Math.sin(angleY) + z1 * Math.cos(angleY);

    // Apply zoom
    x2 *= zoomFactor;
    y1 *= zoomFactor;
    z2 *= zoomFactor;

    // Perspective projection
    const perspective = 500;
    const scale = perspective / (perspective + z2);

    return {
      x: x2 * scale + width / 2,
      y: y1 * scale + height / 2,
    };
  };

  const draw3DSpectrogram = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Find max value for normalization
    const maxValue = Math.max(...spectrogramData.flat());

    // Center the spectrogram
    const centerX = -timeResolution / 2;
    const centerY = -frequencyBins / 2;
    const scale = 3;

    // Draw grid base
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
    ctx.lineWidth = 1;

    // Time axis lines
    for (let t = 0; t <= timeResolution; t += 10) {
      const p1 = project3D(
        (t + centerX) * scale,
        centerY * scale,
        0,
        rotation.x,
        rotation.y,
        zoom
      );
      const p2 = project3D(
        (t + centerX) * scale,
        (centerY + frequencyBins) * scale,
        0,
        rotation.x,
        rotation.y,
        zoom
      );

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Frequency axis lines
    for (let f = 0; f <= frequencyBins; f += 8) {
      const p1 = project3D(
        centerX * scale,
        (f + centerY) * scale,
        0,
        rotation.x,
        rotation.y,
        zoom
      );
      const p2 = project3D(
        (centerX + timeResolution) * scale,
        (f + centerY) * scale,
        0,
        rotation.x,
        rotation.y,
        zoom
      );

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Draw spectrogram bars (back to front for proper occlusion)
    const bars: Array<{ t: number; f: number; power: number; z: number }> = [];

    for (let t = 0; t < timeResolution; t++) {
      for (let f = 0; f < frequencyBins; f++) {
        const power = spectrogramData[t][f];
        const x = (t + centerX) * scale;
        const y = (f + centerY) * scale;
        const z = (power / maxValue) * 100;

        bars.push({ t, f, power, z });
      }
    }

    // Sort bars by distance from camera for proper depth rendering
    bars.sort((a, b) => {
      const aProj = project3D(
        (a.t + centerX) * scale,
        (a.f + centerY) * scale,
        a.z,
        rotation.x,
        rotation.y,
        zoom
      );
      const bProj = project3D(
        (b.t + centerX) * scale,
        (b.f + centerY) * scale,
        b.z,
        rotation.x,
        rotation.y,
        zoom
      );

      // Simple depth sort (back to front)
      return (
        (b.t + centerX) * Math.sin((rotation.y * Math.PI) / 180) -
        (a.t + centerX) * Math.sin((rotation.y * Math.PI) / 180)
      );
    });

    // Draw bars
    bars.forEach(({ t, f, power, z }) => {
      const x = (t + centerX) * scale;
      const y = (f + centerY) * scale;

      // Base point
      const base = project3D(x, y, 0, rotation.x, rotation.y, zoom);
      // Top point
      const top = project3D(x, y, z, rotation.x, rotation.y, zoom);

      // Draw bar
      const color = getColor(power, maxValue);
      const alpha = 0.8;

      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(base.x, base.y);
      ctx.lineTo(top.x, top.y);
      ctx.stroke();

      // Draw top cap
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(top.x, top.y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.globalAlpha = 1.0;

    // Draw axes labels
    ctx.fillStyle = '#00ffff';
    ctx.font = '12px "Orbitron", monospace';
    ctx.textAlign = 'center';

    // Time axis label
    const timeLabel = project3D(0, (centerY - 20) * scale, 0, rotation.x, rotation.y, zoom);
    ctx.fillText('Time', timeLabel.x, timeLabel.y);

    // Frequency axis label
    const freqLabel = project3D(
      (centerX - 20) * scale,
      0,
      0,
      rotation.x,
      rotation.y,
      zoom
    );
    ctx.fillText('Frequency', freqLabel.x, freqLabel.y);

    // Power axis label
    const powerLabel = project3D(
      (centerX - 20) * scale,
      (centerY - 20) * scale,
      50,
      rotation.x,
      rotation.y,
      zoom
    );
    ctx.fillText('Power', powerLabel.x, powerLabel.y);
  };

  // Auto-rotation
  useEffect(() => {
    if (!autoRotate) return;

    const interval = setInterval(() => {
      setRotation((prev) => ({
        x: prev.x,
        y: (prev.y + 1) % 360,
      }));
    }, 50);

    return () => clearInterval(interval);
  }, [autoRotate]);

  // Redraw on state changes
  useEffect(() => {
    draw3DSpectrogram();
  }, [rotation, zoom, colorScale, spectrogramData]);

  // Mouse interaction handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
    setAutoRotate(false);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;

    setRotation((prev) => ({
      x: Math.max(-90, Math.min(90, prev.x + dy * 0.5)),
      y: (prev.y + dx * 0.5) % 360,
    }));

    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setRotation({ x: 45, y: 45 });
    setZoom(1.0);
    setAutoRotate(false);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Box className="w-5 h-5 text-neon-purple" />
            <CardTitle>3D Spectrogram</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-orbitron">
              {timeResolution}×{frequencyBins}
            </Badge>
            <Badge variant="success">Real-time</Badge>
          </div>
        </div>
        <CardDescription>
          Time-frequency-power visualization with interactive 3D controls
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Canvas */}
          <div className="flex justify-center">
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              className="rounded-lg border border-neon-purple/30 bg-black cursor-move"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div>

          {/* Controls */}
          <div className="grid grid-cols-2 gap-4">
            {/* Rotation Controls */}
            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>X Rotation</span>
                  <span>{rotation.x.toFixed(0)}°</span>
                </div>
                <Slider
                  value={[rotation.x]}
                  onValueChange={(v) => {
                    setRotation((prev) => ({ ...prev, x: v[0] }));
                    setAutoRotate(false);
                  }}
                  min={-90}
                  max={90}
                  step={1}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Y Rotation</span>
                  <span>{rotation.y.toFixed(0)}°</span>
                </div>
                <Slider
                  value={[rotation.y]}
                  onValueChange={(v) => {
                    setRotation((prev) => ({ ...prev, y: v[0] }));
                    setAutoRotate(false);
                  }}
                  min={0}
                  max={360}
                  step={1}
                  className="w-full"
                />
              </div>
            </div>

            {/* Zoom and Color */}
            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Zoom</span>
                  <span>{zoom.toFixed(2)}x</span>
                </div>
                <Slider
                  value={[zoom]}
                  onValueChange={(v) => setZoom(v[0])}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <div className="text-xs text-muted-foreground mb-1">Color Scale</div>
                <div className="flex gap-2">
                  <Button
                    variant={colorScale === 'viridis' ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => setColorScale('viridis')}
                  >
                    Viridis
                  </Button>
                  <Button
                    variant={colorScale === 'plasma' ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => setColorScale('plasma')}
                  >
                    Plasma
                  </Button>
                  <Button
                    variant={colorScale === 'inferno' ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1 text-xs"
                    onClick={() => setColorScale('inferno')}
                  >
                    Inferno
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setAutoRotate(!autoRotate)}
              className="flex-1"
            >
              <Move className="w-4 h-4 mr-1" />
              {autoRotate ? 'Stop' : 'Auto'} Rotate
            </Button>
            <Button variant="outline" size="sm" onClick={resetView} className="flex-1">
              <RotateCcw className="w-4 h-4 mr-1" />
              Reset View
            </Button>
          </div>

          {/* Legend */}
          <div className="p-3 rounded-lg bg-background/50 border border-border">
            <div className="text-xs text-muted-foreground mb-2">Interaction</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-neon-cyan" />
                <span>Drag to rotate</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-neon-purple" />
                <span>Scroll to zoom</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
