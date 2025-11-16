'use client';

import { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Radio, RotateCcw, Grid3x3 } from 'lucide-react';

interface ConstellationPoint {
  i: number;
  q: number;
}

interface ConstellationDiagramProps {
  modulation?: string;
  points?: ConstellationPoint[];
  width?: number;
  height?: number;
}

export function ConstellationDiagram({
  modulation = 'QPSK',
  points = [],
  width = 400,
  height = 400,
}: ConstellationDiagramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [showIdeal, setShowIdeal] = useState(true);
  const [evm, setEvm] = useState(2.3); // Error Vector Magnitude

  // Generate sample constellation points if none provided
  useEffect(() => {
    if (points.length === 0) {
      // Generate sample QPSK constellation with noise
      const samplePoints: ConstellationPoint[] = [];
      const noise = 0.1;

      for (let i = 0; i < 1000; i++) {
        const symbol = Math.floor(Math.random() * 4);
        let idealI, idealQ;

        switch (symbol) {
          case 0: idealI = 1; idealQ = 1; break;   // Q1
          case 1: idealI = -1; idealQ = 1; break;  // Q2
          case 2: idealI = -1; idealQ = -1; break; // Q3
          case 3: idealI = 1; idealQ = -1; break;  // Q4
          default: idealI = 0; idealQ = 0;
        }

        samplePoints.push({
          i: idealI + (Math.random() - 0.5) * noise,
          q: idealQ + (Math.random() - 0.5) * noise,
        });
      }

      drawConstellation(samplePoints);
    } else {
      drawConstellation(points);
    }
  }, [points, showGrid, showIdeal, modulation]);

  const drawConstellation = (dataPoints: ConstellationPoint[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 3;

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
      ctx.lineWidth = 1;

      // Vertical and horizontal axes
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Grid lines
      const gridSpacing = scale / 2;
      for (let i = 1; i <= 4; i++) {
        // Vertical
        ctx.beginPath();
        ctx.moveTo(centerX + i * gridSpacing, 0);
        ctx.lineTo(centerX + i * gridSpacing, height);
        ctx.moveTo(centerX - i * gridSpacing, 0);
        ctx.lineTo(centerX - i * gridSpacing, height);
        ctx.stroke();

        // Horizontal
        ctx.beginPath();
        ctx.moveTo(0, centerY + i * gridSpacing);
        ctx.lineTo(width, centerY + i * gridSpacing);
        ctx.moveTo(0, centerY - i * gridSpacing);
        ctx.lineTo(width, centerY - i * gridSpacing);
        ctx.stroke();
      }

      // Circle markers
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.15)';
      [0.5, 1.0, 1.5, 2.0].forEach(radius => {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * scale, 0, 2 * Math.PI);
        ctx.stroke();
      });
    }

    // Draw ideal constellation points
    if (showIdeal) {
      const idealPoints = getIdealConstellation(modulation);
      ctx.fillStyle = '#ff1166';
      ctx.strokeStyle = '#ff1166';
      ctx.lineWidth = 2;

      idealPoints.forEach(point => {
        const x = centerX + point.i * scale;
        const y = centerY - point.q * scale;

        // Cross marker for ideal points
        ctx.beginPath();
        ctx.moveTo(x - 8, y);
        ctx.lineTo(x + 8, y);
        ctx.moveTo(x, y - 8);
        ctx.lineTo(x, y + 8);
        ctx.stroke();

        // Circle around ideal point
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, 2 * Math.PI);
        ctx.stroke();
      });
    }

    // Draw received constellation points
    ctx.fillStyle = 'rgba(0, 255, 255, 0.3)';

    dataPoints.forEach(point => {
      const x = centerX + point.i * scale;
      const y = centerY - point.q * scale;

      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw axes labels
    ctx.fillStyle = '#00ffff';
    ctx.font = '12px "Orbitron", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('I (In-Phase)', centerX, height - 10);

    ctx.save();
    ctx.translate(15, centerY);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Q (Quadrature)', 0, 0);
    ctx.restore();

    // Draw scale markers
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(0, 255, 255, 0.5)';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', centerX - 5, centerY - scale + 5);
    ctx.fillText('-1.0', centerX - 5, centerY + scale + 5);
    ctx.textAlign = 'left';
    ctx.fillText('1.0', centerX + scale + 5, centerY + 5);
    ctx.fillText('-1.0', centerX - scale - 5, centerY + 5);
  };

  const getIdealConstellation = (mod: string): ConstellationPoint[] => {
    switch (mod) {
      case 'BPSK':
        return [
          { i: 1, q: 0 },
          { i: -1, q: 0 },
        ];

      case 'QPSK':
        return [
          { i: 1, q: 1 },
          { i: -1, q: 1 },
          { i: -1, q: -1 },
          { i: 1, q: -1 },
        ].map(p => ({ i: p.i / Math.SQRT2, q: p.q / Math.SQRT2 }));

      case '8PSK':
        const points8 = [];
        for (let i = 0; i < 8; i++) {
          const angle = (2 * Math.PI * i) / 8 + Math.PI / 8;
          points8.push({
            i: Math.cos(angle),
            q: Math.sin(angle),
          });
        }
        return points8;

      case '16QAM':
        const points16: ConstellationPoint[] = [];
        for (let i = -3; i <= 3; i += 2) {
          for (let q = -3; q <= 3; q += 2) {
            points16.push({ i: i / 3, q: q / 3 });
          }
        }
        return points16;

      case '64QAM':
        const points64: ConstellationPoint[] = [];
        for (let i = -7; i <= 7; i += 2) {
          for (let q = -7; q <= 7; q += 2) {
            points64.push({ i: i / 7, q: q / 7 });
          }
        }
        return points64;

      default:
        return [];
    }
  };

  const clearPlot = () => {
    drawConstellation([]);
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-neon-cyan" />
            <CardTitle>Constellation Diagram</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-orbitron">
              {modulation}
            </Badge>
            <Badge variant={evm < 5 ? 'success' : 'danger'} className="text-xs">
              EVM: {evm.toFixed(1)}%
            </Badge>
          </div>
        </div>
        <CardDescription>
          I/Q constellation plot for digital modulation analysis
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
              className="rounded-lg border border-neon-cyan/30 bg-black"
            />
          </div>

          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowGrid(!showGrid)}
              >
                <Grid3x3 className="w-4 h-4 mr-1" />
                {showGrid ? 'Hide' : 'Show'} Grid
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowIdeal(!showIdeal)}
              >
                <Radio className="w-4 h-4 mr-1" />
                {showIdeal ? 'Hide' : 'Show'} Ideal
              </Button>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={clearPlot}
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Clear
            </Button>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-neon-cyan/30" />
              <span className="text-muted-foreground">Received Symbols</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-neon-red border border-neon-red" />
              <span className="text-muted-foreground">Ideal Constellation</span>
            </div>
          </div>

          {/* Quality Metrics */}
          <div className="grid grid-cols-3 gap-3 pt-2 border-t border-border">
            <div className="text-center">
              <div className="text-lg font-orbitron text-neon-cyan">{evm.toFixed(2)}%</div>
              <div className="text-xs text-muted-foreground">EVM</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-orbitron text-neon-green">
                {(Math.random() * 5 + 10).toFixed(1)} dB
              </div>
              <div className="text-xs text-muted-foreground">MER</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-orbitron text-neon-purple">
                {(Math.random() * 2 + 1).toFixed(1)}°
              </div>
              <div className="text-xs text-muted-foreground">Phase Error</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
