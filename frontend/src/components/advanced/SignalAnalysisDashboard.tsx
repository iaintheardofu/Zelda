'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Activity,
  Radio,
  Zap,
  TrendingUp,
  Eye,
  Target,
  Waves,
  BarChart3,
} from 'lucide-react';
import { useState, useEffect } from 'react';

interface SignalMetrics {
  modulation: string;
  confidence: number;
  snr_db: number;
  symbol_rate?: number;
  bandwidth?: number;
  carrier_offset?: number;
}

interface DetectionMetrics {
  detector_type: string;
  detected: boolean;
  confidence: number;
  num_signals?: number;
}

export function SignalAnalysisDashboard() {
  const [signalMetrics, setSignalMetrics] = useState<SignalMetrics>({
    modulation: 'QPSK',
    confidence: 0.95,
    snr_db: 15.3,
    symbol_rate: 1000000,
    bandwidth: 2000000,
    carrier_offset: 1250,
  });

  const [detectionMetrics, setDetectionMetrics] = useState<DetectionMetrics[]>([
    { detector_type: 'Cyclostationary', detected: true, confidence: 0.92 },
    { detector_type: 'Energy (CFAR)', detected: true, confidence: 0.88 },
    { detector_type: 'Blind (Eigenvalue)', detected: true, confidence: 0.85 },
  ]);

  const [spectrumOccupancy, setSpectrumOccupancy] = useState(67);

  const getModulationColor = (modulation: string) => {
    const colors: Record<string, string> = {
      'BPSK': 'bg-blue-500',
      'QPSK': 'bg-cyan-500',
      '8PSK': 'bg-purple-500',
      '16QAM': 'bg-pink-500',
      '64QAM': 'bg-orange-500',
      'OFDM': 'bg-green-500',
      'FM': 'bg-yellow-500',
      'AM': 'bg-red-500',
    };
    return colors[modulation] || 'bg-gray-500';
  };

  const getSNRStatus = (snr: number) => {
    if (snr > 20) return { label: 'Excellent', color: 'text-neon-green' };
    if (snr > 10) return { label: 'Good', color: 'text-neon-cyan' };
    if (snr > 0) return { label: 'Fair', color: 'text-neon-yellow' };
    return { label: 'Poor', color: 'text-neon-red' };
  };

  const snrStatus = getSNRStatus(signalMetrics.snr_db);

  return (
    <div className="space-y-6">
      {/* Modulation Classification */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className="w-5 h-5 text-neon-purple" />
              <CardTitle>Modulation Classification</CardTitle>
            </div>
            <Badge variant="success">ML-Enhanced</Badge>
          </div>
          <CardDescription>
            Automatic modulation recognition using deep learning
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Primary Classification */}
            <div className="flex items-center justify-between p-4 rounded-lg bg-background/50 border border-neon-purple/30">
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-lg ${getModulationColor(signalMetrics.modulation)} flex items-center justify-center font-orbitron font-bold text-white`}>
                  {signalMetrics.modulation.substring(0, 2)}
                </div>
                <div>
                  <h3 className="text-2xl font-orbitron font-bold text-neon-purple">
                    {signalMetrics.modulation}
                  </h3>
                  <p className="text-xs text-muted-foreground">Detected Modulation</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-orbitron text-neon-cyan">
                  {(signalMetrics.confidence * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">Confidence</p>
              </div>
            </div>

            {/* Confidence Meter */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Classification Confidence</span>
                <span>{(signalMetrics.confidence * 100).toFixed(1)}%</span>
              </div>
              <Progress value={signalMetrics.confidence * 100} className="h-2" />
            </div>

            {/* Signal Parameters */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
                <div className="text-xs text-muted-foreground">Symbol Rate</div>
                <div className="text-lg font-orbitron text-neon-cyan">
                  {signalMetrics.symbol_rate ?
                    `${(signalMetrics.symbol_rate / 1e6).toFixed(2)} Msps` :
                    '--'
                  }
                </div>
              </div>
              <div className="p-3 rounded-lg bg-neon-purple/10 border border-neon-purple/30">
                <div className="text-xs text-muted-foreground">Bandwidth</div>
                <div className="text-lg font-orbitron text-neon-purple">
                  {signalMetrics.bandwidth ?
                    `${(signalMetrics.bandwidth / 1e6).toFixed(2)} MHz` :
                    '--'
                  }
                </div>
              </div>
            </div>

            {/* Top Candidates */}
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground font-orbitron">Alternative Classifications:</p>
              <div className="flex gap-2">
                <Badge variant="outline" className="text-xs">8PSK (12.3%)</Badge>
                <Badge variant="outline" className="text-xs">16QAM (8.7%)</Badge>
                <Badge variant="outline" className="text-xs">OQPSK (5.2%)</Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Multi-Algorithm Detection */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-neon-green" />
              <CardTitle>Multi-Algorithm Detection</CardTitle>
            </div>
            <Badge variant="success">
              {detectionMetrics.filter(d => d.detected).length}/{detectionMetrics.length} Detectors
            </Badge>
          </div>
          <CardDescription>
            Fusion of cyclostationary, energy, and blind detection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {detectionMetrics.map((metric, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-lg bg-background/50 border border-border"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${metric.detected ? 'bg-neon-green animate-pulse' : 'bg-muted'}`} />
                  <div>
                    <p className="font-medium text-sm">{metric.detector_type}</p>
                    <p className="text-xs text-muted-foreground">
                      {metric.detected ? 'Signal Detected' : 'No Signal'}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-orbitron text-neon-cyan">
                    {(metric.confidence * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                </div>
              </div>
            ))}

            {/* Fusion Result */}
            <div className="p-4 rounded-lg bg-neon-green/10 border border-neon-green/50 mt-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-neon-green" />
                  <span className="font-orbitron text-neon-green">Fused Decision</span>
                </div>
                <Badge variant="success" className="text-xs">
                  SIGNAL DETECTED (Majority Vote)
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Signal Quality Metrics */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-neon-cyan" />
            <CardTitle>Signal Quality Metrics</CardTitle>
          </div>
          <CardDescription>Real-time signal quality assessment</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* SNR Display */}
            <div className="p-4 rounded-lg bg-background/50 border border-neon-cyan/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">Signal-to-Noise Ratio</span>
                <span className={`text-sm font-orbitron ${snrStatus.color}`}>
                  {snrStatus.label}
                </span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-orbitron text-neon-cyan">
                  {signalMetrics.snr_db.toFixed(1)}
                </span>
                <span className="text-sm text-muted-foreground">dB</span>
              </div>
              <Progress
                value={Math.min((signalMetrics.snr_db + 10) / 30 * 100, 100)}
                className="h-2 mt-2"
              />
            </div>

            {/* Additional Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="text-xs text-muted-foreground mb-1">Carrier Offset</div>
                <div className="text-lg font-orbitron text-neon-purple">
                  {signalMetrics.carrier_offset ?
                    `${(signalMetrics.carrier_offset / 1e3).toFixed(2)} kHz` :
                    '--'
                  }
                </div>
              </div>
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="text-xs text-muted-foreground mb-1">EVM</div>
                <div className="text-lg font-orbitron text-neon-green">
                  2.3%
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Spectrum Occupancy */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Waves className="w-5 h-5 text-neon-pink" />
              <CardTitle>Spectrum Occupancy</CardTitle>
            </div>
            <Badge variant={spectrumOccupancy > 80 ? 'danger' : 'default'}>
              {spectrumOccupancy}%
            </Badge>
          </div>
          <CardDescription>
            Cognitive radio spectrum analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Occupied Channels</span>
                <span>{spectrumOccupancy}% of bandwidth</span>
              </div>
              <Progress value={spectrumOccupancy} className="h-3" />
            </div>

            {/* Spectrum Holes Detected */}
            <div className="p-3 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Available Spectrum Holes</span>
                <Badge variant="success">3 Found</Badge>
              </div>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">915.5 - 925.0 MHz</span>
                  <span className="text-neon-green">9.5 MHz BW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">2.405 - 2.420 GHz</span>
                  <span className="text-neon-green">15 MHz BW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">5.15 - 5.25 GHz</span>
                  <span className="text-neon-green">100 MHz BW</span>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="flex-1">
                <Eye className="w-4 h-4 mr-1" />
                View Holes
              </Button>
              <Button variant="outline" size="sm" className="flex-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                Optimize
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
