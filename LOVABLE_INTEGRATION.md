'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  BarChart3,
  TrendingUp,
  Activity,
  Download,
  Calendar,
} from 'lucide-react';
import { useRealtimeDashboard } from '@/hooks/useRealTimeData';
import { formatRelativeTime } from '@/lib/utils';

// Simple inline chart components (no external dependencies)
function SimpleLineChart({ data, color = '#00ffff' }: { data: number[]; color?: string }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg viewBox="0 0 100 100" className="w-full h-32" preserveAspectRatio="none">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
        className="drop-shadow-[0_0_8px_rgba(0,255,255,0.8)]"
      />
      <polyline
        points={`0,100 ${points} 100,100`}
        fill={`${color}20`}
        stroke="none"
      />
    </svg>
  );
}

function SimpleBarChart({ data, labels, color = '#00ffff' }: { data: number[]; labels: string[]; color?: string }) {
  const max = Math.max(...data);

  return (
    <div className="flex items-end justify-between gap-2 h-32">
      {data.map((value, index) => {
        const height = (value / max) * 100;
        return (
          <div key={index} className="flex-1 flex flex-col items-center gap-1">
            <div className="text-xs font-orbitron text-muted-foreground">{value}</div>
            <div
              className="w-full rounded-t transition-all"
              style={{
                height: `${height}%`,
                background: `linear-gradient(to top, ${color}, ${color}80)`,
                boxShadow: `0 0 10px ${color}80`,
              }}
            />
            <div className="text-[10px] text-muted-foreground uppercase">{labels[index]}</div>
          </div>
        );
      })}
    </div>
  );
}

function SimplePieChart({ data, labels, colors }: { data: number[]; labels: string[]; colors: string[] }) {
  const total = data.reduce((sum, val) => sum + val, 0);
  let currentAngle = 0;

  return (
    <div className="flex items-center gap-6">
      <svg viewBox="0 0 100 100" className="w-32 h-32">
        {data.map((value, index) => {
          const percentage = value / total;
          const angle = percentage * 360;
          const x1 = 50 + 45 * Math.cos((currentAngle - 90) * Math.PI / 180);
          const y1 = 50 + 45 * Math.sin((currentAngle - 90) * Math.PI / 180);
          const x2 = 50 + 45 * Math.cos((currentAngle + angle - 90) * Math.PI / 180);
          const y2 = 50 + 45 * Math.sin((currentAngle + angle - 90) * Math.PI / 180);
          const largeArc = angle > 180 ? 1 : 0;

          const path = `M 50 50 L ${x1} ${y1} A 45 45 0 ${largeArc} 1 ${x2} ${y2} Z`;
          currentAngle += angle;

          return (
            <path
              key={index}
              d={path}
              fill={colors[index]}
              className="transition-all hover:opacity-80 cursor-pointer"
              style={{ filter: `drop-shadow(0 0 5px ${colors[index]})` }}
            />
          );
        })}
      </svg>
      <div className="flex-1 space-y-2">
        {labels.map((label, index) => (
          <div key={index} className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-sm"
                style={{ background: colors[index], boxShadow: `0 0 5px ${colors[index]}` }}
              />
              <span>{label}</span>
            </div>
            <span className="font-orbitron text-muted-foreground">
              {((data[index] / total) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const { detections, threats, isConnected } = useRealtimeDashboard();
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('1h');

  // Generate analytics data from real-time streams
  const [detectionTimeline, setDetectionTimeline] = useState<number[]>([]);
  const [signalTypes, setSignalTypes] = useState<Record<string, number>>({});
  const [threatsBySeverity, setThreatsBySeverity] = useState<Record<string, number>>({});

  useEffect(() => {
    // Update detection timeline (last 20 intervals)
    setDetectionTimeline(prev => {
      const updated = [...prev, detections.detections.length];
      return updated.slice(-20);
    });

    // Count signal types
    const types: Record<string, number> = {};
    detections.detections.forEach(d => {
      types[d.signal_type] = (types[d.signal_type] || 0) + 1;
    });
    setSignalTypes(types);

    // Count threats by severity
    const severities: Record<string, number> = { low: 0, medium: 0, high: 0, critical: 0 };
    threats.threats.forEach(t => {
      severities[t.severity] = (severities[t.severity] || 0) + 1;
    });
    setThreatsBySeverity(severities);
  }, [detections.detections, threats.threats]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-cyan uppercase tracking-wider">
            ANALYTICS
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time intelligence metrics and performance data
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant={isConnected ? 'success' : 'danger'} className="animate-pulse">
            {isConnected ? 'Live' : 'Disconnected'}
          </Badge>
          <div className="flex gap-2">
            {(['1h', '24h', '7d'] as const).map(range => (
              <Button
                key={range}
                variant={timeRange === range ? 'neon' : 'outline'}
                size="sm"
                onClick={() => setTimeRange(range)}
              >
                {range}
              </Button>
            ))}
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs uppercase text-muted-foreground font-orbitron">Total Detections</h3>
              <TrendingUp className="w-4 h-4 text-neon-green" />
            </div>
            <div className="text-3xl font-orbitron font-bold text-neon-cyan text-glow-cyan">
              {detections.detections.length}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Last hour
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs uppercase text-muted-foreground font-orbitron">Threat Alerts</h3>
              <Activity className="w-4 h-4 text-neon-red" />
            </div>
            <div className="text-3xl font-orbitron font-bold text-neon-pink text-glow-pink">
              {threats.threats.length}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {threats.unacknowledgedCount} unacknowledged
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs uppercase text-muted-foreground font-orbitron">Avg Confidence</h3>
              <BarChart3 className="w-4 h-4 text-neon-purple" />
            </div>
            <div className="text-3xl font-orbitron font-bold text-neon-purple text-glow-purple">
              {detections.detections.length > 0
                ? ((detections.detections.reduce((sum, d) => sum + d.confidence, 0) / detections.detections.length) * 100).toFixed(1)
                : '0'}%
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Detection accuracy
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xs uppercase text-muted-foreground font-orbitron">Uptime</h3>
              <Calendar className="w-4 h-4 text-neon-green" />
            </div>
            <div className="text-3xl font-orbitron font-bold text-neon-green glow-green">
              99.9%
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Last 30 days
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Detection Timeline */}
        <Card>
          <CardHeader>
            <CardTitle>Detection Timeline</CardTitle>
            <CardDescription>Signals detected over time</CardDescription>
          </CardHeader>
          <CardContent>
            <SimpleLineChart
              data={detectionTimeline.length > 0 ? detectionTimeline : [0, 0, 0, 0, 0]}
              color="#00ffff"
            />
            <div className="flex justify-between mt-4 text-xs text-muted-foreground">
              <span>{detectionTimeline.length > 0 ? detectionTimeline.length * 5 : 0}m ago</span>
              <span>Now</span>
            </div>
          </CardContent>
        </Card>

        {/* Signal Types Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Signal Types</CardTitle>
            <CardDescription>Distribution by signal type</CardDescription>
          </CardHeader>
          <CardContent>
            {Object.keys(signalTypes).length > 0 ? (
              <SimplePieChart
                data={Object.values(signalTypes)}
                labels={Object.keys(signalTypes)}
                colors={['#00ffff', '#ff1166', '#aa00ff', '#00ff00', '#ff8800', '#ffff00']}
              />
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No signals detected yet</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Threat Severity Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Threat Severity</CardTitle>
            <CardDescription>Breakdown by severity level</CardDescription>
          </CardHeader>
          <CardContent>
            <SimpleBarChart
              data={[
                threatsBySeverity.low || 0,
                threatsBySeverity.medium || 0,
                threatsBySeverity.high || 0,
                threatsBySeverity.critical || 0,
              ]}
              labels={['Low', 'Medium', 'High', 'Critical']}
              color="#ff1166"
            />
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Latest detections and threats</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {[...detections.detections.slice(0, 3), ...threats.threats.slice(0, 3)]
                .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                .slice(0, 5)
                .map((item, index) => {
                  const isDetection = 'signal_type' in item;
                  return (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                    >
                      <div className="flex-1">
                        <p className="text-sm font-orbitron">
                          {isDetection ? item.signal_type : item.type.replace(/_/g, ' ')}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatRelativeTime(item.timestamp)}
                        </p>
                      </div>
                      <Badge variant={isDetection ? 'info' : 'danger'} className="text-[10px]">
                        {isDetection ? 'Detection' : 'Threat'}
                      </Badge>
                    </div>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
