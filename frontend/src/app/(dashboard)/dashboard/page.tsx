'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Activity,
  Radio,
  Shield,
  Target,
  TrendingUp,
  TrendingDown,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  MapPin,
} from 'lucide-react';

export default function DashboardPage() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="group hover:glow-cyan transition-all cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-orbitron">
                  Active Receivers
                </p>
                <p className="text-4xl font-orbitron font-bold text-neon-cyan text-glow-cyan mt-2">
                  3
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <TrendingUp className="w-3 h-3 text-neon-green" />
                  <span className="text-xs text-neon-green">+1 today</span>
                </div>
              </div>
              <div className="w-16 h-16 rounded-lg bg-neon-cyan/20 flex items-center justify-center group-hover:glow-cyan transition-all">
                <Radio className="w-8 h-8 text-neon-cyan" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="group hover:glow-purple transition-all cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-orbitron">
                  Signals Detected
                </p>
                <p className="text-4xl font-orbitron font-bold text-neon-purple text-glow-purple mt-2">
                  127
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <TrendingUp className="w-3 h-3 text-neon-green" />
                  <span className="text-xs text-neon-green">+23 this hour</span>
                </div>
              </div>
              <div className="w-16 h-16 rounded-lg bg-neon-purple/20 flex items-center justify-center group-hover:glow-purple transition-all">
                <Activity className="w-8 h-8 text-neon-purple" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="group hover:glow-pink transition-all cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-orbitron">
                  Threat Alerts
                </p>
                <p className="text-4xl font-orbitron font-bold text-neon-pink text-glow-pink mt-2">
                  2
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <TrendingDown className="w-3 h-3 text-neon-green" />
                  <span className="text-xs text-neon-green">-3 from yesterday</span>
                </div>
              </div>
              <div className="w-16 h-16 rounded-lg bg-neon-pink/20 flex items-center justify-center group-hover:glow-pink transition-all">
                <Shield className="w-8 h-8 text-neon-pink" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="group hover:glow-green transition-all cursor-pointer">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-orbitron">
                  ML Accuracy
                </p>
                <p className="text-4xl font-orbitron font-bold text-neon-green glow-green mt-2">
                  97.2%
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <TrendingUp className="w-3 h-3 text-neon-green" />
                  <span className="text-xs text-neon-green">+0.3% this week</span>
                </div>
              </div>
              <div className="w-16 h-16 rounded-lg bg-neon-green/20 flex items-center justify-center glow-green transition-all">
                <Target className="w-8 h-8 text-neon-green" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Missions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5 text-neon-cyan" />
              Active Missions
            </CardTitle>
            <CardDescription>Currently running RF intelligence operations</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between p-4 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30 hover:glow-cyan transition-all cursor-pointer">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-orbitron text-sm font-bold text-neon-cyan">
                      MISSION-001
                    </h4>
                    <Badge variant="success" className="text-[10px]">
                      Active
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Urban area sweep - 915 MHz ISM band
                  </p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Radio className="w-3 h-3" />3 receivers
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      2h 34m
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-orbitron font-bold text-neon-cyan">45</div>
                  <div className="text-[10px] text-muted-foreground uppercase">Detections</div>
                </div>
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg bg-neon-purple/10 border border-neon-purple/30 hover:glow-purple transition-all cursor-pointer">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-orbitron text-sm font-bold text-neon-purple">
                      MISSION-002
                    </h4>
                    <Badge variant="warning" className="text-[10px]">
                      Paused
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Perimeter monitoring - 2.4 GHz WiFi
                  </p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Radio className="w-3 h-3" />
                      2 receivers
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      1h 12m
                    </span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-orbitron font-bold text-neon-purple">23</div>
                  <div className="text-[10px] text-muted-foreground uppercase">Detections</div>
                </div>
              </div>
            </div>

            <Button variant="neon" className="w-full" size="sm">
              <Zap className="w-4 h-4" />
              Start New Mission
            </Button>
          </CardContent>
        </Card>

        {/* Threat Alerts */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-neon-red" />
              Threat Alerts
            </CardTitle>
            <CardDescription>Recent security threats detected</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-start gap-3 p-4 rounded-lg bg-neon-red/10 border border-neon-red/30 hover:glow-red transition-all cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-neon-red/20 flex items-center justify-center flex-shrink-0 glow-red">
                <Shield className="w-5 h-5 text-neon-red" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-orbitron text-sm font-bold text-neon-red">
                    GPS SPOOFING
                  </h4>
                  <Badge variant="danger" className="text-[10px]">
                    Critical
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  Multiple GPS signals detected from same location - simulation attack suspected
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <MapPin className="w-3 h-3" />
                    37.7749°N, 122.4194°W
                  </span>
                  <span className="text-muted-foreground">2 min ago</span>
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-lg bg-neon-orange/10 border border-neon-orange/30 transition-all cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-neon-orange/20 flex items-center justify-center flex-shrink-0">
                <Activity className="w-5 h-5 text-neon-orange" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-orbitron text-sm font-bold text-neon-orange">
                    JAMMING DETECTED
                  </h4>
                  <Badge variant="warning" className="text-[10px]">
                    Medium
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  Barrage jamming on 915 MHz ISM band - SNR degradation observed
                </p>
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <MapPin className="w-3 h-3" />
                    37.8044°N, 122.2712°W
                  </span>
                  <span className="text-muted-foreground">15 min ago</span>
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-lg bg-muted/30 border border-border">
              <div className="w-10 h-10 rounded-lg bg-neon-green/20 flex items-center justify-center flex-shrink-0">
                <CheckCircle className="w-5 h-5 text-neon-green" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-orbitron text-sm font-bold text-muted-foreground line-through">
                    UNKNOWN SIGNAL
                  </h4>
                  <Badge variant="outline" className="text-[10px]">
                    Resolved
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Identified as commercial drone telemetry - whitelisted
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Spectrum Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-neon-cyan" />
            Live Spectrum Analysis
          </CardTitle>
          <CardDescription>Real-time RF spectrum waterfall display</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative h-64 bg-black/50 rounded-lg border border-neon-cyan/30 overflow-hidden">
            {/* Placeholder for spectrum analyzer */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center space-y-4">
                <Activity className="w-12 h-12 text-neon-cyan mx-auto animate-pulse" />
                <p className="text-sm text-muted-foreground font-orbitron">
                  Spectrum analyzer visualization component
                </p>
                <p className="text-xs text-muted-foreground">
                  Connect WebSocket for live data stream
                </p>
              </div>
            </div>

            {/* Frequency labels */}
            <div className="absolute bottom-0 left-0 right-0 flex justify-between px-4 py-2 text-xs text-muted-foreground font-mono">
              <span>900 MHz</span>
              <span>915 MHz</span>
              <span>930 MHz</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">System Health</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">CPU Usage</span>
              <span className="text-sm font-orbitron text-neon-cyan">34%</span>
            </div>
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full w-[34%] bg-gradient-to-r from-neon-cyan to-neon-purple glow-cyan" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Memory</span>
              <span className="text-sm font-orbitron text-neon-purple">56%</span>
            </div>
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full w-[56%] bg-gradient-to-r from-neon-purple to-neon-pink glow-purple" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Storage</span>
              <span className="text-sm font-orbitron text-neon-green">23%</span>
            </div>
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full w-[23%] bg-gradient-to-r from-neon-green to-neon-cyan glow-green" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Performance</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Latency</span>
              <span className="text-sm font-orbitron text-neon-green">342ms</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Throughput</span>
              <span className="text-sm font-orbitron text-neon-cyan">48 Mbps</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Uptime</span>
              <span className="text-sm font-orbitron text-neon-purple">99.97%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Last Sync</span>
              <span className="text-sm font-orbitron text-neon-pink">2s ago</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button variant="neon" size="sm" className="w-full justify-start">
              <Zap className="w-4 h-4" />
              Start Mission
            </Button>
            <Button variant="outline" size="sm" className="w-full justify-start">
              <Radio className="w-4 h-4" />
              Add Receiver
            </Button>
            <Button variant="outline" size="sm" className="w-full justify-start">
              <Activity className="w-4 h-4" />
              View Spectrum
            </Button>
            <Button variant="outline" size="sm" className="w-full justify-start">
              <Shield className="w-4 h-4" />
              Threat Analysis
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
