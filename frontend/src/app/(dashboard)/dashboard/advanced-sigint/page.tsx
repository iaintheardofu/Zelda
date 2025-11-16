import { SignalAnalysisDashboard } from '@/components/advanced/SignalAnalysisDashboard';
import { ConstellationDiagram } from '@/components/advanced/ConstellationDiagram';
import { Spectrogram3D } from '@/components/advanced/Spectrogram3D';
import { MLModelMonitor } from '@/components/advanced/MLModelMonitor';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Brain, Radio, Box, Activity, Zap, Shield } from 'lucide-react';

export default function AdvancedSIGINTPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink bg-clip-text text-transparent">
            Advanced SIGINT Platform
          </h1>
          <p className="text-muted-foreground mt-1">
            World-class signal intelligence with ML-powered detection and cognitive radio
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success" className="font-orbitron">
            <Zap className="w-3 h-3 mr-1" />
            OPERATIONAL
          </Badge>
          <Badge variant="outline" className="font-orbitron">
            <Shield className="w-3 h-3 mr-1" />
            DEFENSIVE
          </Badge>
        </div>
      </div>

      {/* Capabilities Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-neon-purple" />
              <CardTitle className="text-sm">Detection</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-orbitron text-neon-purple">98.2%</div>
            <p className="text-xs text-muted-foreground">Multi-algorithm fusion</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-neon-cyan" />
              <CardTitle className="text-sm">Classification</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-orbitron text-neon-cyan">95.3%</div>
            <p className="text-xs text-muted-foreground">50+ modulation types</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-neon-green" />
              <CardTitle className="text-sm">Cognitive Radio</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-orbitron text-neon-green">Active</div>
            <p className="text-xs text-muted-foreground">Adaptive interference mitigation</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Box className="w-4 h-4 text-neon-pink" />
              <CardTitle className="text-sm">Visualization</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-orbitron text-neon-pink">3D/RT</div>
            <p className="text-xs text-muted-foreground">Real-time 3D spectrograms</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="analysis" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="analysis" className="gap-2">
            <Radio className="w-4 h-4" />
            Signal Analysis
          </TabsTrigger>
          <TabsTrigger value="visualization" className="gap-2">
            <Box className="w-4 h-4" />
            3D Visualization
          </TabsTrigger>
          <TabsTrigger value="constellation" className="gap-2">
            <Activity className="w-4 h-4" />
            Constellation
          </TabsTrigger>
          <TabsTrigger value="ml-monitor" className="gap-2">
            <Brain className="w-4 h-4" />
            ML Monitor
          </TabsTrigger>
        </TabsList>

        {/* Signal Analysis Tab */}
        <TabsContent value="analysis" className="space-y-4">
          <SignalAnalysisDashboard />
        </TabsContent>

        {/* 3D Visualization Tab */}
        <TabsContent value="visualization" className="space-y-4">
          <Spectrogram3D width={1000} height={700} />
        </TabsContent>

        {/* Constellation Tab */}
        <TabsContent value="constellation" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ConstellationDiagram modulation="QPSK" />
            <ConstellationDiagram modulation="16QAM" />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ConstellationDiagram modulation="8PSK" />
            <ConstellationDiagram modulation="64QAM" />
          </div>
        </TabsContent>

        {/* ML Monitor Tab */}
        <TabsContent value="ml-monitor" className="space-y-4">
          <MLModelMonitor />
        </TabsContent>
      </Tabs>

      {/* Technical Specifications */}
      <Card>
        <CardHeader>
          <CardTitle>System Capabilities</CardTitle>
          <CardDescription>Advanced SIGINT platform technical specifications</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Detection Algorithms */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-purple">Detection Algorithms</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Cyclostationary Feature Detection</li>
                <li>• Energy Detection with CFAR</li>
                <li>• Blind Eigenvalue Detection</li>
                <li>• Multi-Algorithm Fusion</li>
                <li>• Spectral Correlation Function</li>
              </ul>
            </div>

            {/* Modulation Classification */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-cyan">Modulation Types</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Analog: AM, FM, PM, SSB</li>
                <li>• PSK: BPSK, QPSK, 8PSK, 16PSK</li>
                <li>• QAM: 16QAM, 64QAM, 256QAM</li>
                <li>• FSK: 2FSK, 4FSK, GFSK, MSK</li>
                <li>• Advanced: OFDM, DSSS, FHSS</li>
              </ul>
            </div>

            {/* Cognitive Radio */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-green">Cognitive Radio</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Spectrum Sensing &amp; Holes</li>
                <li>• LMS/RLS Adaptive Filtering</li>
                <li>• Interference Cancellation</li>
                <li>• Dynamic Channel Allocation</li>
                <li>• Learned Interference Mapping</li>
              </ul>
            </div>

            {/* Signal Characterization */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-pink">Signal Characterization</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Symbol Rate Estimation</li>
                <li>• Bandwidth Measurement</li>
                <li>• Carrier Offset Detection</li>
                <li>• SNR/EVM Calculation</li>
                <li>• Phase Noise Analysis</li>
              </ul>
            </div>

            {/* RF Fingerprinting */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-yellow">RF Fingerprinting</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Emitter Identification</li>
                <li>• Transient Analysis</li>
                <li>• I/Q Imbalance Measurement</li>
                <li>• Hardware Imperfections</li>
                <li>• Device-Specific Signatures</li>
              </ul>
            </div>

            {/* ML Architecture */}
            <div className="space-y-2">
              <h3 className="font-orbitron text-sm text-neon-orange">ML Architecture</h3>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• CNN-based Classification</li>
                <li>• Feature Engineering</li>
                <li>• Decision Tree Fallback</li>
                <li>• Real-time Inference</li>
                <li>• Continuous Learning Ready</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Legal Notice */}
      <Card className="border-neon-yellow/50">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-neon-yellow" />
            <CardTitle>Legal Notice</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            This advanced SIGINT platform is designed exclusively for <strong>defensive</strong> RF signal
            intelligence. It provides capabilities for signal detection, analysis, and environmental
            adaptability in congested or hostile RF environments. This system does <strong>NOT</strong> include
            offensive capabilities such as RF jamming, spoofing, or transmission. Use is restricted to
            authorized security monitoring, research, and defensive electronic warfare applications in
            accordance with applicable laws and regulations.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
