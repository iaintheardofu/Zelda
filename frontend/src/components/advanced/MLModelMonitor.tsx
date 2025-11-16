'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import {
  Brain,
  Activity,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Zap,
  Target,
  BarChart3,
} from 'lucide-react';

interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  inference_time_ms: number;
  total_predictions: number;
  correct_predictions: number;
  last_updated: string;
}

interface ConfusionCell {
  predicted: string;
  actual: string;
  count: number;
}

export function MLModelMonitor() {
  const [modulationModel, setModulationModel] = useState<ModelMetrics>({
    model_name: 'Modulation Classifier CNN',
    accuracy: 0.9534,
    precision: 0.9612,
    recall: 0.9489,
    f1_score: 0.9550,
    inference_time_ms: 12.3,
    total_predictions: 15847,
    correct_predictions: 15110,
    last_updated: new Date().toISOString(),
  });

  const [detectionModel, setDetectionModel] = useState<ModelMetrics>({
    model_name: 'Signal Detection Ensemble',
    accuracy: 0.9821,
    precision: 0.9756,
    recall: 0.9888,
    f1_score: 0.9822,
    inference_time_ms: 8.7,
    total_predictions: 23456,
    correct_predictions: 23036,
    last_updated: new Date().toISOString(),
  });

  const [recentPredictions, setRecentPredictions] = useState([
    { timestamp: '2025-11-15T10:23:45', predicted: 'QPSK', actual: 'QPSK', confidence: 0.97, correct: true },
    { timestamp: '2025-11-15T10:23:44', predicted: '16QAM', actual: '16QAM', confidence: 0.93, correct: true },
    { timestamp: '2025-11-15T10:23:43', predicted: '8PSK', actual: 'QPSK', confidence: 0.72, correct: false },
    { timestamp: '2025-11-15T10:23:42', predicted: 'BPSK', actual: 'BPSK', confidence: 0.99, correct: true },
    { timestamp: '2025-11-15T10:23:41', predicted: 'OFDM', actual: 'OFDM', confidence: 0.88, correct: true },
  ]);

  // Confusion matrix data (simplified for top modulations)
  const confusionMatrix: ConfusionCell[] = [
    { predicted: 'QPSK', actual: 'QPSK', count: 4523 },
    { predicted: 'QPSK', actual: '8PSK', count: 87 },
    { predicted: 'QPSK', actual: '16QAM', count: 23 },
    { predicted: '8PSK', actual: '8PSK', count: 3287 },
    { predicted: '8PSK', actual: 'QPSK', count: 124 },
    { predicted: '8PSK', actual: '16PSK', count: 45 },
    { predicted: '16QAM', actual: '16QAM', count: 2845 },
    { predicted: '16QAM', actual: '64QAM', count: 67 },
    { predicted: '16QAM', actual: 'QPSK', count: 34 },
    { predicted: 'BPSK', actual: 'BPSK', count: 1876 },
    { predicted: 'BPSK', actual: 'QPSK', count: 12 },
  ];

  const getMetricColor = (value: number, threshold: number = 0.9): string => {
    if (value >= threshold) return 'text-neon-green';
    if (value >= threshold - 0.1) return 'text-neon-cyan';
    if (value >= threshold - 0.2) return 'text-neon-yellow';
    return 'text-neon-red';
  };

  const getModelHealth = (accuracy: number): { status: string; color: string; icon: any } => {
    if (accuracy >= 0.95) return { status: 'Excellent', color: 'text-neon-green', icon: CheckCircle };
    if (accuracy >= 0.9) return { status: 'Good', color: 'text-neon-cyan', icon: Activity };
    if (accuracy >= 0.8) return { status: 'Fair', color: 'text-neon-yellow', icon: AlertCircle };
    return { status: 'Poor', color: 'text-neon-red', icon: AlertCircle };
  };

  const modulationHealth = getModelHealth(modulationModel.accuracy);
  const detectionHealth = getModelHealth(detectionModel.accuracy);

  // Simulate real-time metric updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate small fluctuations in metrics
      setModulationModel((prev) => ({
        ...prev,
        accuracy: Math.min(0.99, Math.max(0.90, prev.accuracy + (Math.random() - 0.5) * 0.01)),
        inference_time_ms: Math.max(8, prev.inference_time_ms + (Math.random() - 0.5) * 2),
        total_predictions: prev.total_predictions + Math.floor(Math.random() * 5),
      }));

      setDetectionModel((prev) => ({
        ...prev,
        accuracy: Math.min(0.99, Math.max(0.95, prev.accuracy + (Math.random() - 0.5) * 0.005)),
        inference_time_ms: Math.max(5, prev.inference_time_ms + (Math.random() - 0.5) * 1),
        total_predictions: prev.total_predictions + Math.floor(Math.random() * 8),
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      {/* Model Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Modulation Classifier Model */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-neon-purple" />
                <CardTitle className="text-base">Modulation Classifier</CardTitle>
              </div>
              <Badge variant={modulationHealth.status === 'Excellent' ? 'success' : 'default'}>
                {modulationHealth.status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Accuracy */}
              <div className="p-3 rounded-lg bg-background/50 border border-neon-purple/30">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">Overall Accuracy</span>
                  <modulationHealth.icon className={`w-4 h-4 ${modulationHealth.color}`} />
                </div>
                <div className={`text-3xl font-orbitron ${getMetricColor(modulationModel.accuracy, 0.95)}`}>
                  {(modulationModel.accuracy * 100).toFixed(2)}%
                </div>
                <Progress value={modulationModel.accuracy * 100} className="h-2 mt-2" />
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-2 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
                  <div className="text-xs text-muted-foreground">Precision</div>
                  <div className="text-lg font-orbitron text-neon-cyan">
                    {(modulationModel.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-green/10 border border-neon-green/30">
                  <div className="text-xs text-muted-foreground">Recall</div>
                  <div className="text-lg font-orbitron text-neon-green">
                    {(modulationModel.recall * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-purple/10 border border-neon-purple/30">
                  <div className="text-xs text-muted-foreground">F1 Score</div>
                  <div className="text-lg font-orbitron text-neon-purple">
                    {(modulationModel.f1_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-pink/10 border border-neon-pink/30">
                  <div className="text-xs text-muted-foreground">Inference</div>
                  <div className="text-lg font-orbitron text-neon-pink">
                    {modulationModel.inference_time_ms.toFixed(1)}ms
                  </div>
                </div>
              </div>

              {/* Prediction Stats */}
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Total Predictions</span>
                <span className="font-orbitron text-foreground">
                  {modulationModel.total_predictions.toLocaleString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Detection Model */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="w-5 h-5 text-neon-green" />
                <CardTitle className="text-base">Signal Detection</CardTitle>
              </div>
              <Badge variant={detectionHealth.status === 'Excellent' ? 'success' : 'default'}>
                {detectionHealth.status}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Accuracy */}
              <div className="p-3 rounded-lg bg-background/50 border border-neon-green/30">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">Detection Accuracy</span>
                  <detectionHealth.icon className={`w-4 h-4 ${detectionHealth.color}`} />
                </div>
                <div className={`text-3xl font-orbitron ${getMetricColor(detectionModel.accuracy, 0.95)}`}>
                  {(detectionModel.accuracy * 100).toFixed(2)}%
                </div>
                <Progress value={detectionModel.accuracy * 100} className="h-2 mt-2" />
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-2 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
                  <div className="text-xs text-muted-foreground">Precision</div>
                  <div className="text-lg font-orbitron text-neon-cyan">
                    {(detectionModel.precision * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-green/10 border border-neon-green/30">
                  <div className="text-xs text-muted-foreground">Recall</div>
                  <div className="text-lg font-orbitron text-neon-green">
                    {(detectionModel.recall * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-purple/10 border border-neon-purple/30">
                  <div className="text-xs text-muted-foreground">F1 Score</div>
                  <div className="text-lg font-orbitron text-neon-purple">
                    {(detectionModel.f1_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2 rounded-lg bg-neon-pink/10 border border-neon-pink/30">
                  <div className="text-xs text-muted-foreground">Inference</div>
                  <div className="text-lg font-orbitron text-neon-pink">
                    {detectionModel.inference_time_ms.toFixed(1)}ms
                  </div>
                </div>
              </div>

              {/* Prediction Stats */}
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Total Detections</span>
                <span className="font-orbitron text-foreground">
                  {detectionModel.total_predictions.toLocaleString()}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Predictions */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-neon-cyan" />
            <CardTitle>Recent Predictions</CardTitle>
          </div>
          <CardDescription>Live classification results with confidence scores</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {recentPredictions.map((pred, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-lg bg-background/50 border border-border"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      pred.correct ? 'bg-neon-green animate-pulse' : 'bg-neon-red'
                    }`}
                  />
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-orbitron text-sm text-neon-cyan">{pred.predicted}</span>
                      {!pred.correct && (
                        <>
                          <span className="text-xs text-muted-foreground">→</span>
                          <span className="text-xs text-neon-red">(actual: {pred.actual})</span>
                        </>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {new Date(pred.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-orbitron text-neon-purple">
                    {(pred.confidence * 100).toFixed(0)}%
                  </div>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Confusion Matrix Heatmap */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-neon-purple" />
            <CardTitle>Confusion Matrix (Top Classes)</CardTitle>
          </div>
          <CardDescription>Classification accuracy breakdown by modulation type</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {/* QPSK */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>QPSK</span>
                <span>{((4523 / (4523 + 87 + 23)) * 100).toFixed(1)}% correct</span>
              </div>
              <div className="flex gap-1 h-6">
                <div
                  className="bg-neon-green/80 rounded-l flex items-center justify-center text-xs font-bold text-black"
                  style={{ width: `${(4523 / 4633) * 100}%` }}
                >
                  4523
                </div>
                <div
                  className="bg-neon-yellow/60 flex items-center justify-center text-xs"
                  style={{ width: `${(87 / 4633) * 100}%` }}
                >
                  87
                </div>
                <div
                  className="bg-neon-red/40 rounded-r flex items-center justify-center text-xs"
                  style={{ width: `${(23 / 4633) * 100}%` }}
                >
                  23
                </div>
              </div>
            </div>

            {/* 8PSK */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>8PSK</span>
                <span>{((3287 / (3287 + 124 + 45)) * 100).toFixed(1)}% correct</span>
              </div>
              <div className="flex gap-1 h-6">
                <div
                  className="bg-neon-green/80 rounded-l flex items-center justify-center text-xs font-bold text-black"
                  style={{ width: `${(3287 / 3456) * 100}%` }}
                >
                  3287
                </div>
                <div
                  className="bg-neon-yellow/60 flex items-center justify-center text-xs"
                  style={{ width: `${(124 / 3456) * 100}%` }}
                >
                  124
                </div>
                <div
                  className="bg-neon-red/40 rounded-r flex items-center justify-center text-xs"
                  style={{ width: `${(45 / 3456) * 100}%` }}
                >
                  45
                </div>
              </div>
            </div>

            {/* 16QAM */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>16QAM</span>
                <span>{((2845 / (2845 + 67 + 34)) * 100).toFixed(1)}% correct</span>
              </div>
              <div className="flex gap-1 h-6">
                <div
                  className="bg-neon-green/80 rounded-l flex items-center justify-center text-xs font-bold text-black"
                  style={{ width: `${(2845 / 2946) * 100}%` }}
                >
                  2845
                </div>
                <div
                  className="bg-neon-yellow/60 flex items-center justify-center text-xs"
                  style={{ width: `${(67 / 2946) * 100}%` }}
                >
                  67
                </div>
                <div
                  className="bg-neon-red/40 rounded-r flex items-center justify-center text-xs"
                  style={{ width: `${(34 / 2946) * 100}%` }}
                >
                  34
                </div>
              </div>
            </div>

            {/* BPSK */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>BPSK</span>
                <span>{((1876 / (1876 + 12)) * 100).toFixed(1)}% correct</span>
              </div>
              <div className="flex gap-1 h-6">
                <div
                  className="bg-neon-green/80 rounded flex items-center justify-center text-xs font-bold text-black"
                  style={{ width: `${(1876 / 1888) * 100}%` }}
                >
                  1876
                </div>
                <div
                  className="bg-neon-yellow/60 rounded flex items-center justify-center text-xs"
                  style={{ width: `${(12 / 1888) * 100}%` }}
                >
                  12
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 text-xs pt-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-neon-green/80" />
                <span className="text-muted-foreground">Correct</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-neon-yellow/60" />
                <span className="text-muted-foreground">Minor Error</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-neon-red/40" />
                <span className="text-muted-foreground">Major Error</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Performance Trends */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-neon-green" />
              <CardTitle>Performance Trends</CardTitle>
            </div>
            <Badge variant="outline">Last 7 days</Badge>
          </div>
          <CardDescription>Model accuracy and throughput over time</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 rounded-lg bg-neon-green/10 border border-neon-green/30">
              <div className="text-2xl font-orbitron text-neon-green">+2.3%</div>
              <div className="text-xs text-muted-foreground mt-1">Accuracy Improvement</div>
            </div>
            <div className="text-center p-4 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30">
              <div className="text-2xl font-orbitron text-neon-cyan">-15%</div>
              <div className="text-xs text-muted-foreground mt-1">Latency Reduction</div>
            </div>
            <div className="text-center p-4 rounded-lg bg-neon-purple/10 border border-neon-purple/30">
              <div className="text-2xl font-orbitron text-neon-purple">99.8%</div>
              <div className="text-xs text-muted-foreground mt-1">Uptime</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
