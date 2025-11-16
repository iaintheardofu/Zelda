'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Shield, AlertTriangle, Radio, Zap, MapPin, Check, X } from 'lucide-react';
import { useGlobalThreats } from '@/contexts/ThreatContext';
import { formatFrequency, formatRelativeTime } from '@/lib/utils';

interface ThreatClassifierProps {
  maxItems?: number;
  showFilters?: boolean;
  compact?: boolean;
}

export function ThreatClassifier({ maxItems = 10, showFilters = true, compact = false }: ThreatClassifierProps) {
  const { threats, acknowledgeThreat, filterBySeverity, filterByType, unacknowledgedCount } = useGlobalThreats();

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="w-4 h-4 text-neon-red" />;
      case 'high':
        return <Zap className="w-4 h-4 text-neon-orange" />;
      case 'medium':
        return <Radio className="w-4 h-4 text-neon-yellow" />;
      case 'low':
        return <Shield className="w-4 h-4 text-neon-cyan" />;
      default:
        return <Shield className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-neon-red/20 border-neon-red/50 text-neon-red';
      case 'high':
        return 'bg-neon-orange/20 border-neon-orange/50 text-neon-orange';
      case 'medium':
        return 'bg-neon-yellow/20 border-neon-yellow/50 text-neon-yellow';
      case 'low':
        return 'bg-neon-cyan/20 border-neon-cyan/50 text-neon-cyan';
      default:
        return 'bg-muted/20 border-muted/50 text-muted-foreground';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'jamming':
        return '🚫';
      case 'spoofing':
        return '🎭';
      case 'unauthorized':
        return '⚠️';
      case 'interference':
        return '📡';
      default:
        return '❓';
    }
  };

  if (compact) {
    return (
      <div className="space-y-2">
        {threats.slice(0, maxItems).map((threat) => (
          <div
            key={threat.id}
            className={`flex items-center justify-between p-3 rounded-lg border ${getSeverityColor(threat.severity)} transition-all`}
          >
            <div className="flex items-center gap-3 flex-1">
              {getSeverityIcon(threat.severity)}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-orbitron uppercase">{threat.type}</span>
                  <span className="text-xs text-muted-foreground">•</span>
                  <span className="text-xs text-muted-foreground">{formatRelativeTime(threat.timestamp)}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
                  {threat.description}
                </p>
              </div>
            </div>
            {!threat.acknowledged && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => acknowledgeThreat(threat.id)}
                className="ml-2"
              >
                <Check className="w-3 h-3" />
              </Button>
            )}
          </div>
        ))}
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-neon-red animate-pulse" />
              Threat Classifications
            </CardTitle>
            <CardDescription>
              Real-time threat detection and classification
            </CardDescription>
          </div>
          {unacknowledgedCount > 0 && (
            <Badge variant="danger" className="animate-pulse">
              {unacknowledgedCount} New
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {threats.length === 0 ? (
            <div className="text-center py-8">
              <Shield className="w-12 h-12 mx-auto text-muted-foreground opacity-50 mb-3" />
              <p className="text-sm text-muted-foreground">No threats detected</p>
              <p className="text-xs text-muted-foreground mt-1">System operating normally</p>
            </div>
          ) : (
            <>
              {threats.slice(0, maxItems).map((threat) => (
                <div
                  key={threat.id}
                  className={`p-4 rounded-lg border ${getSeverityColor(threat.severity)} transition-all hover:scale-[1.02]`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(threat.severity)}
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-orbitron uppercase font-bold">
                            {getTypeIcon(threat.type)} {threat.type}
                          </span>
                          <Badge
                            variant={threat.severity === 'critical' ? 'danger' : 'default'}
                            className="text-[10px]"
                          >
                            {threat.severity.toUpperCase()}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {formatRelativeTime(threat.timestamp)}
                        </p>
                      </div>
                    </div>
                    {!threat.acknowledged ? (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => acknowledgeThreat(threat.id)}
                        className="flex items-center gap-1"
                      >
                        <Check className="w-3 h-3" />
                        Acknowledge
                      </Button>
                    ) : (
                      <Badge variant="success" className="text-[10px]">
                        <Check className="w-3 h-3 mr-1" />
                        Acknowledged
                      </Badge>
                    )}
                  </div>

                  <p className="text-sm mb-3">{threat.description}</p>

                  {threat.location && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                      <MapPin className="w-3 h-3" />
                      <span className="font-mono">
                        {threat.location.latitude.toFixed(6)}, {threat.location.longitude.toFixed(6)}
                      </span>
                    </div>
                  )}

                  <div className="bg-background/50 rounded p-3">
                    <p className="text-xs text-muted-foreground mb-1">Recommended Action:</p>
                    <p className="text-xs font-medium">{threat.recommended_action}</p>
                  </div>
                </div>
              ))}

              {threats.length > maxItems && (
                <div className="text-center pt-2">
                  <p className="text-xs text-muted-foreground">
                    +{threats.length - maxItems} more threats
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
