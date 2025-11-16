'use client';

import { useThreats } from '@/hooks/useRealTimeData';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AlertTriangle, Shield, MapPin, CheckCircle } from 'lucide-react';
import { formatRelativeTime, getSeverityColor } from '@/lib/utils';
import { cn } from '@/lib/utils';

export function RealTimeThreats() {
  const { threats, unacknowledgedCount, isConnected, acknowledgeThreat } = useThreats();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-neon-red" />
            Threat Alerts
            {unacknowledgedCount > 0 && (
              <Badge variant="danger" className="ml-2 animate-pulse">
                {unacknowledgedCount} New
              </Badge>
            )}
          </CardTitle>
          <Badge variant={isConnected ? 'success' : 'danger'}>
            {isConnected ? 'Live' : 'Offline'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 max-h-[600px] overflow-y-auto">
        {threats.length === 0 ? (
          <div className="text-center py-12">
            <Shield className="w-16 h-16 text-neon-green mx-auto mb-4 opacity-50" />
            <p className="text-sm text-muted-foreground">No threats detected</p>
            <p className="text-xs text-muted-foreground mt-1">System monitoring active</p>
          </div>
        ) : (
          threats.map((threat) => (
            <div
              key={threat.id}
              className={cn(
                'flex items-start gap-3 p-4 rounded-lg border transition-all cursor-pointer',
                !threat.acknowledged && 'animate-pulse',
                threat.severity === 'critical' && 'bg-neon-red/10 border-neon-red/30 hover:glow-red',
                threat.severity === 'high' && 'bg-neon-orange/10 border-neon-orange/30',
                threat.severity === 'medium' && 'bg-neon-yellow/10 border-neon-yellow/30',
                threat.severity === 'low' && 'bg-neon-blue/10 border-neon-blue/30',
                threat.acknowledged && 'opacity-50 bg-muted/30 border-border'
              )}
            >
              <div
                className={cn(
                  'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
                  getSeverityColor(threat.severity),
                  threat.severity === 'critical' && 'glow-red'
                )}
              >
                {threat.acknowledged ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <Shield className="w-5 h-5" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className={cn('font-orbitron text-sm font-bold uppercase', !threat.acknowledged && 'text-glow-red')}>
                    {threat.type.replace(/_/g, ' ')}
                  </h4>
                  <Badge
                    variant={threat.severity === 'critical' ? 'danger' : 'warning'}
                    className="text-[10px]"
                  >
                    {threat.severity}
                  </Badge>
                  {threat.acknowledged && (
                    <Badge variant="outline" className="text-[10px]">
                      Acknowledged
                    </Badge>
                  )}
                </div>

                <p className="text-xs text-muted-foreground mb-2">
                  {threat.description}
                </p>

                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-4">
                    {threat.location && (
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <MapPin className="w-3 h-3" />
                        {threat.location.latitude.toFixed(4)}°, {threat.location.longitude.toFixed(4)}°
                      </span>
                    )}
                    <span className="text-muted-foreground">
                      {formatRelativeTime(threat.timestamp)}
                    </span>
                  </div>

                  {!threat.acknowledged && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-6 text-xs"
                      onClick={() => acknowledgeThreat(threat.id)}
                    >
                      Acknowledge
                    </Button>
                  )}
                </div>

                {threat.recommended_action && !threat.acknowledged && (
                  <div className="mt-2 p-2 rounded bg-neon-cyan/10 border border-neon-cyan/30">
                    <p className="text-xs text-neon-cyan">
                      <strong>Action:</strong> {threat.recommended_action}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
