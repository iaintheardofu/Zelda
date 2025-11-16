'use client';

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useThreats } from '@/hooks/useRealTimeData';
import { createClient } from '@/lib/supabase/client';
import type { ThreatAlert } from '@/types';

interface ThreatClassification {
  type: 'jamming' | 'spoofing' | 'unauthorized' | 'interference' | 'unknown';
  severity: 'critical' | 'high' | 'medium' | 'low';
  confidence: number;
  recommended_action: string;
}

interface ThreatContextValue {
  // Real-time threats from WebSocket
  threats: ThreatAlert[];
  unacknowledgedCount: number;
  isConnected: boolean;

  // Database threats
  dbThreats: ThreatAlert[];
  loadingDb: boolean;

  // Actions
  acknowledgeThreat: (threatId: string) => Promise<void>;
  classifySignal: (frequency: number, power: number, bandwidth: number, modulation?: string) => ThreatClassification;
  createThreat: (threat: Partial<ThreatAlert>) => Promise<void>;
  refreshDbThreats: () => Promise<void>;
  clearThreats: () => void;

  // Filtering
  filterBySeverity: (severity: string) => ThreatAlert[];
  filterByType: (type: string) => ThreatAlert[];
}

const ThreatContext = createContext<ThreatContextValue | undefined>(undefined);

export function ThreatProvider({ children }: { children: React.ReactNode }) {
  const supabase = createClient();

  // Real-time WebSocket threats
  const {
    threats: wsThreats,
    unacknowledgedCount,
    isConnected,
    acknowledgeThreat: ackWsThreat,
    clearThreats: clearWsThreats,
  } = useThreats();

  // Database threats
  const [dbThreats, setDbThreats] = useState<ThreatAlert[]>([]);
  const [loadingDb, setLoadingDb] = useState(true);

  // Combine WebSocket and database threats
  const allThreats = [...wsThreats, ...dbThreats];

  // Load threats from database
  const refreshDbThreats = useCallback(async () => {
    setLoadingDb(true);
    try {
      const { data, error } = await supabase
        .from('threats')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(100);

      if (error) throw error;

      if (data) {
        setDbThreats(data.map(threat => ({
          id: threat.id,
          timestamp: threat.created_at,
          severity: threat.severity,
          type: threat.classification,
          description: threat.description || `${threat.classification} detected`,
          location: threat.location ? {
            latitude: threat.location.coordinates[1],
            longitude: threat.location.coordinates[0],
          } : undefined,
          recommended_action: getRecommendedAction(threat.severity, threat.classification),
          acknowledged: threat.acknowledged || false,
        })));
      }
    } catch (error) {
      console.error('Error loading threats from database:', error);
    } finally {
      setLoadingDb(false);
    }
  }, [supabase]);

  // Load on mount
  useEffect(() => {
    refreshDbThreats();

    // Subscribe to real-time changes
    const channel = supabase
      .channel('threats-changes')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'threats'
      }, () => {
        refreshDbThreats();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [refreshDbThreats, supabase]);

  // Acknowledge threat (both WebSocket and database)
  const acknowledgeThreat = useCallback(async (threatId: string) => {
    // Acknowledge in WebSocket state
    ackWsThreat(threatId);

    // Acknowledge in database
    try {
      const { error } = await supabase
        .from('threats')
        .update({ acknowledged: true })
        .eq('id', threatId);

      if (error) throw error;

      // Update local state
      setDbThreats(prev =>
        prev.map(t =>
          t.id === threatId ? { ...t, acknowledged: true } : t
        )
      );
    } catch (error) {
      console.error('Error acknowledging threat:', error);
    }
  }, [ackWsThreat, supabase]);

  // ML-based signal classification
  const classifySignal = useCallback((
    frequency: number,
    power: number,
    bandwidth: number,
    modulation?: string
  ): ThreatClassification => {
    // Simple rule-based classification (replace with ML model in production)
    let type: ThreatClassification['type'] = 'unknown';
    let severity: ThreatClassification['severity'] = 'low';
    let confidence = 0.5;

    // Jamming detection: High power, wide bandwidth
    if (power > -30 && bandwidth > 20e6) {
      type = 'jamming';
      severity = 'critical';
      confidence = 0.95;
    }
    // GPS spoofing: ~1575 MHz, moderate power
    else if (frequency > 1574e6 && frequency < 1576e6 && power > -80) {
      type = 'spoofing';
      severity = 'high';
      confidence = 0.88;
    }
    // Unauthorized transmitter: Unexpected frequency, high power
    else if (power > -40 && !isAuthorizedBand(frequency)) {
      type = 'unauthorized';
      severity = 'high';
      confidence = 0.82;
    }
    // Interference: Moderate power in known bands
    else if (power > -60 && bandwidth > 5e6) {
      type = 'interference';
      severity = 'medium';
      confidence = 0.70;
    }

    return {
      type,
      severity,
      confidence,
      recommended_action: getRecommendedAction(severity, type),
    };
  }, []);

  // Create new threat
  const createThreat = useCallback(async (threat: Partial<ThreatAlert>) => {
    try {
      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) throw new Error('Not authenticated');

      const { error } = await supabase
        .from('threats')
        .insert({
          user_id: userData.user.id,
          classification: threat.type || 'unknown',
          severity: threat.severity || 'low',
          description: threat.description,
          location: threat.location ? {
            type: 'Point',
            coordinates: [threat.location.longitude, threat.location.latitude]
          } : null,
          acknowledged: false,
        });

      if (error) throw error;

      // Refresh to get the new threat
      await refreshDbThreats();
    } catch (error) {
      console.error('Error creating threat:', error);
    }
  }, [supabase, refreshDbThreats]);

  // Filter by severity
  const filterBySeverity = useCallback((severity: string) => {
    return allThreats.filter(t => t.severity === severity);
  }, [allThreats]);

  // Filter by type
  const filterByType = useCallback((type: string) => {
    return allThreats.filter(t => t.type === type);
  }, [allThreats]);

  // Clear all threats
  const clearThreats = useCallback(() => {
    clearWsThreats();
    setDbThreats([]);
  }, [clearWsThreats]);

  return (
    <ThreatContext.Provider
      value={{
        threats: allThreats,
        unacknowledgedCount,
        isConnected,
        dbThreats,
        loadingDb,
        acknowledgeThreat,
        classifySignal,
        createThreat,
        refreshDbThreats,
        clearThreats,
        filterBySeverity,
        filterByType,
      }}
    >
      {children}
    </ThreatContext.Provider>
  );
}

export function useGlobalThreats() {
  const context = useContext(ThreatContext);
  if (context === undefined) {
    throw new Error('useGlobalThreats must be used within a ThreatProvider');
  }
  return context;
}

// Helper functions
function isAuthorizedBand(frequency: number): boolean {
  // ISM bands
  if (frequency >= 902e6 && frequency <= 928e6) return true;  // 915 MHz ISM
  if (frequency >= 2.4e9 && frequency <= 2.5e9) return true;  // 2.4 GHz ISM
  if (frequency >= 5.725e9 && frequency <= 5.875e9) return true;  // 5.8 GHz ISM

  // GPS bands (authorized for reception only)
  if (frequency >= 1574e6 && frequency <= 1576e6) return true;  // GPS L1

  // Add more authorized bands as needed
  return false;
}

function getRecommendedAction(severity: string, type: string): string {
  const actions: Record<string, Record<string, string>> = {
    critical: {
      jamming: 'Activate countermeasures, alert command center, evacuate area if necessary',
      spoofing: 'Switch to backup navigation, verify all GPS signals, alert security',
      unauthorized: 'Triangulate source, dispatch security team, document for authorities',
      interference: 'Identify source, switch frequencies, notify spectrum management',
      unknown: 'Increase monitoring, collect more data, prepare defensive measures',
    },
    high: {
      jamming: 'Prepare countermeasures, increase monitoring, alert personnel',
      spoofing: 'Verify signal authenticity, enable cross-checks, monitor closely',
      unauthorized: 'Locate source, increase surveillance, prepare to notify authorities',
      interference: 'Switch to alternate frequencies, identify source',
      unknown: 'Intensify monitoring, classify threat, prepare response',
    },
    medium: {
      jamming: 'Monitor signal strength, log activity, prepare mitigation',
      spoofing: 'Compare with known-good signals, log anomaly',
      unauthorized: 'Log details, monitor pattern, assess threat level',
      interference: 'Document interference, assess impact',
      unknown: 'Continue monitoring, attempt classification',
    },
    low: {
      jamming: 'Log for reference, continue normal operations',
      spoofing: 'Document and monitor',
      unauthorized: 'Log occurrence, periodic monitoring',
      interference: 'Note for spectrum analysis',
      unknown: 'Passive monitoring',
    },
  };

  return actions[severity]?.[type] || 'Monitor and assess';
}
