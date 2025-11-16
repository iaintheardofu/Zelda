import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import type {
  SpectrumData,
  SignalDetection,
  ThreatAlert,
  Mission,
  Receiver,
} from '@/types';

export function useSpectrumData() {
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [history, setHistory] = useState<SpectrumData[]>([]);

  const { isConnected } = useWebSocket({
    channels: ['spectrum'],
    onMessage: (message) => {
      if (message.type === 'spectrum' && message.data) {
        const data: SpectrumData = {
          frequencies: message.data.frequencies,
          powers: message.data.powers,
          timestamp: message.data.timestamp,
          receiver_id: message.data.receiver_id || 'rx_001',
        };

        setSpectrumData(data);

        // Keep last 100 spectrum snapshots for waterfall
        setHistory((prev) => {
          const updated = [...prev, data];
          return updated.slice(-100);
        });
      }
    },
  });

  return {
    spectrumData,
    history,
    isConnected,
  };
}

export function useDetections() {
  const [detections, setDetections] = useState<SignalDetection[]>([]);
  const [latestDetection, setLatestDetection] = useState<SignalDetection | null>(null);

  const { isConnected } = useWebSocket({
    channels: ['detections'],
    onMessage: (message) => {
      if (message.type === 'detection' && message.data) {
        const detection: SignalDetection = {
          id: message.data.id,
          timestamp: message.data.timestamp,
          frequency: message.data.frequency,
          signal_type: message.data.signal_type,
          confidence: message.data.confidence,
          power: message.data.power,
          bandwidth: message.data.bandwidth,
          modulation: message.data.modulation,
          location: message.data.location,
        };

        setLatestDetection(detection);
        setDetections((prev) => [detection, ...prev].slice(0, 50)); // Keep last 50
      }
    },
  });

  const clearDetections = useCallback(() => {
    setDetections([]);
    setLatestDetection(null);
  }, []);

  return {
    detections,
    latestDetection,
    isConnected,
    clearDetections,
  };
}

export function useThreats() {
  const [threats, setThreats] = useState<ThreatAlert[]>([]);
  const [unacknowledgedCount, setUnacknowledgedCount] = useState(0);

  const { isConnected } = useWebSocket({
    channels: ['threats'],
    onMessage: (message) => {
      if (message.type === 'threat_alert' && message.data) {
        const threat: ThreatAlert = {
          id: message.data.id,
          timestamp: message.data.timestamp,
          severity: message.data.severity,
          type: message.data.type,
          description: message.data.description,
          location: message.data.location,
          recommended_action: message.data.recommended_action,
          acknowledged: false,
        };

        setThreats((prev) => [threat, ...prev].slice(0, 100)); // Keep last 100
        setUnacknowledgedCount((prev) => prev + 1);
      }
    },
  });

  const acknowledgeThre at = useCallback((threatId: string) => {
    setThreats((prev) =>
      prev.map((t) =>
        t.id === threatId ? { ...t, acknowledged: true } : t
      )
    );
    setUnacknowledgedCount((prev) => Math.max(0, prev - 1));
  }, []);

  const clearThreats = useCallback(() => {
    setThreats([]);
    setUnacknowledgedCount(0);
  }, []);

  return {
    threats,
    unacknowledgedCount,
    isConnected,
    acknowledgeThreat,
    clearThreats,
  };
}

export function useMissionUpdates(missionId?: string) {
  const [missionData, setMissionData] = useState<Mission | null>(null);
  const [activeMissions, setActiveMissions] = useState<Mission[]>([]);

  const { isConnected } = useWebSocket({
    channels: ['missions'],
    onMessage: (message) => {
      if (message.type === 'mission_update' && message.data) {
        const mission: Partial<Mission> = {
          id: message.data.mission_id,
          status: message.data.status,
          detections_count: message.data.detections_count,
          threats_count: message.data.threats_count,
        };

        // Update specific mission if watching one
        if (missionId && message.data.mission_id === missionId) {
          setMissionData((prev) => ({
            ...prev!,
            ...mission,
          }));
        }

        // Update active missions list
        setActiveMissions((prev) => {
          const index = prev.findIndex((m) => m.id === mission.id);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = { ...updated[index], ...mission };
            return updated;
          }
          return prev;
        });
      }
    },
  });

  return {
    missionData,
    activeMissions,
    isConnected,
  };
}

export function useReceiverStatus() {
  const [receivers, setReceivers] = useState<Receiver[]>([]);
  const [onlineCount, setOnlineCount] = useState(0);

  const { isConnected } = useWebSocket({
    channels: ['receivers'],
    onMessage: (message) => {
      if (message.type === 'receiver_status' && message.data) {
        const receiverUpdate: Partial<Receiver> = {
          id: message.data.receiver_id,
          status: message.data.status,
          frequency: message.data.frequency,
          sample_rate: message.data.sample_rate,
          gain: message.data.gain,
          last_seen: message.data.timestamp,
        };

        setReceivers((prev) => {
          const index = prev.findIndex((r) => r.id === receiverUpdate.id);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = { ...updated[index], ...receiverUpdate };
            return updated;
          }
          return prev;
        });

        // Update online count
        setReceivers((current) => {
          const online = current.filter((r) => r.status === 'online').length;
          setOnlineCount(online);
          return current;
        });
      }
    },
  });

  return {
    receivers,
    onlineCount,
    isConnected,
  };
}

// Hook for combined real-time dashboard data
export function useRealtimeDashboard() {
  const spectrum = useSpectrumData();
  const detections = useDetections();
  const threats = useThreats();
  const missions = useMissionUpdates();
  const receivers = useReceiverStatus();

  return {
    spectrum,
    detections,
    threats,
    missions,
    receivers,
    isConnected:
      spectrum.isConnected ||
      detections.isConnected ||
      threats.isConnected ||
      missions.isConnected ||
      receivers.isConnected,
  };
}
