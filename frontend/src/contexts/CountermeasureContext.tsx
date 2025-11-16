'use client';

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useGlobalThreats } from './ThreatContext';
import { createClient } from '@/lib/supabase/client';
import type { ThreatAlert } from '@/types';

export type CountermeasureType =
  | 'frequency_hopping'
  | 'power_adjustment'
  | 'jamming_mitigation'
  | 'beamforming'
  | 'null_steering'
  | 'spectrum_evasion'
  | 'alert_only';

export interface CountermeasureAction {
  id: string;
  threat_id: string;
  threat_type: string;
  threat_severity: string;
  countermeasure_type: CountermeasureType;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  parameters: Record<string, any>;
  result?: string;
  executed_at?: string;
  completed_at?: string;
  created_at: string;
}

export interface CountermeasureConfig {
  enabled: boolean;
  auto_execute: boolean;
  threat_types: {
    jamming: {
      enabled: boolean;
      countermeasure: CountermeasureType;
      parameters: {
        hop_pattern?: 'random' | 'sequential' | 'adaptive';
        hop_rate_ms?: number;
        power_reduction_db?: number;
        backup_frequencies?: number[];
      };
    };
    spoofing: {
      enabled: boolean;
      countermeasure: CountermeasureType;
      parameters: {
        cross_check_sources?: boolean;
        switch_to_backup?: boolean;
        increase_monitoring?: boolean;
      };
    };
    unauthorized: {
      enabled: boolean;
      countermeasure: CountermeasureType;
      parameters: {
        evade_frequency?: boolean;
        reduce_transmission_power?: boolean;
        enable_stealth_mode?: boolean;
      };
    };
    interference: {
      enabled: boolean;
      countermeasure: CountermeasureType;
      parameters: {
        find_clear_channel?: boolean;
        adjust_power?: boolean;
        enable_filtering?: boolean;
      };
    };
  };
  severity_thresholds: {
    critical: {
      auto_execute: boolean;
      notify_command: boolean;
    };
    high: {
      auto_execute: boolean;
      notify_command: boolean;
    };
    medium: {
      auto_execute: boolean;
      notify_command: boolean;
    };
    low: {
      auto_execute: boolean;
      notify_command: boolean;
    };
  };
}

interface CountermeasureContextValue {
  config: CountermeasureConfig;
  actions: CountermeasureAction[];
  pendingCount: number;
  executingCount: number;

  // Configuration
  updateConfig: (config: Partial<CountermeasureConfig>) => void;
  resetConfig: () => void;

  // Actions
  executeCountermeasure: (threat: ThreatAlert, countermeasure?: CountermeasureType) => Promise<CountermeasureAction>;
  cancelCountermeasure: (actionId: string) => Promise<void>;
  getActionsForThreat: (threatId: string) => CountermeasureAction[];

  // Status
  isAutoExecuteEnabled: (severity: string) => boolean;
}

const defaultConfig: CountermeasureConfig = {
  enabled: true,
  auto_execute: false, // Manual approval by default for safety
  threat_types: {
    jamming: {
      enabled: true,
      countermeasure: 'frequency_hopping',
      parameters: {
        hop_pattern: 'adaptive',
        hop_rate_ms: 500,
        backup_frequencies: [868e6, 915e6, 2450e6],
      },
    },
    spoofing: {
      enabled: true,
      countermeasure: 'alert_only',
      parameters: {
        cross_check_sources: true,
        switch_to_backup: true,
        increase_monitoring: true,
      },
    },
    unauthorized: {
      enabled: true,
      countermeasure: 'spectrum_evasion',
      parameters: {
        evade_frequency: true,
        reduce_transmission_power: false,
        enable_stealth_mode: true,
      },
    },
    interference: {
      enabled: true,
      countermeasure: 'power_adjustment',
      parameters: {
        find_clear_channel: true,
        adjust_power: true,
        enable_filtering: true,
      },
    },
  },
  severity_thresholds: {
    critical: {
      auto_execute: true,
      notify_command: true,
    },
    high: {
      auto_execute: false,
      notify_command: true,
    },
    medium: {
      auto_execute: false,
      notify_command: false,
    },
    low: {
      auto_execute: false,
      notify_command: false,
    },
  },
};

const CountermeasureContext = createContext<CountermeasureContextValue | undefined>(undefined);

export function CountermeasureProvider({ children }: { children: React.ReactNode }) {
  const supabase = createClient();
  const { threats } = useGlobalThreats();

  const [config, setConfig] = useState<CountermeasureConfig>(defaultConfig);
  const [actions, setActions] = useState<CountermeasureAction[]>([]);

  // Load config from localStorage on mount
  useEffect(() => {
    const savedConfig = localStorage.getItem('zelda_countermeasure_config');
    if (savedConfig) {
      try {
        setConfig(JSON.parse(savedConfig));
      } catch (error) {
        console.error('Failed to load countermeasure config:', error);
      }
    }
  }, []);

  // Save config to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('zelda_countermeasure_config', JSON.stringify(config));
  }, [config]);

  // Auto-execute countermeasures for new threats
  useEffect(() => {
    if (!config.enabled || !config.auto_execute) return;

    threats.forEach(async (threat) => {
      // Check if we've already processed this threat
      const existingActions = actions.filter(a => a.threat_id === threat.id);
      if (existingActions.length > 0) return;

      // Check if auto-execute is enabled for this severity
      const severityConfig = config.severity_thresholds[threat.severity as keyof typeof config.severity_thresholds];
      if (!severityConfig?.auto_execute) return;

      // Execute countermeasure
      await executeCountermeasure(threat);
    });
  }, [threats, config, actions]);

  const updateConfig = useCallback((updates: Partial<CountermeasureConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  }, []);

  const resetConfig = useCallback(() => {
    setConfig(defaultConfig);
    localStorage.removeItem('zelda_countermeasure_config');
  }, []);

  const executeCountermeasure = useCallback(async (
    threat: ThreatAlert,
    countermeasureOverride?: CountermeasureType
  ): Promise<CountermeasureAction> => {
    // Determine countermeasure type
    const threatTypeConfig = config.threat_types[threat.type as keyof typeof config.threat_types];
    const countermeasureType = countermeasureOverride || threatTypeConfig?.countermeasure || 'alert_only';
    const parameters = threatTypeConfig?.parameters || {};

    // Create action record
    const action: CountermeasureAction = {
      id: `cm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      threat_id: threat.id,
      threat_type: threat.type,
      threat_severity: threat.severity,
      countermeasure_type: countermeasureType,
      status: 'pending',
      parameters,
      created_at: new Date().toISOString(),
    };

    setActions(prev => [action, ...prev]);

    // Execute countermeasure via Supabase Edge Function
    try {
      action.status = 'executing';
      action.executed_at = new Date().toISOString();
      setActions(prev => prev.map(a => a.id === action.id ? action : a));

      const { data: userData } = await supabase.auth.getUser();
      if (!userData.user) throw new Error('Not authenticated');

      const { data, error } = await supabase.functions.invoke('execute-countermeasure', {
        body: {
          threat_id: threat.id,
          threat_type: threat.type,
          threat_severity: threat.severity,
          countermeasure_type: countermeasureType,
          parameters,
          location: threat.location,
        },
      });

      if (error) throw error;

      action.status = 'completed';
      action.completed_at = new Date().toISOString();
      action.result = data?.message || 'Countermeasure executed successfully';

      setActions(prev => prev.map(a => a.id === action.id ? action : a));

      // Log to database
      await supabase.from('countermeasure_actions').insert({
        user_id: userData.user.id,
        threat_id: threat.id,
        countermeasure_type: countermeasureType,
        parameters,
        status: 'completed',
        result: action.result,
      });

      return action;
    } catch (error) {
      action.status = 'failed';
      action.result = error instanceof Error ? error.message : 'Unknown error';
      action.completed_at = new Date().toISOString();

      setActions(prev => prev.map(a => a.id === action.id ? action : a));

      throw error;
    }
  }, [config, supabase]);

  const cancelCountermeasure = useCallback(async (actionId: string) => {
    const action = actions.find(a => a.id === actionId);
    if (!action || action.status !== 'pending') return;

    setActions(prev => prev.map(a =>
      a.id === actionId
        ? { ...a, status: 'failed' as const, result: 'Cancelled by user' }
        : a
    ));
  }, [actions]);

  const getActionsForThreat = useCallback((threatId: string) => {
    return actions.filter(a => a.threat_id === threatId);
  }, [actions]);

  const isAutoExecuteEnabled = useCallback((severity: string) => {
    const severityConfig = config.severity_thresholds[severity as keyof typeof config.severity_thresholds];
    return config.enabled && config.auto_execute && (severityConfig?.auto_execute || false);
  }, [config]);

  const pendingCount = actions.filter(a => a.status === 'pending').length;
  const executingCount = actions.filter(a => a.status === 'executing').length;

  return (
    <CountermeasureContext.Provider
      value={{
        config,
        actions,
        pendingCount,
        executingCount,
        updateConfig,
        resetConfig,
        executeCountermeasure,
        cancelCountermeasure,
        getActionsForThreat,
        isAutoExecuteEnabled,
      }}
    >
      {children}
    </CountermeasureContext.Provider>
  );
}

export function useCountermeasures() {
  const context = useContext(CountermeasureContext);
  if (context === undefined) {
    throw new Error('useCountermeasures must be used within a CountermeasureProvider');
  }
  return context;
}
