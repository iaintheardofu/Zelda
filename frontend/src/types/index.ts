// ZELDA Frontend Type Definitions

export interface User {
  id: string;
  email: string;
  name?: string;
  role: 'admin' | 'operator' | 'viewer';
  avatar_url?: string;
  created_at: string;
  updated_at: string;
}

export interface Receiver {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  altitude: number;
  status: 'online' | 'offline' | 'error';
  sdr_type: 'KrakenSDR' | 'USRP' | 'RTL-SDR' | 'HackRF' | 'Custom';
  frequency: number;
  sample_rate: number;
  gain: number;
  last_seen: string;
}

export interface SignalDetection {
  id: string;
  timestamp: string;
  frequency: number;
  signal_type: string;
  confidence: number;
  power: number;
  bandwidth: number;
  modulation?: string;
  location?: {
    latitude: number;
    longitude: number;
    accuracy: number; // CEP in meters
  };
}

export interface JammingDetection {
  id: string;
  timestamp: string;
  jamming_type: 'barrage' | 'spot' | 'swept' | 'pulse' | 'follower' | 'deceptive';
  detected: boolean;
  confidence: number;
  power_db: number;
  bandwidth: number;
  center_frequency: number;
  description: string;
}

export interface SpoofingDetection {
  id: string;
  timestamp: string;
  threat_type: 'gps_meaconing' | 'gps_simulation' | 'imsi_catcher' | 'rogue_femtocell' | 'wifi_evil_twin' | 'wifi_rogue_ap';
  detected: boolean;
  confidence: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  details: Record<string, any>;
  recommendations: string[];
}

export interface Mission {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'active' | 'paused' | 'completed' | 'failed';
  created_by: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  receivers: string[]; // receiver IDs
  detections_count: number;
  threats_count: number;
  config: MissionConfig;
}

export interface MissionConfig {
  frequency_min: number;
  frequency_max: number;
  sample_rate: number;
  tdoa_enabled: boolean;
  ml_detection_enabled: boolean;
  jamming_detection_enabled: boolean;
  spoofing_detection_enabled: boolean;
  auto_mitigation: boolean;
}

export interface MissionResult {
  mission_id: string;
  timestamp: string;
  signal_detection?: SignalDetection;
  tdoa_location?: {
    latitude: number;
    longitude: number;
    altitude?: number;
    cep: number; // meters
  };
  jamming_status?: JammingDetection;
  spoofing_status?: SpoofingDetection;
  threat_level: 'none' | 'low' | 'medium' | 'high' | 'critical';
  recommendations: string[];
}

export interface SpectrumData {
  frequencies: number[];
  powers: number[];
  timestamp: string;
  receiver_id: string;
}

export interface WaterfallData {
  data: number[][]; // 2D array: [time][frequency]
  frequencies: number[];
  timestamps: string[];
  receiver_id: string;
}

export interface AnalyticsData {
  period: 'hour' | 'day' | 'week' | 'month';
  total_detections: number;
  threat_detections: number;
  ml_accuracy: number;
  tdoa_accuracy: number;
  uptime_percentage: number;
  detections_by_type: Record<string, number>;
  threats_by_severity: Record<string, number>;
  timeline: {
    timestamp: string;
    detections: number;
    threats: number;
  }[];
}

export interface WebSocketMessage {
  type: 'spectrum' | 'detection' | 'mission_update' | 'receiver_status' | 'threat_alert';
  data: any;
  timestamp: string;
}

export interface ThreatAlert {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: 'jamming' | 'spoofing' | 'unauthorized' | 'interference' | 'unknown';
  description: string;
  location?: {
    latitude: number;
    longitude: number;
  };
  recommended_action: string;
  acknowledged: boolean;
}

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'down';
  receivers: {
    total: number;
    online: number;
    offline: number;
    error: number;
  };
  missions: {
    active: number;
    pending: number;
    completed_today: number;
  };
  performance: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
  uptime: number; // seconds
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}
