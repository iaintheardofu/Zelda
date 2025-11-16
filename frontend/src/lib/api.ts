// API Client for Python FastAPI Backend
import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  ApiResponse,
  SignalDetection,
  JammingDetection,
  SpoofingDetection,
  MissionResult,
  SystemStatus,
  Receiver,
} from '@/types';

class ZeldaApiClient {
  private client: AxiosInstance;
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          window.dispatchEvent(new Event('unauthorized'));
        }
        return Promise.reject(error);
      }
    );
  }

  // System Status
  async getSystemStatus(): Promise<ApiResponse<SystemStatus>> {
    const response = await this.client.get('/api/status');
    return response.data;
  }

  // Receiver Management
  async getReceivers(): Promise<ApiResponse<Receiver[]>> {
    const response = await this.client.get('/api/receivers');
    return response.data;
  }

  async getReceiver(id: string): Promise<ApiResponse<Receiver>> {
    const response = await this.client.get(`/api/receivers/${id}`);
    return response.data;
  }

  async addReceiver(receiver: Partial<Receiver>): Promise<ApiResponse<Receiver>> {
    const response = await this.client.post('/api/receivers', receiver);
    return response.data;
  }

  async updateReceiver(id: string, updates: Partial<Receiver>): Promise<ApiResponse<Receiver>> {
    const response = await this.client.put(`/api/receivers/${id}`, updates);
    return response.data;
  }

  async deleteReceiver(id: string): Promise<ApiResponse<void>> {
    const response = await this.client.delete(`/api/receivers/${id}`);
    return response.data;
  }

  // Mission Management
  async startMission(config: any): Promise<ApiResponse<{ mission_id: string }>> {
    const response = await this.client.post('/api/missions/start', config);
    return response.data;
  }

  async stopMission(missionId: string): Promise<ApiResponse<void>> {
    const response = await this.client.post(`/api/missions/${missionId}/stop`);
    return response.data;
  }

  async pauseMission(missionId: string): Promise<ApiResponse<void>> {
    const response = await this.client.post(`/api/missions/${missionId}/pause`);
    return response.data;
  }

  async resumeMission(missionId: string): Promise<ApiResponse<void>> {
    const response = await this.client.post(`/api/missions/${missionId}/resume`);
    return response.data;
  }

  // Signal Processing
  async processSignal(iqData: number[], config?: any): Promise<ApiResponse<MissionResult>> {
    const response = await this.client.post('/api/process', {
      iq_data: iqData,
      config,
    });
    return response.data;
  }

  // TDOA Geolocation
  async calculateTDOA(delays: number[], receiverPositions: any[]): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tdoa/calculate', {
      delays,
      receiver_positions: receiverPositions,
    });
    return response.data;
  }

  // ML Signal Detection
  async detectSignals(iqData: number[]): Promise<ApiResponse<SignalDetection[]>> {
    const response = await this.client.post('/api/ml/detect', {
      iq_data: iqData,
    });
    return response.data;
  }

  // Jamming Detection
  async detectJamming(iqData: number[]): Promise<ApiResponse<JammingDetection>> {
    const response = await this.client.post('/api/ew/jamming/detect', {
      iq_data: iqData,
    });
    return response.data;
  }

  // Spoofing Detection
  async detectSpoofing(data: any): Promise<ApiResponse<SpoofingDetection[]>> {
    const response = await this.client.post('/api/ew/spoofing/detect', data);
    return response.data;
  }

  // Anti-Jam Processing
  async applyAntiJam(iqData: number[], jammingType?: string): Promise<ApiResponse<{ processed_iq: number[]; snr_improvement: number }>> {
    const response = await this.client.post('/api/ew/antijam/process', {
      iq_data: iqData,
      jamming_type: jammingType,
    });
    return response.data;
  }

  // Analytics
  async getAnalytics(period: 'hour' | 'day' | 'week' | 'month'): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/analytics?period=${period}`);
    return response.data;
  }

  // Health Check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.status === 200;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const api = new ZeldaApiClient();

// Export class for custom instances
export default ZeldaApiClient;
