// Supabase Client Configuration
import { createClient } from '@supabase/supabase-js';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

// Client for use in Client Components
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Client for use in Client Components with Next.js App Router
export const createSupabaseClient = () => createClientComponentClient();

// Database types
export type Database = {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          email: string;
          name: string | null;
          role: 'admin' | 'operator' | 'viewer';
          avatar_url: string | null;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<Database['public']['Tables']['users']['Row'], 'id' | 'created_at' | 'updated_at'>;
        Update: Partial<Database['public']['Tables']['users']['Insert']>;
      };
      missions: {
        Row: {
          id: string;
          name: string;
          description: string | null;
          status: 'pending' | 'active' | 'paused' | 'completed' | 'failed';
          created_by: string;
          created_at: string;
          started_at: string | null;
          completed_at: string | null;
          receivers: string[];
          detections_count: number;
          threats_count: number;
          config: any;
        };
        Insert: Omit<Database['public']['Tables']['missions']['Row'], 'id' | 'created_at'>;
        Update: Partial<Database['public']['Tables']['missions']['Insert']>;
      };
      receivers: {
        Row: {
          id: string;
          name: string;
          latitude: number;
          longitude: number;
          altitude: number;
          status: 'online' | 'offline' | 'error';
          sdr_type: string;
          frequency: number;
          sample_rate: number;
          gain: number;
          last_seen: string;
          created_at: string;
          updated_at: string;
        };
        Insert: Omit<Database['public']['Tables']['receivers']['Row'], 'id' | 'created_at' | 'updated_at'>;
        Update: Partial<Database['public']['Tables']['receivers']['Insert']>;
      };
      detections: {
        Row: {
          id: string;
          mission_id: string;
          timestamp: string;
          frequency: number;
          signal_type: string;
          confidence: number;
          power: number;
          bandwidth: number;
          modulation: string | null;
          location: any | null;
          created_at: string;
        };
        Insert: Omit<Database['public']['Tables']['detections']['Row'], 'id' | 'created_at'>;
        Update: Partial<Database['public']['Tables']['detections']['Insert']>;
      };
      threat_alerts: {
        Row: {
          id: string;
          mission_id: string;
          timestamp: string;
          severity: 'low' | 'medium' | 'high' | 'critical';
          type: string;
          description: string;
          location: any | null;
          recommended_action: string;
          acknowledged: boolean;
          acknowledged_by: string | null;
          acknowledged_at: string | null;
          created_at: string;
        };
        Insert: Omit<Database['public']['Tables']['threat_alerts']['Row'], 'id' | 'created_at'>;
        Update: Partial<Database['public']['Tables']['threat_alerts']['Insert']>;
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
};

export type Tables<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Row'];
export type Inserts<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Insert'];
export type Updates<T extends keyof Database['public']['Tables']> = Database['public']['Tables'][T]['Update'];
