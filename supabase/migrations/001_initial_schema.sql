-- ZELDA Supabase Database Schema
-- Initial Migration

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.users (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT,
  role TEXT NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')) DEFAULT 'viewer',
  avatar_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view their own profile" ON public.users
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON public.users
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Admins can view all users" ON public.users
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM public.users
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- Receivers table
CREATE TABLE IF NOT EXISTS public.receivers (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  name TEXT NOT NULL,
  latitude DOUBLE PRECISION NOT NULL,
  longitude DOUBLE PRECISION NOT NULL,
  altitude DOUBLE PRECISION DEFAULT 0,
  status TEXT CHECK (status IN ('online', 'offline', 'error')) DEFAULT 'offline',
  sdr_type TEXT NOT NULL,
  frequency BIGINT NOT NULL,
  sample_rate BIGINT NOT NULL,
  gain DOUBLE PRECISION DEFAULT 0,
  last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE public.receivers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view receivers" ON public.receivers
  FOR SELECT USING (true);

CREATE POLICY "Operators can manage receivers" ON public.receivers
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.users
      WHERE id = auth.uid() AND role IN ('admin', 'operator')
    )
  );

-- Missions table
CREATE TABLE IF NOT EXISTS public.missions (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT CHECK (status IN ('pending', 'active', 'paused', 'completed', 'failed')) DEFAULT 'pending',
  created_by UUID REFERENCES public.users(id) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  receivers UUID[] DEFAULT '{}',
  detections_count INTEGER DEFAULT 0,
  threats_count INTEGER DEFAULT 0,
  config JSONB DEFAULT '{}'::jsonb
);

ALTER TABLE public.missions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view missions" ON public.missions
  FOR SELECT USING (true);

CREATE POLICY "Operators can create missions" ON public.missions
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.users
      WHERE id = auth.uid() AND role IN ('admin', 'operator')
    )
  );

CREATE POLICY "Operators can update missions" ON public.missions
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.users
      WHERE id = auth.uid() AND role IN ('admin', 'operator')
    )
  );

-- Detections table
CREATE TABLE IF NOT EXISTS public.detections (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  mission_id UUID REFERENCES public.missions(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  frequency BIGINT NOT NULL,
  signal_type TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  power DOUBLE PRECISION NOT NULL,
  bandwidth DOUBLE PRECISION NOT NULL,
  modulation TEXT,
  location JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_detections_mission_id ON public.detections(mission_id);
CREATE INDEX idx_detections_timestamp ON public.detections(timestamp DESC);

ALTER TABLE public.detections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view detections" ON public.detections
  FOR SELECT USING (true);

CREATE POLICY "System can insert detections" ON public.detections
  FOR INSERT WITH CHECK (true);

-- Threat Alerts table
CREATE TABLE IF NOT EXISTS public.threat_alerts (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  mission_id UUID REFERENCES public.missions(id) ON DELETE CASCADE,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  severity TEXT CHECK (severity IN ('low', 'medium', 'high', 'critical')) NOT NULL,
  type TEXT NOT NULL,
  description TEXT NOT NULL,
  location JSONB,
  recommended_action TEXT NOT NULL,
  acknowledged BOOLEAN DEFAULT FALSE,
  acknowledged_by UUID REFERENCES public.users(id) ON DELETE SET NULL,
  acknowledged_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_threat_alerts_mission_id ON public.threat_alerts(mission_id);
CREATE INDEX idx_threat_alerts_timestamp ON public.threat_alerts(timestamp DESC);
CREATE INDEX idx_threat_alerts_acknowledged ON public.threat_alerts(acknowledged);

ALTER TABLE public.threat_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view threat alerts" ON public.threat_alerts
  FOR SELECT USING (true);

CREATE POLICY "System can insert threat alerts" ON public.threat_alerts
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Operators can acknowledge threats" ON public.threat_alerts
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM public.users
      WHERE id = auth.uid() AND role IN ('admin', 'operator')
    )
  );

-- Functions

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_receivers_updated_at
  BEFORE UPDATE ON public.receivers
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Function to get mission statistics
CREATE OR REPLACE FUNCTION get_mission_stats(mission_uuid UUID)
RETURNS JSON AS $$
DECLARE
  result JSON;
BEGIN
  SELECT json_build_object(
    'total_detections', COUNT(DISTINCT d.id),
    'total_threats', COUNT(DISTINCT t.id),
    'threat_breakdown', (
      SELECT json_object_agg(severity, count)
      FROM (
        SELECT severity, COUNT(*) as count
        FROM public.threat_alerts
        WHERE mission_id = mission_uuid
        GROUP BY severity
      ) AS severity_counts
    ),
    'detection_by_type', (
      SELECT json_object_agg(signal_type, count)
      FROM (
        SELECT signal_type, COUNT(*) as count
        FROM public.detections
        WHERE mission_id = mission_uuid
        GROUP BY signal_type
      ) AS type_counts
    )
  ) INTO result
  FROM public.detections d
  FULL OUTER JOIN public.threat_alerts t ON d.mission_id = t.mission_id
  WHERE d.mission_id = mission_uuid OR t.mission_id = mission_uuid;

  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Realtime
ALTER PUBLICATION supabase_realtime ADD TABLE public.receivers;
ALTER PUBLICATION supabase_realtime ADD TABLE public.missions;
ALTER PUBLICATION supabase_realtime ADD TABLE public.detections;
ALTER PUBLICATION supabase_realtime ADD TABLE public.threat_alerts;
