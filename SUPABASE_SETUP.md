# Supabase Database Setup for ZELDA

## Quick Setup (5 minutes)

### Step 1: Run Database Migration

1. Go to your Supabase project: https://supabase.com/dashboard/project/vwhbebhewtxuptbqddvp
2. Click "SQL Editor" in the left sidebar
3. Click "+ New query"
4. Copy and paste the entire content of `supabase/migrations/001_initial_schema.sql`
5. Click "Run" button
6. You should see "Success. No rows returned"

### Step 2: Verify Tables Created

1. Click "Table Editor" in the left sidebar
2. You should now see these tables:
   - users
   - receivers
   - missions
   - detections
   - threat_alerts

### Step 3: Enable Realtime

1. Go to "Database" → "Replication" in the left sidebar
2. Enable replication for these tables:
   - receivers
   - missions
   - detections
   - threat_alerts

### Step 4: Configure Authentication (Optional)

1. Go to "Authentication" in the left sidebar
2. Configure email provider:
   - Settings → Email → Enable email confirmations (optional)
3. Add OAuth providers (optional):
   - Settings → Providers → Enable GitHub, Google, etc.

## Verification

Run this query in SQL Editor to verify everything is set up:

```sql
SELECT
  'users' as table_name, COUNT(*) as row_count FROM public.users
UNION ALL
SELECT 'receivers', COUNT(*) FROM public.receivers
UNION ALL
SELECT 'missions', COUNT(*) FROM public.missions
UNION ALL
SELECT 'detections', COUNT(*) FROM public.detections
UNION ALL
SELECT 'threat_alerts', COUNT(*) FROM public.threat_alerts;
```

You should see all tables with 0 rows (they're empty to start).

## Connection Details

Your frontend is already configured with these credentials (in `frontend/.env.local`):

- **Project URL**: `https://vwhbebhewtxuptbqddvp.supabase.co`
- **Anon Key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## Next Steps

Once the database is set up, your ZELDA frontend will be able to:
- Store user profiles and authentication
- Save mission configurations
- Store detection history
- Track threat alerts
- Provide real-time updates via Supabase Realtime

## Troubleshooting

**Problem**: Migration fails with "permission denied"
**Solution**: Make sure you're running the query in the SQL Editor as the project owner

**Problem**: Tables not showing up
**Solution**: Refresh the page and check "Table Editor" → "public" schema

**Problem**: RLS blocking queries
**Solution**: The migration includes RLS policies. Make sure you're authenticated when querying from the frontend

## Optional: Add Sample Data

To test with sample data, run this in SQL Editor:

```sql
-- Insert sample receiver
INSERT INTO public.receivers (name, latitude, longitude, altitude, status, sdr_type, frequency, sample_rate, gain)
VALUES
  ('Receiver 1', 37.7749, -122.4194, 10.0, 'online', 'KrakenSDR', 915000000, 40000000, 20.0),
  ('Receiver 2', 37.8044, -122.2712, 15.0, 'online', 'USRP', 915000000, 40000000, 20.0),
  ('Receiver 3', 37.4419, -122.1430, 5.0, 'offline', 'RTL-SDR', 915000000, 40000000, 15.0);
```

---

**Status**: Database ready for ZELDA! 🚀
