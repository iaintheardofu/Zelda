# ZELDA Lovable Cloud Deployment Guide

> Complete guide to deploying ZELDA's cyberpunk RF intelligence platform using Lovable Cloud

## Overview

This guide walks you through deploying the ZELDA frontend to Lovable Cloud, which automatically handles:
- **Frontend Hosting**: Next.js app deployed to Vercel
- **Backend**: Supabase (PostgreSQL + Realtime + Auth + Storage)
- **Analytics**: Built-in React dashboards (no Grafana needed)
- **Real-time Streaming**: WebSocket connections to Python backend

**Your Lovable Project**: https://lovable.dev/projects/d1ec9287-a260-45d0-9590-8388563e4cb4

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Connect GitHub to Lovable](#connect-github-to-lovable)
3. [Enable Lovable Cloud](#enable-lovable-cloud)
4. [Configure Environment Variables](#configure-environment-variables)
5. [Deploy Frontend](#deploy-frontend)
6. [Deploy Python Backend](#deploy-python-backend)
7. [Verify Deployment](#verify-deployment)
8. [Monitor Usage & Costs](#monitor-usage--costs)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

✅ **Completed**:
- GitHub repository: https://github.com/iaintheardofu/Zelda
- Lovable account with project created
- Supabase project configured: `vwhbebhewtxuptbqddvp.supabase.co`
- Frontend built with Next.js 14 + TypeScript
- Backend API with FastAPI + WebSocket

✅ **You'll Need**:
- GitHub repository access (you have this)
- Lovable workspace with available Cloud balance
- Python backend hosting solution (Railway, Render, or DigitalOcean)

---

## Connect GitHub to Lovable

### Step 1: Access Your Lovable Project

1. Go to https://lovable.dev/projects/d1ec9287-a260-45d0-9590-8388563e4cb4
2. Open **Settings → Integrations → GitHub**

### Step 2: Connect Repository

1. Click **Connect GitHub Repository**
2. Select repository: `iaintheardofu/Zelda`
3. Choose branch: `main`
4. Set root directory: `/frontend` (since frontend code is in subfolder)
5. Click **Save**

Lovable will now sync with your GitHub repository and auto-deploy on every push to `main`.

### Step 3: Verify Sync

After connecting, you should see:
- ✅ Latest commit ID from GitHub
- ✅ Auto-deploy enabled
- ✅ Build status: Success

---

## Enable Lovable Cloud

### Understanding Lovable Cloud

Lovable Cloud includes:
- **Supabase backend** (already configured for ZELDA)
- **Auto-scaling hosting** (Vercel deployment)
- **Built-in authentication** (Supabase Auth)
- **Database & storage** (PostgreSQL + file storage)
- **Edge functions** (for serverless logic if needed)

### Enable Cloud for Your Project

1. In your Lovable project, go to **Settings → Account → Tools**
2. Find **Cloud** setting
3. Set to: **"Always allow"** (recommended) or **"Ask each time"**
4. Click **Save**

Your project is now Cloud-enabled! 🎉

### Configure Cloud Tools

Since ZELDA needs database access and real-time features, ensure these are enabled:

| Tool | Setting | Why |
|------|---------|-----|
| Read database | Always allow | Analytics needs to query detection/threat data |
| Modify database | Always allow | Schema updates for new features |
| Add data | Always allow | Seeding initial data |
| Read analytics | Always allow | Monitoring WebSocket performance |
| Configure auth | Always allow | User authentication setup |

---

## Configure Environment Variables

### Step 1: Set Supabase Credentials

Your frontend already has these in `.env.local`, but Lovable Cloud needs them too:

1. Go to **Cloud tab → Secrets**
2. Add the following secrets:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://vwhbebhewtxuptbqddvp.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Step 2: Set Backend API URLs

Add these for connecting to your Python backend:

```bash
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
NEXT_PUBLIC_WS_URL=wss://your-backend-domain.com
```

**Note**: Replace `your-backend-domain.com` with your actual Python backend URL after deploying it (see [Deploy Python Backend](#deploy-python-backend)).

### Step 3: Verify Secrets

After adding secrets:
1. Go to **Cloud tab → Secrets**
2. Verify all 4 secrets are listed
3. They should show as "Encrypted" (you won't see the values)

---

## Deploy Frontend

### Automatic Deployment

Lovable Cloud automatically deploys your frontend to Vercel when:
- You push to GitHub `main` branch
- You make changes in Lovable's visual editor
- You use Lovable's AI to update components

### Manual Deployment

If you need to trigger a manual deploy:

1. Go to your Lovable project
2. Click **Deploy** button (top right)
3. Wait for build to complete (~2-3 minutes)
4. Check **Logs** tab for any errors

### Verify Frontend Deployment

Your frontend will be available at a Vercel URL like:
```
https://zelda-rf-intelligence.vercel.app
```

Test these features:
- ✅ Landing page loads
- ✅ Dashboard accessible at `/dashboard`
- ✅ Spectrum analyzer at `/dashboard/spectrum`
- ✅ Analytics at `/dashboard/analytics`
- ✅ Authentication works (Supabase Auth)

---

## Deploy Python Backend

Lovable Cloud handles your frontend, but your Python FastAPI backend needs separate hosting. Here are recommended options:

### Option 1: Railway (Recommended)

**Why**: Simple, affordable, automatic deployments from GitHub.

1. Go to https://railway.app/
2. Click **New Project → Deploy from GitHub**
3. Select repository: `iaintheardofu/Zelda`
4. Set root directory: `/backend`
5. Add environment variables:
   ```bash
   DATABASE_URL=postgresql://...  # Supabase PostgreSQL URL
   REDIS_URL=redis://...          # Optional caching
   INFLUX_URL=http://...          # Optional time-series data
   ```
6. Deploy!

Your backend URL will be: `https://zelda-backend-production.up.railway.app`

### Option 2: Render

1. Go to https://render.com/
2. Create **New Web Service**
3. Connect GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn backend.api.app:app --host 0.0.0.0 --port $PORT`
6. Add environment variables (same as Railway)
7. Deploy!

### Option 3: DigitalOcean App Platform

1. Go to https://cloud.digitalocean.com/apps
2. Create app from GitHub
3. Configure Python buildpack
4. Set environment variables
5. Deploy!

### Update Frontend with Backend URL

After deploying your Python backend, update the Lovable Cloud secrets:

1. Go to **Cloud tab → Secrets**
2. Update:
   ```bash
   NEXT_PUBLIC_API_URL=https://your-actual-backend-url.com
   NEXT_PUBLIC_WS_URL=wss://your-actual-backend-url.com
   ```
3. Redeploy frontend (automatic after secret change)

---

## Verify Deployment

### Frontend Verification

1. **Landing Page**: Visit `https://your-frontend-url.vercel.app`
   - ✅ Cyberpunk theme loads
   - ✅ "ZELDA" title with glow effect
   - ✅ Feature cards visible
   - ✅ "Get Started" CTA works

2. **Dashboard**: Go to `/dashboard`
   - ✅ Sidebar navigation
   - ✅ 4 KPI cards (Receivers, Detections, Threats, ML Accuracy)
   - ✅ Active missions panel
   - ✅ Threat alerts panel

3. **Spectrum Analyzer**: Go to `/dashboard/spectrum`
   - ✅ WebSocket connects (green "Live" badge)
   - ✅ Waterfall display fills with data
   - ✅ Live detections appear
   - ✅ Power statistics update

4. **Analytics**: Go to `/dashboard/analytics`
   - ✅ Line chart renders
   - ✅ Pie chart shows signal types
   - ✅ Bar chart shows threat severity
   - ✅ Recent activity updates

### Backend Verification

1. **Health Check**: Visit `https://your-backend-url.com/api/health`
   - Should return: `{"status": "healthy"}`

2. **WebSocket**: Open browser console and run:
   ```javascript
   const ws = new WebSocket('wss://your-backend-url.com/ws');
   ws.onopen = () => console.log('✅ Connected!');
   ws.onmessage = (e) => console.log('📡 Message:', JSON.parse(e.data));
   ```
   - Should see "Connected!" and streaming data

3. **API Endpoints**: Test with curl:
   ```bash
   curl https://your-backend-url.com/api/status
   ```
   - Should return system status JSON

### Database Verification

1. Go to **Cloud tab → Database**
2. Verify tables exist:
   - ✅ `users`
   - ✅ `receivers`
   - ✅ `missions`
   - ✅ `signal_detections`
   - ✅ `threat_alerts`

3. Check Row Level Security (RLS) is enabled
4. Verify sample data if you seeded any

---

## Monitor Usage & Costs

### Understanding Lovable Cloud Pricing

- **Free tier**: $25/month Cloud + $1/month AI usage (all plans)
- **Resets**: Every 1st of the month at 00:00 UTC
- **Usage-based**: Beyond free tier, you pay for actual usage

### Check Your Balance

1. Go to **Settings → Cloud & AI balance**
2. View current balance
3. See usage breakdown per project
4. Check if automatic top-up is enabled

### Typical ZELDA Costs (Estimate)

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| Frontend hosting | 5,000 visits, WebSocket streaming | ~$5 |
| Supabase database | 1000 queries/day, 500 MB storage | ~$0 (within Supabase free tier) |
| AI features (if used) | Gemini 2.5 Flash for signal analysis | ~$2 |
| **Total** | Small-scale deployment | **~$7/month** |

**For production with 50k+ users**: ~$65-100/month

### Set Up Alerts

1. Go to **Settings → Cloud & AI balance**
2. Enable automatic top-up (paid plans only)
3. Set monthly charge limit: $100 (recommended starting point)
4. Enable email notifications for:
   - 50% usage reached
   - 80% usage reached
   - Balance low

---

## Troubleshooting

### Frontend Won't Deploy

**Symptom**: Build fails in Lovable

**Solutions**:
1. Check **Logs** tab for specific error
2. Verify `package.json` has all dependencies
3. Ensure TypeScript types are correct
4. Check if `.env.local` variables are set in Lovable Secrets

**Common errors**:
```
Module not found: Can't resolve '@/components/...'
→ Fix: Check all imports use correct paths from tsconfig.json
```

```
Type error: Property 'X' does not exist
→ Fix: Update types in frontend/src/types/index.ts
```

### WebSocket Won't Connect

**Symptom**: "Disconnected" badge in Spectrum Analyzer

**Solutions**:
1. Verify Python backend is running: `curl https://your-backend-url.com/api/health`
2. Check WebSocket URL in Lovable Secrets matches backend
3. Ensure backend CORS allows frontend domain:
   ```python
   # backend/api/app.py
   origins = [
       "https://your-frontend-url.vercel.app",
       "http://localhost:3000",  # for local dev
   ]
   ```
4. Check browser console for connection errors
5. Verify backend WebSocket routes are registered

### Database Connection Failed

**Symptom**: "Error connecting to database"

**Solutions**:
1. Go to **Cloud tab → Database → Overview**
2. Check instance status (should be green)
3. Verify Supabase credentials in Secrets
4. Check if RLS policies are too restrictive
5. Try upgrading instance size if data is too large

### No Real-Time Data Appearing

**Symptom**: Charts/components don't update

**Solutions**:
1. Check WebSocket connection (see above)
2. Verify Python backend is streaming data:
   ```python
   # backend/api/websocket_manager.py
   # Check start_data_streaming() is running
   ```
3. Check browser console for JavaScript errors
4. Verify custom hooks are subscribed to correct channels:
   ```typescript
   // Should see in console:
   // "Subscribed to channel: spectrum"
   ```

### High Cloud Costs

**Symptom**: Using more than $25/month

**Solutions**:
1. Check **Cloud tab → Logs** for unusual activity
2. Reduce WebSocket update rate:
   ```python
   # backend/api/websocket_manager.py
   spectrum_interval = 0.5  # Change from 0.1 (10 Hz) to 0.5 (2 Hz)
   ```
3. Upgrade to larger instance if many users (more efficient)
4. Implement caching with Redis for repeated queries
5. Monitor usage in **Settings → Cloud & AI balance**

### Analytics Charts Not Rendering

**Symptom**: Empty charts or "No signals detected"

**Solutions**:
1. Verify Python backend is sending data
2. Check WebSocket messages in browser console
3. Ensure custom hooks are receiving data:
   ```typescript
   console.log('Detections:', detections.detections.length);
   ```
4. Check for JavaScript errors in chart components
5. Verify data format matches TypeScript types

---

## Best Practices

### Development Workflow

1. **Local Development**:
   ```bash
   # Backend
   cd backend
   uvicorn api.app:app --reload

   # Frontend
   cd frontend
   npm run dev
   ```

2. **Test Locally** before pushing to GitHub
3. **Push to GitHub** → Automatic Lovable deployment
4. **Verify in Production** using Vercel preview URL
5. **Monitor** Cloud usage and logs

### Version Control

- **Pin stable versions** in Lovable after major features
- **Use Chat Mode** for planning complex changes
- **Test WebSocket** connections after backend updates
- **Backup database** before schema migrations

### Security

- ✅ Enable Row Level Security (RLS) on all Supabase tables
- ✅ Never commit secrets to GitHub
- ✅ Rotate Supabase API keys periodically
- ✅ Use environment-specific URLs (dev vs prod)
- ✅ Monitor logs for suspicious activity

### Performance

- **Optimize WebSocket**: Send only changed data
- **Use Redis**: Cache frequently accessed data
- **Compress responses**: Enable gzip in FastAPI
- **CDN assets**: Serve static files from Vercel Edge Network
- **Database indexing**: Add indexes on frequently queried columns

---

## Next Steps

### 🎨 Customize Your Dashboard

Use Lovable's Chat Mode to refine your UI:
```
Make the spectrum analyzer waterfall use a different color gradient -
dark blue to bright green to yellow for signal intensity.
```

### 🔐 Add Authentication

Your Supabase Auth is already configured. Enable user signup:
```
Add a signup page at /auth/signup with email and password fields.
Show error messages for invalid emails or weak passwords.
```

### 📊 Extend Analytics

Add more chart types:
```
Create a heatmap showing signal activity by time of day and frequency band.
Use the same cyberpunk color scheme as the rest of the dashboard.
```

### 🚀 Scale for Production

1. **Upgrade Lovable Cloud instance**: Settings → Advanced → Medium or Large
2. **Enable CDN**: Vercel automatically handles this
3. **Add monitoring**: Integrate with Sentry for error tracking
4. **Set up alerts**: Email notifications for critical threats
5. **Load testing**: Simulate 1000+ concurrent WebSocket connections

---

## Resources

### Documentation

- **Lovable Docs**: https://docs.lovable.dev/
- **Supabase Docs**: https://supabase.com/docs
- **Next.js Docs**: https://nextjs.org/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/

### ZELDA Project Docs

- **README.md**: Legal/ethical notice and feature overview
- **PROJECT_STATUS.md**: Current implementation status
- **WEBSOCKET_GUIDE.md**: WebSocket API reference and testing
- **ZELDA_MARKET_ANALYSIS_2025.md**: Market analysis and product strategy

### Support

- **Lovable Discord**: https://discord.gg/lovable-dev
- **GitHub Issues**: https://github.com/iaintheardofu/Zelda/issues
- **Lovable Support**: help@lovable.dev

---

## Summary

You now have:

✅ **Frontend**: Deployed to Vercel via Lovable Cloud
✅ **Backend**: Python FastAPI with WebSocket streaming
✅ **Database**: Supabase PostgreSQL with RLS
✅ **Analytics**: Custom React charts (no Grafana)
✅ **Real-time**: Live spectrum data and threat alerts
✅ **Monitoring**: Cloud usage tracking and alerts

**Your ZELDA platform is live!** 🎉

Visit your deployment:
- **Frontend**: https://your-frontend-url.vercel.app
- **Backend API**: https://your-backend-url.com
- **Lovable Project**: https://lovable.dev/projects/d1ec9287-a260-45d0-9590-8388563e4cb4

---

**Built with [Lovable](https://lovable.dev/) • Powered by [Supabase](https://supabase.com/) • Deployed on [Vercel](https://vercel.com/)**
