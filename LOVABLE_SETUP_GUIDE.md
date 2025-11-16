# ZELDA Lovable Integration Guide

## Step-by-Step: Connect Your GitHub Repo to Lovable

### Prerequisites
- GitHub account with access to `iaintheardofu/Zelda` repository
- Lovable account (create at https://lovable.dev/)

### Connection Steps

#### 1. Access Lovable
1. Go to https://lovable.dev/
2. Click "Sign In" or "Get Started"
3. Sign in with your preferred method (GitHub recommended)

#### 2. Create New Project
1. Once logged in, click "New Project" or "Create Project"
2. Name your project: **"ZELDA RF Intelligence Platform"**
3. Select project type: **"Full-stack Application"**

#### 3. Connect GitHub Repository
1. In your Lovable project, look for **Settings** (gear icon or settings menu)
2. Find **"GitHub Integration"** or **"Version Control"** section
3. Click **"Connect GitHub"** button
4. Authorize Lovable to access your GitHub account (if first time)
5. Select:
   - **Organization/User**: `iaintheardofu`
   - **Repository**: `Zelda`
   - **Branch**: `main` (default)
6. Click **"Connect Repository"** or **"Enable Sync"**
7. Confirm two-way sync is enabled (your Lovable changes will push to GitHub)

#### 4. Initial Import (Optional)
If Lovable asks whether to:
- **Import existing code**: Select NO (we'll build fresh frontend)
- **Start from scratch**: Select YES
- The Python backend will remain in the repo, we're adding a new frontend

#### 5. Verify Connection
1. Check that you see a GitHub icon or badge showing "Connected"
2. Verify you can see the branch name (`main`)
3. Test by making a small change in Lovable and checking if it appears in GitHub

---

## Architecture Overview

### Hybrid System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     ZELDA Web Application                    │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐          ┌──────────────────────────┐
│   Frontend (Lovable) │          │  Backend (Hybrid)        │
│                      │          │                          │
│  Next.js + React     │◄────────►│  Python FastAPI          │
│  TypeScript          │   REST   │  (RF Processing)         │
│  Tailwind CSS        │   API    │  - TDOA Geolocation     │
│  Shadcn/ui           │          │  - ML Signal Detection   │
│  WebSocket Client    │◄────────►│  - Defensive EW Suite    │
│                      │   WS     │                          │
└──────────────────────┘          └──────────────────────────┘
         │                                    │
         │                                    │
         ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────────┐
│  Supabase            │          │  Hardware/SDR            │
│  (User Management)   │          │                          │
│  - Authentication    │          │  - KrakenSDR             │
│  - User Profiles     │          │  - USRP                  │
│  - Mission History   │          │  - RTL-SDR               │
│  - Analytics Data    │          │  - HackRF                │
│  - Real-time DB      │          │                          │
└──────────────────────┘          └──────────────────────────┘
```

### Technology Stack

**Frontend (Built in Lovable)**:
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Shadcn/ui
- **State Management**: Zustand or React Context
- **Charts**: Recharts / D3.js for RF visualization
- **Maps**: Mapbox GL / Leaflet for geolocation
- **Real-time**: WebSocket client for live RF data

**Backend (Existing + New)**:
- **RF Processing**: Python FastAPI (existing)
  - TDOA calculations
  - ML inference
  - EW threat detection
  - SDR hardware interface
- **User Management**: Supabase
  - Authentication (email, OAuth)
  - User profiles and permissions
  - Mission history storage
  - Real-time database for dashboards
- **API Gateway**: Next.js API routes (middleware)
  - Auth verification
  - Request proxying to Python backend
  - Rate limiting

**Database**:
- **Supabase PostgreSQL**: User data, missions, analytics
- **Python Backend**: In-memory for real-time RF data
- **Optional**: Redis for caching

---

## Features to Build

### 1. Real-time RF Visualization
- **Spectrum Analyzer**: Live waterfall display
- **Signal Detection**: Real-time signal classification overlay
- **Geolocation Map**: Live TDOA positioning on interactive map
- **Threat Indicators**: Jamming/spoofing alerts with visual cues

### 2. Mission Control Dashboard
- **Receiver Configuration**: Add/remove/configure SDR receivers
- **Mission Management**: Start/stop/pause missions
- **Live Status**: Real-time health monitoring of receivers
- **Threat Analysis**: Unified threat dashboard showing all detections

### 3. User Authentication
- **Login/Signup**: Email/password + OAuth (GitHub, Google)
- **Role-Based Access**: Admin, Operator, Viewer roles
- **Team Management**: Multi-user organizations
- **Audit Logs**: Track all user actions

### 4. Data Analytics
- **Historical Data**: Browse past missions and detections
- **Performance Metrics**: ML model accuracy over time
- **Reports**: Generate PDF/CSV reports
- **Visualizations**: Charts and graphs for trends

---

## File Structure (What We'll Build)

```
zelda/
├── frontend/                    # NEW - Lovable-built frontend
│   ├── app/                    # Next.js App Router
│   │   ├── (auth)/            # Auth pages
│   │   │   ├── login/
│   │   │   └── signup/
│   │   ├── (dashboard)/       # Protected routes
│   │   │   ├── layout.tsx     # Dashboard layout
│   │   │   ├── page.tsx       # Main dashboard
│   │   │   ├── missions/      # Mission control
│   │   │   ├── receivers/     # Receiver management
│   │   │   ├── analytics/     # Data analytics
│   │   │   └── settings/      # User settings
│   │   ├── api/               # Next.js API routes
│   │   │   ├── auth/          # Auth endpoints
│   │   │   └── proxy/         # Proxy to Python backend
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Landing page
│   ├── components/             # React components
│   │   ├── ui/                # Shadcn/ui components
│   │   ├── spectrum/          # RF visualization
│   │   ├── map/               # Geolocation map
│   │   ├── dashboard/         # Dashboard widgets
│   │   └── charts/            # Analytics charts
│   ├── lib/                   # Utilities
│   │   ├── supabase.ts       # Supabase client
│   │   ├── api.ts            # API client for Python backend
│   │   └── websocket.ts      # WebSocket client
│   ├── hooks/                 # Custom React hooks
│   ├── types/                 # TypeScript types
│   ├── public/               # Static assets
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── next.config.js
│
├── backend/                    # EXISTING - Python backend
│   ├── core/                  # RF processing (keep as-is)
│   ├── api/                   # FastAPI endpoints
│   │   └── app.py            # UPDATE - Add CORS, WebSocket
│   └── ...
│
├── supabase/                   # NEW - Supabase configuration
│   ├── migrations/            # Database migrations
│   ├── functions/             # Edge functions
│   └── config.toml
│
└── ...
```

---

## Next Steps

### After Connecting to Lovable

1. **I will build**:
   - Complete Next.js frontend structure
   - Supabase database schema
   - All React components
   - API integration layer
   - Real-time WebSocket client

2. **You will configure**:
   - Supabase project (I'll provide schema)
   - Environment variables in Lovable
   - Python backend CORS settings

3. **We will deploy**:
   - Frontend: Lovable auto-deploys to Vercel
   - Python backend: Your choice (Railway, Render, AWS, etc.)
   - Supabase: Managed cloud service

---

## Environment Variables Needed

Once connected, you'll need to set these in Lovable:

```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-key

# Python Backend
NEXT_PUBLIC_API_URL=http://localhost:8000  # Development
NEXT_PUBLIC_WS_URL=ws://localhost:8000     # Development
# (Update for production deployment)

# Optional
NEXT_PUBLIC_MAPBOX_TOKEN=your-mapbox-token  # For maps
```

---

## Ready to Build?

Once you've completed the GitHub connection in Lovable (Steps 1-5 above), let me know and I'll:

1. Build the entire frontend application
2. Create the Supabase schema
3. Update the Python backend for CORS and WebSocket
4. Provide deployment instructions

**Estimated Build Time**: 2-3 hours for complete system

---

## Support

If you encounter issues connecting GitHub to Lovable:
- Check GitHub permissions (Lovable needs read/write access)
- Ensure you're the owner/admin of the Zelda repository
- Try disconnecting and reconnecting
- Contact Lovable support: support@lovable.dev
