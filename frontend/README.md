# ZELDA Frontend - Cyberpunk Dashboard

Interactive cyberpunk-themed RF Signal Intelligence dashboard built with Next.js 14, TypeScript, and Tailwind CSS.

## 🎨 Design Theme

Cyberpunk aesthetic featuring:
- **Neon Color Palette**: Cyan, Pink, Purple, Green accents
- **Glow Effects**: Dynamic neon glows on interactive elements
- **Scan Lines**: Retro CRT monitor effects
- **Grid Backgrounds**: Futuristic grid patterns
- **Glitch Animations**: Subtle cyberpunk glitches
- **Orbitron Font**: Bold, futuristic typography

## 🚀 Features

### Implemented

- ✅ **Landing Page** - Cyberpunk hero section with feature showcase
- ✅ **Dashboard Layout** - Sidebar navigation with real-time status
- ✅ **Main Dashboard** - System overview with stats, missions, and threats
- ✅ **Spectrum Analyzer** - RF spectrum waterfall visualization (UI ready)
- ✅ **Cyberpunk UI Components** - Button, Card, Badge with neon effects
- ✅ **Authentication Hook** - Supabase auth integration
- ✅ **Type Definitions** - Complete TypeScript types
- ✅ **API Client** - Python backend integration
- ✅ **WebSocket Client** - Real-time data streaming

### Ready to Build in Lovable

- 🔨 **Missions Page** - Mission creation and management
- 🔨 **Receivers Page** - SDR receiver configuration
- 🔨 **Threats Page** - Threat analysis and alerts
- 🔨 **Analytics Page** - Historical data and reports
- 🔨 **Settings Page** - User preferences and configuration
- 🔨 **Auth Pages** - Login/signup with Supabase
- 🔨 **Real-time Spectrum** - Connect WebSocket for live data
- 🔨 **Geolocation Map** - TDOA positioning visualization

## 📁 Structure

```
frontend/
├── src/
│   ├── app/                      # Next.js App Router
│   │   ├── layout.tsx           # Root layout with fonts
│   │   ├── page.tsx             # Landing page
│   │   └── (dashboard)/         # Dashboard routes
│   │       ├── layout.tsx       # Dashboard sidebar layout
│   │       └── dashboard/
│   │           ├── page.tsx     # Main dashboard
│   │           └── spectrum/
│   │               └── page.tsx # Spectrum analyzer
│   ├── components/
│   │   └── ui/                  # UI components
│   │       ├── button.tsx       # Cyberpunk button
│   │       ├── card.tsx         # Glowing cards
│   │       └── badge.tsx        # Status badges
│   ├── lib/                     # Utilities
│   │   ├── api.ts              # Python backend client
│   │   ├── supabase.ts         # Supabase client
│   │   ├── websocket.ts        # WebSocket client
│   │   └── utils.ts            # Helpers
│   ├── hooks/
│   │   └── useAuth.ts          # Authentication hook
│   ├── types/
│   │   └── index.ts            # TypeScript types
│   └── styles/
│       └── globals.css         # Cyberpunk theme
├── package.json
├── tsconfig.json
├── tailwind.config.ts          # Cyberpunk colors & animations
└── next.config.js
```

## 🎨 Cyberpunk Theme

### Colors

```typescript
// Neon Colors (from globals.css)
--neon-cyan: 180 100% 50%      // #00FFFF
--neon-pink: 340 100% 55%      // #FF1166
--neon-purple: 280 100% 60%    // #AA00FF
--neon-green: 120 100% 50%     // #00FF00
--neon-orange: 30 100% 55%     // #FF8800
--neon-red: 0 100% 50%         // #FF0000
```

### Typography

- **Headings**: Orbitron (bold, uppercase, with glow)
- **Body**: Inter (clean, readable)
- **Monospace**: For technical data

### Effects

- `.glow-cyan` - Cyan neon glow
- `.glow-pink` - Pink neon glow
- `.glow-purple` - Purple neon glow
- `.text-glow-cyan` - Text glow effect
- `.cyber-panel` - Cyberpunk panel style
- `.scanlines` - CRT scan line overlay
- `.grid-bg` - Grid background pattern

## 🛠️ Setup

### Install Dependencies

```bash
cd frontend
npm install
```

### Environment Variables

Create `.env.local`:

```bash
# Already configured with your Supabase project
NEXT_PUBLIC_SUPABASE_URL=https://vwhbebhewtxuptbqddvp.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-key-here

# Python Backend (local dev)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Optional: Mapbox for maps
NEXT_PUBLIC_MAPBOX_TOKEN=your-token
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 🔗 Integration

### Supabase Setup

1. Run database migration:
   ```bash
   # In Supabase SQL Editor
   # Run: ../supabase/migrations/001_initial_schema.sql
   ```

2. Enable Realtime for tables:
   - receivers
   - missions
   - detections
   - threat_alerts

### Python Backend

```bash
# Start FastAPI backend
cd ../backend
uvicorn api.app:app --reload
```

Backend will run on `http://localhost:8000`

### WebSocket Connection

```typescript
import { wsClient } from '@/lib/websocket';

// Connect
wsClient.connect();

// Subscribe to spectrum data
wsClient.on('spectrum', (message) => {
  console.log('Spectrum data:', message.data);
});

// Subscribe to detections
wsClient.on('detection', (message) => {
  console.log('New detection:', message.data);
});
```

## 🎯 Next Steps in Lovable

### 1. Connect to Lovable

Follow `../LOVABLE_SETUP_GUIDE.md`

### 2. Build Missing Pages

Tell Lovable:

```
"Using the existing dashboard layout, create these pages:

1. Missions page (/dashboard/missions):
   - List of all missions with status
   - Create new mission form
   - Mission details view

2. Receivers page (/dashboard/receivers):
   - Grid of receiver cards
   - Add/edit receiver forms
   - Real-time status indicators

3. Authentication pages:
   - /login - Login form with Supabase
   - /signup - Registration form

Use the existing UI components and theme from src/components/ui/
Follow the cyberpunk design system in src/styles/globals.css"
```

### 3. Add Real-time Data

Connect WebSocket in spectrum analyzer:

```typescript
useEffect(() => {
  wsClient.connect();

  wsClient.on('spectrum', (msg) => {
    // Update spectrum waterfall
  });

  return () => wsClient.disconnect();
}, []);
```

### 4. Add Charts

Install charting library:

```bash
npm install recharts
```

Create analytics visualizations with cyberpunk styling.

## 🎨 Component Usage

### Cyberpunk Button

```typescript
import { Button } from '@/components/ui/button';

<Button variant="neon">Launch Mission</Button>
<Button variant="danger">Alert</Button>
<Button variant="success">Confirm</Button>
```

### Glowing Card

```typescript
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

<Card className="hover:glow-cyan">
  <CardHeader>
    <CardTitle>Signal Detection</CardTitle>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

### Status Badge

```typescript
import { Badge } from '@/components/ui/badge';

<Badge variant="success">Online</Badge>
<Badge variant="danger">Critical</Badge>
<Badge variant="info">Detected</Badge>
```

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Supabase Docs](https://supabase.com/docs)
- [Lovable Guide](../LOVABLE_SETUP_GUIDE.md)

## 🎉 Status

**ZELDA Frontend**: Cyberpunk-themed dashboard with core UI complete, ready for feature development in Lovable!

**Built Components**: Landing, Dashboard, Spectrum Analyzer, Auth Hook, API Clients

**Next**: Build remaining pages, connect real-time data, deploy!
