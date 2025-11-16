# ZELDA Lovable Copy-Paste Instructions

## 🚀 Quick Setup

Lovable has bidirectional sync with GitHub enabled. These changes have been pushed to GitHub and **Lovable should pull them automatically within 1-2 minutes**.

**What's Already Synced to GitHub:**
- ✅ Magenta lightning bolt logo (animated)
- ✅ Settings page (complete with all controls)
- ✅ Updated dashboard layout with new logo

**What You Need to Manually Update in Lovable:**
Since these pages might have been created before the GitHub connection, you'll need to update them in Lovable:

1. Receivers page (add Mapbox integration)
2. Spectrum analyzer (fix frequency ranges)
3. Dashboard page (clear demo data)

---

## 📋 Step-by-Step Instructions

### Step 1: Wait for GitHub Sync (1-2 minutes)

Lovable will automatically pull:
- `frontend/src/components/ZeldaLogo.tsx`
- `frontend/src/app/(dashboard)/dashboard/settings/page.tsx`
- `frontend/src/app/(dashboard)/layout.tsx` (updated with logo)

**To verify sync:**
1. In Lovable, check the file tree
2. Look for `src/components/ZeldaLogo.tsx`
3. If it appears, sync is complete!
4. Refresh preview to see the magenta lightning logo

---

### Step 2: Update Receivers Page with Mapbox

**File:** `src/app/(dashboard)/dashboard/receivers/page.tsx`

**In Lovable:**
1. Open the file in Lovable editor
2. **Replace entire contents** with the code below
3. Save (Ctrl/Cmd + S)

<details>
<summary>Click to see complete Receivers page code</summary>

\`\`\`typescript
'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Radio, MapPin, Plus, Trash2, Power } from 'lucide-react';
import Map, { Marker } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_PUBLIC_TOKEN || '';

interface Receiver {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  status: 'online' | 'offline';
  lastSeen: string;
}

export default function ReceiversPage() {
  const [receivers, setReceivers] = useState<Receiver[]>([]);
  const [viewState, setViewState] = useState({
    latitude: 37.7749,
    longitude: -122.4194,
    zoom: 10
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-cyan uppercase tracking-wider">
            RECEIVERS
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage distributed sensor network
          </p>
        </div>
        <Button variant="neon" size="sm">
          <Plus className="w-4 h-4" />
          Add Receiver
        </Button>
      </div>

      {/* Map */}
      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div className="h-[500px] relative">
            {MAPBOX_TOKEN ? (
              <Map
                {...viewState}
                onMove={evt => setViewState(evt.viewState)}
                mapStyle="mapbox://styles/mapbox/dark-v11"
                mapboxAccessToken={MAPBOX_TOKEN}
              >
                {receivers.map(receiver => (
                  <Marker
                    key={receiver.id}
                    latitude={receiver.latitude}
                    longitude={receiver.longitude}
                    anchor="bottom"
                  >
                    <div className="relative">
                      <Radio
                        className={`w-8 h-8 ${
                          receiver.status === 'online'
                            ? 'text-neon-cyan drop-shadow-[0_0_10px_rgba(0,255,255,0.8)]'
                            : 'text-muted-foreground'
                        }`}
                      />
                      {receiver.status === 'online' && (
                        <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-neon-green animate-pulse" />
                      )}
                    </div>
                  </Marker>
                ))}
              </Map>
            ) : (
              <div className="h-full flex items-center justify-center bg-muted/50">
                <div className="text-center space-y-2">
                  <MapPin className="w-12 h-12 mx-auto text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Add MAPBOX_PUBLIC_TOKEN to Lovable Secrets
                  </p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Receiver List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {receivers.length === 0 ? (
          <Card className="col-span-full">
            <CardContent className="p-12 text-center">
              <Radio className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
              <h3 className="font-orbitron text-lg mb-2">No Receivers Configured</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Add your first receiver to start monitoring RF signals
              </p>
              <Button variant="neon">
                <Plus className="w-4 h-4" />
                Add Receiver
              </Button>
            </CardContent>
          </Card>
        ) : (
          receivers.map(receiver => (
            <Card key={receiver.id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base">{receiver.name}</CardTitle>
                    <CardDescription className="text-xs">
                      {receiver.latitude.toFixed(4)}, {receiver.longitude.toFixed(4)}
                    </CardDescription>
                  </div>
                  <Badge variant={receiver.status === 'online' ? 'success' : 'destructive'}>
                    {receiver.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Last Seen</span>
                  <span>{receiver.lastSeen}</span>
                </div>
                <div className="flex gap-2 pt-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Power className="w-3 h-3" />
                  </Button>
                  <Button variant="danger" size="sm" className="flex-1">
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
\`\`\`

</details>

---

### Step 3: Update Spectrum Analyzer

**File:** `src/app/(dashboard)/dashboard/spectrum/page.tsx`

**Changes needed:**
1. Set default frequency to 915 MHz (ISM band)
2. Add frequency presets for 900-930 MHz and 2.4-2.5 GHz
3. Connect to WebSocket for live data

**In Lovable:**
1. Open `src/app/(dashboard)/dashboard/spectrum/page.tsx`
2. Find the frequency controls section
3. Update the default values

**Key changes:**
\`\`\`typescript
// Change from:
const [centerFreq, setCenterFreq] = useState(2.45); // GHz

// To:
const [centerFreq, setCenterFreq] = useState(0.915); // GHz (915 MHz)
const [span, setSpan] = useState(0.030); // 30 MHz span

// Add frequency presets:
const FREQUENCY_PRESETS = [
  { name: 'ISM 915', center: 0.915, span: 0.030 },
  { name: 'WiFi 2.4', center: 2.45, span: 0.100 },
  { name: 'GPS L1', center: 1.575, span: 0.010 }
];
\`\`\`

---

### Step 4: Clear Demo Data from Dashboard

**File:** `src/app/(dashboard)/dashboard/page.tsx`

**Find and remove:**
1. Hardcoded mission data (`OPERATION NIGHTWATCH`, `URBAN SWEEP`)
2. Hardcoded threat alerts
3. Replace with empty state messages

**In Lovable:**
1. Open the dashboard page
2. Find the missions and threats sections
3. Replace with empty state (or connect to real Supabase data)

**Example empty state:**
\`\`\`typescript
{missions.length === 0 ? (
  <div className="text-center py-8 text-muted-foreground">
    <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
    <p className="text-sm">No active missions</p>
    <Button variant="outline" size="sm" className="mt-3">
      <Plus className="w-3 h-3" />
      Create Mission
    </Button>
  </div>
) : (
  // existing mission list
)}
\`\`\`

---

## 🎯 Verification Checklist

After making changes in Lovable:

- [ ] **Logo**: Refresh preview, see magenta lightning bolt in sidebar
- [ ] **Settings**: Navigate to `/dashboard/settings`, see all controls
- [ ] **Receivers**: Page loads without 404, shows map (if token added)
- [ ] **Spectrum**: Default frequency is 915 MHz (0.915 GHz)
- [ ] **Dashboard**: No demo data, empty states for missions/threats

---

## 🔧 If Mapbox Map Doesn't Show

**Reason:** Missing Mapbox token

**Fix:**
1. Go to https://account.mapbox.com/access-tokens/
2. Copy your public token (starts with `pk.`)
3. In Lovable: Cloud tab → Secrets
4. Add: `NEXT_PUBLIC_MAPBOX_PUBLIC_TOKEN` = your token
5. Redeploy preview

---

## 📦 Package Dependencies

If you get errors about missing packages, add to `package.json`:

\`\`\`json
{
  "dependencies": {
    "react-map-gl": "^7.1.7",
    "mapbox-gl": "^3.1.2"
  }
}
\`\`\`

Then run: `npm install` (Lovable does this automatically)

---

## 🚀 Final Result

After all changes:

1. **Sidebar**: Magenta animated lightning logo ⚡
2. **Settings page**: Working with all controls
3. **Receivers page**: Interactive Mapbox map
4. **Spectrum page**: Scans 900-930 MHz and 2.4 GHz
5. **Dashboard**: Clean, no demo data
6. **All pages**: Cyberpunk theme maintained

---

## 💡 Next Steps

Once everything is working in Lovable:

1. **Connect Python backend WebSocket**
   - Update `NEXT_PUBLIC_API_URL` in secrets
   - Real-time spectrum data
   - Live signal detections

2. **Add real Supabase data**
   - Missions from database
   - Receivers from database
   - Threats from database

3. **Test end-to-end**
   - Create mission in UI → saves to Supabase
   - Python backend sends detection → appears in UI
   - Threat detected → notification + sound alert

---

**Questions?** Reply with the page name or error message and I'll help debug! 🛠️
