'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Settings as SettingsIcon,
  Bell,
  Radio,
  Shield,
  User,
  Save,
  RotateCcw
} from 'lucide-react';

export default function SettingsPage() {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [autoDetect, setAutoDetect] = useState(true);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-cyan uppercase tracking-wider">
            SETTINGS
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure system parameters and preferences
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <RotateCcw className="w-4 h-4" />
            Reset
          </Button>
          <Button variant="neon" size="sm">
            <Save className="w-4 h-4" />
            Save Changes
          </Button>
        </div>
      </div>

      {/* Notification Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-neon-cyan" />
            <CardTitle>Notifications</CardTitle>
          </div>
          <CardDescription>Configure alert and notification preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div>
              <h3 className="font-orbitron text-sm">Enable Notifications</h3>
              <p className="text-xs text-muted-foreground">
                Show toast alerts for new threats and events
              </p>
            </div>
            <button
              onClick={() => setNotificationsEnabled(!notificationsEnabled)}
              className={`relative w-12 h-6 rounded-full transition-all ${
                notificationsEnabled ? 'bg-neon-cyan glow-cyan' : 'bg-muted'
              }`}
            >
              <div
                className={`absolute top-1 w-4 h-4 rounded-full bg-background transition-all ${
                  notificationsEnabled ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div>
              <h3 className="font-orbitron text-sm">Sound Alerts</h3>
              <p className="text-xs text-muted-foreground">
                Play audio alerts for critical threats
              </p>
            </div>
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`relative w-12 h-6 rounded-full transition-all ${
                soundEnabled ? 'bg-neon-green glow-green' : 'bg-muted'
              }`}
            >
              <div
                className={`absolute top-1 w-4 h-4 rounded-full bg-background transition-all ${
                  soundEnabled ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Detection Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-neon-purple" />
            <CardTitle>Signal Detection</CardTitle>
          </div>
          <CardDescription>Configure RF detection parameters</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div>
              <h3 className="font-orbitron text-sm">Auto-Detect Signals</h3>
              <p className="text-xs text-muted-foreground">
                Automatically classify detected RF signals
              </p>
            </div>
            <button
              onClick={() => setAutoDetect(!autoDetect)}
              className={`relative w-12 h-6 rounded-full transition-all ${
                autoDetect ? 'bg-neon-purple glow-purple' : 'bg-muted'
              }`}
            >
              <div
                className={`absolute top-1 w-4 h-4 rounded-full bg-background transition-all ${
                  autoDetect ? 'left-7' : 'left-1'
                }`}
              />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-orbitron">Default Center Frequency</label>
              <select className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm">
                <option>915 MHz (ISM)</option>
                <option>2.45 GHz (WiFi)</option>
                <option>1.575 GHz (GPS L1)</option>
                <option>Custom</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-orbitron">Default Span</label>
              <select className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm">
                <option>30 MHz</option>
                <option>100 MHz</option>
                <option>500 MHz</option>
                <option>1 GHz</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-orbitron">Update Rate</label>
              <select className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm">
                <option>2 Hz</option>
                <option>5 Hz</option>
                <option>10 Hz</option>
                <option>20 Hz</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-orbitron">ML Confidence Threshold</label>
              <input
                type="number"
                defaultValue="85"
                min="0"
                max="100"
                className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Security Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-neon-pink" />
            <CardTitle>Security</CardTitle>
          </div>
          <CardDescription>Authentication and access control</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div>
              <h3 className="font-orbitron text-sm">Role</h3>
              <p className="text-xs text-muted-foreground">Current user role</p>
            </div>
            <Badge variant="success">Operator</Badge>
          </div>

          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
            <div>
              <h3 className="font-orbitron text-sm">Session Timeout</h3>
              <p className="text-xs text-muted-foreground">Auto-logout after inactivity</p>
            </div>
            <select className="p-2 rounded-md bg-background border border-primary/20 text-sm">
              <option>15 minutes</option>
              <option>30 minutes</option>
              <option>1 hour</option>
              <option>Never</option>
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Profile Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <User className="w-5 h-5 text-neon-green" />
            <CardTitle>Profile</CardTitle>
          </div>
          <CardDescription>User account settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-orbitron">Username</label>
              <input
                type="text"
                defaultValue="michael.pendleton.20"
                className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-orbitron">Email</label>
              <input
                type="email"
                placeholder="user@example.com"
                className="w-full p-2 rounded-md bg-muted border border-primary/20 text-sm"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-orbitron">Change Password</label>
            <div className="flex gap-2">
              <input
                type="password"
                placeholder="New password"
                className="flex-1 p-2 rounded-md bg-muted border border-primary/20 text-sm"
              />
              <Button variant="outline" size="sm">Update</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Info */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SettingsIcon className="w-5 h-5 text-neon-orange" />
            <CardTitle>System Information</CardTitle>
          </div>
          <CardDescription>Platform version and status</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Version</span>
            <span className="font-orbitron">v2.0.0</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Backend API</span>
            <Badge variant="success">Connected</Badge>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Database</span>
            <Badge variant="success">Healthy</Badge>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">WebSocket</span>
            <Badge variant="success">Live</Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
