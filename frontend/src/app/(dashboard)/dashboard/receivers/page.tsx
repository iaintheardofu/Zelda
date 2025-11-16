'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Radio, MapPin, Plus, Trash2, Power, Edit, Activity } from 'lucide-react';
import Map, { Marker, Source, Layer } from 'react-map-gl';
import { createClient } from '@/lib/supabase/client';
import 'mapbox-gl/dist/mapbox-gl.css';

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN || '';

interface Receiver {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  status: 'online' | 'offline';
  last_seen: string;
  cpu_usage?: number;
  memory_usage?: number;
  user_id: string;
}

export default function ReceiversPage() {
  const [receivers, setReceivers] = useState<Receiver[]>([]);
  const [viewState, setViewState] = useState({
    latitude: 29.4897,
    longitude: -98.7443,
    zoom: 12
  });
  const [loading, setLoading] = useState(true);
  const supabase = createClient();

  // Fetch receivers from database
  const fetchReceivers = useCallback(async () => {
    try {
      const { data, error } = await supabase
        .from('receivers')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;

      if (data && data.length > 0) {
        setReceivers(data as Receiver[]);
        // Center map on first receiver
        setViewState(prev => ({
          ...prev,
          latitude: data[0].latitude,
          longitude: data[0].longitude,
        }));
      }
    } catch (error) {
      console.error('Error fetching receivers:', error);
    } finally {
      setLoading(false);
    }
  }, [supabase]);

  useEffect(() => {
    fetchReceivers();

    // Set up real-time subscription
    const channel = supabase
      .channel('receivers-changes')
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: 'receivers'
      }, () => {
        fetchReceivers();
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [fetchReceivers, supabase]);

  // Update receiver position on map drag
  const handleMarkerDragEnd = async (receiverId: string, event: any) => {
    const { lng, lat } = event.lngLat;

    try {
      const { error } = await supabase
        .from('receivers')
        .update({
          latitude: lat,
          longitude: lng
        })
        .eq('id', receiverId);

      if (error) throw error;

      await fetchReceivers();
    } catch (error) {
      console.error('Error updating receiver position:', error);
    }
  };

  const onlineCount = receivers.filter(r => r.status === 'online').length;
  const offlineCount = receivers.filter(r => r.status === 'offline').length;

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
          <Plus className="w-4 h-4 mr-2" />
          Add Receiver
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Online</p>
                <p className="text-2xl font-orbitron font-bold text-neon-green">{onlineCount}</p>
              </div>
              <Activity className="w-8 h-8 text-neon-green" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Offline</p>
                <p className="text-2xl font-orbitron font-bold text-muted-foreground">{offlineCount}</p>
              </div>
              <Radio className="w-8 h-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Map */}
      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="text-base font-orbitron">Coverage Map</CardTitle>
          <CardDescription>
            Drag markers to reposition • 🟢 Online | 🔴 Offline
          </CardDescription>
        </CardHeader>
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
                    draggable
                    onDragEnd={(e) => handleMarkerDragEnd(receiver.id, e)}
                  >
                    <div className="relative group cursor-move">
                      <Radio
                        className={`w-8 h-8 transition-all ${
                          receiver.status === 'online'
                            ? 'text-neon-cyan drop-shadow-[0_0_10px_rgba(0,255,255,0.8)]'
                            : 'text-red-500'
                        }`}
                      />
                      {receiver.status === 'online' && (
                        <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-neon-green animate-pulse" />
                      )}
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 hidden group-hover:block">
                        <div className="bg-card border border-neon-cyan/30 rounded-lg px-3 py-2 whitespace-nowrap">
                          <p className="text-xs font-orbitron text-neon-cyan">{receiver.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {receiver.latitude.toFixed(6)}, {receiver.longitude.toFixed(6)}
                          </p>
                        </div>
                      </div>
                    </div>
                  </Marker>
                ))}

                {/* Coverage circles */}
                {receivers.map(receiver => (
                  <Source
                    key={`coverage-${receiver.id}`}
                    type="geojson"
                    data={{
                      type: 'Feature',
                      geometry: {
                        type: 'Point',
                        coordinates: [receiver.longitude, receiver.latitude]
                      },
                      properties: {}
                    }}
                  >
                    <Layer
                      id={`coverage-layer-${receiver.id}`}
                      type="circle"
                      paint={{
                        'circle-radius': {
                          stops: [
                            [0, 0],
                            [20, receiver.status === 'online' ? 100 : 50]
                          ],
                          base: 2
                        },
                        'circle-color': receiver.status === 'online' ? '#00ffff' : '#ef4444',
                        'circle-opacity': 0.1,
                        'circle-stroke-width': 2,
                        'circle-stroke-color': receiver.status === 'online' ? '#00ffff' : '#ef4444',
                        'circle-stroke-opacity': 0.3
                      }}
                    />
                  </Source>
                ))}
              </Map>
            ) : (
              <div className="h-full flex items-center justify-center bg-muted/50">
                <div className="text-center space-y-2">
                  <MapPin className="w-12 h-12 mx-auto text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    Add MAPBOX_ACCESS_TOKEN to environment variables
                  </p>
                  <p className="text-xs text-muted-foreground/70">
                    Get your token at mapbox.com
                  </p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* TDOA Test Section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-orbitron">TDOA Signal Localization</CardTitle>
          <CardDescription>
            Test time-difference-of-arrival geolocation • Requires 3+ online receivers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            variant="neon"
            disabled={onlineCount < 3}
            className="w-full sm:w-auto"
          >
            <MapPin className="w-4 h-4 mr-2" />
            Test TDOA
          </Button>
          {onlineCount < 3 && (
            <p className="text-xs text-muted-foreground mt-2">
              Need at least 3 online receivers for TDOA localization
            </p>
          )}
          <div className="mt-4 grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-muted-foreground">• </span>
              <span className="text-neon-green">Green</span> = Online receivers
            </div>
            <div>
              <span className="text-muted-foreground">• </span>
              <span className="text-red-500">Red</span> = Offline receivers
            </div>
            <div className="col-span-2">
              <span className="text-muted-foreground">• </span>
              <span className="text-neon-pink">Pink crosshair</span> = Localized signal source
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Receiver List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <Card className="col-span-full">
            <CardContent className="p-12 text-center">
              <Radio className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50 animate-pulse" />
              <p className="text-sm text-muted-foreground">Loading receivers...</p>
            </CardContent>
          </Card>
        ) : receivers.length === 0 ? (
          <Card className="col-span-full">
            <CardContent className="p-12 text-center">
              <Radio className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
              <h3 className="font-orbitron text-lg mb-2">No Receivers Configured</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Add your first receiver to start monitoring RF signals
              </p>
              <Button variant="neon">
                <Plus className="w-4 h-4 mr-2" />
                Add Receiver
              </Button>
            </CardContent>
          </Card>
        ) : (
          receivers.map(receiver => (
            <Card key={receiver.id} className="group hover:border-neon-cyan/50 transition-all">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Radio className="w-4 h-4" />
                      {receiver.name}
                    </CardTitle>
                    <CardDescription className="text-xs mt-1">
                      {receiver.latitude.toFixed(6)}, {receiver.longitude.toFixed(6)}
                    </CardDescription>
                  </div>
                  <Badge variant={receiver.status === 'online' ? 'success' : 'destructive'}>
                    {receiver.status.toUpperCase()}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Last Seen</span>
                    <span className="font-mono">
                      {new Date(receiver.last_seen).toLocaleTimeString()}
                    </span>
                  </div>
                  {receiver.cpu_usage !== undefined && (
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">CPU Usage</span>
                      <span className="font-mono">{receiver.cpu_usage}%</span>
                    </div>
                  )}
                  {receiver.memory_usage !== undefined && (
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Memory Usage</span>
                      <span className="font-mono">{receiver.memory_usage}%</span>
                    </div>
                  )}
                </div>
                <div className="flex gap-2 pt-2 border-t border-border">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Edit className="w-3 h-3 mr-1" />
                    Edit
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                  >
                    <Power className="w-3 h-3 mr-1" />
                    {receiver.status === 'online' ? 'Stop' : 'Start'}
                  </Button>
                  <Button variant="danger" size="sm">
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
