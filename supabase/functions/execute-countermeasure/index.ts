import { serve } from 'https://deno.land/std@0.177.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.0';

// Resource constraints
const MAX_EXECUTION_TIME_MS = 120000; // 2 minutes (well under 150s limit)
const DB_TIMEOUT_MS = 5000; // 5 seconds for DB operations
const BACKEND_TIMEOUT_MS = 10000; // 10 seconds for backend calls

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface CountermeasureRequest {
  threat_id: string;
  threat_type: string;
  threat_severity: string;
  countermeasure_type: string;
  parameters: Record<string, any>;
  location?: { latitude: number; longitude: number };
}

interface CountermeasureResponse {
  success: boolean;
  message: string;
  actions_taken: string[];
  new_parameters: Record<string, any>;
  duration_ms: number;
}

// Timeout helper for database operations
async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMsg: string
): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(errorMsg)), timeoutMs)
  );
  return Promise.race([promise, timeout]);
}

// Non-blocking backend notification
async function notifyBackend(
  url: string,
  payload: any,
  timeoutMs: number
): Promise<void> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error) {
    console.error('Backend notification failed (non-critical):', error.message);
  } finally {
    clearTimeout(timeoutId);
  }
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const startTime = Date.now();

  try {
    // Enforce maximum execution time
    const executionTimeoutId = setTimeout(() => {
      console.error('Function approaching timeout limit');
    }, MAX_EXECUTION_TIME_MS - 10000); // Warning 10s before limit

    try {
      // Get request body with timeout
      const body: CountermeasureRequest = await withTimeout(
        req.json(),
        5000,
        'Request body parsing timeout'
      );

      const {
        threat_id,
        threat_type,
        threat_severity,
        countermeasure_type,
        parameters,
        location,
      } = body;

      // Validate required fields early
      if (!threat_id || !countermeasure_type) {
        throw new Error('Missing required fields: threat_id, countermeasure_type');
      }

      // Get auth header
      const authHeader = req.headers.get('Authorization');
      if (!authHeader) {
        throw new Error('Missing authorization header');
      }

      // Initialize Supabase client with optimized settings
      const supabaseClient = createClient(
        Deno.env.get('SUPABASE_URL') ?? '',
        Deno.env.get('SUPABASE_ANON_KEY') ?? '',
        {
          global: { headers: { Authorization: authHeader } },
          db: { schema: 'public' },
          auth: { persistSession: false }, // Don't persist in edge function
        }
      );

      // Get user with timeout
      const {
        data: { user },
        error: userError,
      } = await withTimeout(
        supabaseClient.auth.getUser(),
        DB_TIMEOUT_MS,
        'User authentication timeout'
      );

      if (userError || !user) {
        throw new Error('Unauthorized');
      }

      console.log(`Executing countermeasure: ${countermeasure_type} for threat ${threat_id}`);

      const actionsTaken: string[] = [];
      let newParameters: Record<string, any> = {};
      let success = false;
      let message = '';

    // Execute countermeasure based on type
    switch (countermeasure_type) {
      case 'frequency_hopping':
        actionsTaken.push('Activated frequency hopping');

        const hopPattern = parameters.hop_pattern || 'adaptive';
        const hopRate = parameters.hop_rate_ms || 500;
        const backupFreqs = parameters.backup_frequencies || [868e6, 915e6, 2450e6];

        // Select clear frequency (mock - replace with actual spectrum sensing)
        const newFreq = backupFreqs[Math.floor(Math.random() * backupFreqs.length)];

        actionsTaken.push(`Hopped to ${(newFreq / 1e6).toFixed(3)} MHz`);
        actionsTaken.push(`Started ${hopPattern} hopping (rate: ${hopRate}ms)`);

        newParameters = {
          new_frequency: newFreq,
          hop_pattern: hopPattern,
          hop_rate_ms: hopRate,
        };

        success = true;
        message = `Frequency hopping activated on ${(newFreq / 1e6).toFixed(3)} MHz`;
        break;

      case 'power_adjustment':
        actionsTaken.push('Adjusted transmission power');

        const powerReduction = parameters.power_reduction_db || 5;
        const currentPower = parameters.current_power || 0;
        const newPower = currentPower - powerReduction;

        actionsTaken.push(`Reduced power by ${powerReduction} dB`);
        actionsTaken.push(`New power: ${newPower.toFixed(1)} dBm`);

        newParameters = {
          old_power: currentPower,
          new_power: newPower,
          power_delta: -powerReduction,
        };

        success = true;
        message = `Power adjusted from ${currentPower.toFixed(1)} to ${newPower.toFixed(1)} dBm`;
        break;

      case 'jamming_mitigation':
        actionsTaken.push('Activated jamming mitigation');

        // Combine frequency hopping + power reduction
        const jammingHopFreq = parameters.backup_frequencies?.[0] || 2450e6;
        actionsTaken.push(`Hopped to ${(jammingHopFreq / 1e6).toFixed(3)} MHz`);
        actionsTaken.push('Reduced power by 10 dB');

        if (parameters.enable_spread_spectrum) {
          actionsTaken.push('Enabled spread spectrum modulation');
        }

        if (parameters.enable_fec) {
          actionsTaken.push('Enabled forward error correction');
        }

        newParameters = {
          new_frequency: jammingHopFreq,
          power_reduction: 10,
          spread_spectrum: parameters.enable_spread_spectrum || false,
          fec_enabled: parameters.enable_fec || false,
        };

        success = true;
        message = 'Jamming mitigation completed successfully';
        break;

      case 'spectrum_evasion':
        actionsTaken.push('Activated spectrum evasion');

        const evasionFreq = parameters.backup_frequencies?.[1] || 915e6;
        actionsTaken.push(`Evaded to ${(evasionFreq / 1e6).toFixed(3)} MHz`);
        actionsTaken.push('Reduced power by 5 dB for stealth');

        if (parameters.enable_stealth_mode) {
          actionsTaken.push('Enabled stealth mode');
        }

        newParameters = {
          new_frequency: evasionFreq,
          power_reduction: 5,
          stealth_mode: parameters.enable_stealth_mode || false,
        };

        success = true;
        message = 'Spectrum evasion completed';
        break;

      case 'alert_only':
        actionsTaken.push('Operator alerted, no automated action taken');
        success = true;
        message = 'Alert sent to operator';
        break;

      default:
        actionsTaken.push(`Unknown countermeasure type: ${countermeasure_type}`);
        success = false;
        message = `Countermeasure ${countermeasure_type} not implemented`;
    }

      const durationMs = Date.now() - startTime;

      // Prepare response
      const response: CountermeasureResponse = {
        success,
        message,
        actions_taken: actionsTaken,
        new_parameters: newParameters,
        duration_ms: durationMs,
      };

      // Log countermeasure action to database with timeout
      try {
        await withTimeout(
          supabaseClient
            .from('countermeasure_actions')
            .insert({
              user_id: user.id,
              threat_id,
              countermeasure_type,
              parameters,
              status: success ? 'completed' : 'failed',
              actions_taken: actionsTaken,
              result: message,
              duration_ms: durationMs,
            }),
          DB_TIMEOUT_MS,
          'Database logging timeout'
        );
      } catch (logError) {
        console.error('Failed to log countermeasure (non-critical):', logError.message);
        // Don't fail the request if logging fails
      }

      // Send webhook to Python backend (non-blocking)
      const pythonBackendUrl = Deno.env.get('PYTHON_BACKEND_URL');
      if (pythonBackendUrl) {
        // Fire and forget - don't wait for backend
        notifyBackend(
          `${pythonBackendUrl}/countermeasure`,
          {
            threat_id,
            threat_type,
            threat_severity,
            countermeasure_type,
            parameters,
            location,
            response,
          },
          BACKEND_TIMEOUT_MS
        ).catch(() => {
          // Already logged in notifyBackend
        });
      }

      // Clear execution timeout
      clearTimeout(executionTimeoutId);

      return new Response(JSON.stringify(response), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      });
    } finally {
      clearTimeout(executionTimeoutId);
    }
  } catch (error) {
    console.error('Countermeasure execution error:', error);

    const durationMs = Date.now() - startTime;

    return new Response(
      JSON.stringify({
        success: false,
        message: error.message || 'Internal server error',
        actions_taken: [],
        new_parameters: {},
        duration_ms: durationMs,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message?.includes('timeout') || error.message?.includes('Unauthorized') ? 408 : 400,
      }
    );
  }
});
