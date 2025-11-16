import { serve } from 'https://deno.land/std@0.177.0/http/server.ts';

// Resource constraints for Supabase Edge Functions
const MAX_SAMPLES = 1024; // Limit sample size to avoid memory issues
const TIMEOUT_MS = 30000; // 30 second timeout

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface DetectionRequest {
  samples: number[];
  sample_rate: number;
  detector_type?: 'energy' | 'cyclo' | 'blind' | 'all';
}

interface DetectionResponse {
  detected: boolean;
  confidence: number;
  algorithm: string;
  snr_estimate?: number;
  signal_power?: number;
  noise_floor?: number;
  execution_time_ms: number;
}

// Simplified energy detection (memory efficient)
function energyDetection(samples: number[], sampleRate: number): DetectionResponse {
  const start = Date.now();

  // Limit samples to avoid memory issues
  const limitedSamples = samples.slice(0, MAX_SAMPLES);
  const N = limitedSamples.length;

  // Calculate signal power (mean square)
  let sumSquares = 0;
  for (let i = 0; i < N; i++) {
    sumSquares += limitedSamples[i] * limitedSamples[i];
  }
  const signalPower = sumSquares / N;

  // Estimate noise floor (lower 25th percentile)
  const sorted = [...limitedSamples].map(Math.abs).sort((a, b) => a - b);
  const noiseFloor = sorted[Math.floor(N * 0.25)] ** 2;

  // CFAR threshold
  const threshold = noiseFloor * 3.5; // 3.5x noise floor

  // Detection decision
  const detected = signalPower > threshold;

  // SNR estimate
  const snr = detected ? 10 * Math.log10(signalPower / noiseFloor) : -20;

  return {
    detected,
    confidence: detected ? Math.min(snr / 20, 1.0) : 0.0,
    algorithm: 'energy_cfar',
    snr_estimate: snr,
    signal_power: 10 * Math.log10(signalPower),
    noise_floor: 10 * Math.log10(noiseFloor),
    execution_time_ms: Date.now() - start,
  };
}

// Simplified cyclostationary detection (memory efficient)
function cyclostationaryDetection(samples: number[], sampleRate: number): DetectionResponse {
  const start = Date.now();

  // Use limited samples
  const limitedSamples = samples.slice(0, MAX_SAMPLES);
  const N = limitedSamples.length;

  // Simplified autocorrelation at lag = 1
  let autocorr = 0;
  for (let i = 0; i < N - 1; i++) {
    autocorr += limitedSamples[i] * limitedSamples[i + 1];
  }
  autocorr /= (N - 1);

  // Variance
  const mean = limitedSamples.reduce((a, b) => a + b, 0) / N;
  let variance = 0;
  for (let i = 0; i < N; i++) {
    variance += (limitedSamples[i] - mean) ** 2;
  }
  variance /= N;

  // Cyclic feature indicator
  const cyclicFeature = Math.abs(autocorr) / (variance + 1e-10);

  // Detection threshold
  const detected = cyclicFeature > 0.3;

  return {
    detected,
    confidence: Math.min(cyclicFeature, 1.0),
    algorithm: 'cyclostationary',
    execution_time_ms: Date.now() - start,
  };
}

// Simplified blind detection (eigenvalue-based, memory efficient)
function blindDetection(samples: number[], sampleRate: number): DetectionResponse {
  const start = Date.now();

  // Use limited samples
  const limitedSamples = samples.slice(0, MAX_SAMPLES);
  const N = limitedSamples.length;

  // Simple variance ratio test
  // Split into two halves
  const half = Math.floor(N / 2);
  const first = limitedSamples.slice(0, half);
  const second = limitedSamples.slice(half);

  // Calculate variances
  const mean1 = first.reduce((a, b) => a + b, 0) / first.length;
  const mean2 = second.reduce((a, b) => a + b, 0) / second.length;

  let var1 = 0, var2 = 0;
  for (let i = 0; i < first.length; i++) {
    var1 += (first[i] - mean1) ** 2;
  }
  for (let i = 0; i < second.length; i++) {
    var2 += (second[i] - mean2) ** 2;
  }
  var1 /= first.length;
  var2 /= second.length;

  // Variance ratio (signals have non-uniform power)
  const ratio = Math.max(var1, var2) / (Math.min(var1, var2) + 1e-10);

  // Detection threshold
  const detected = ratio > 2.0;

  return {
    detected,
    confidence: Math.min(ratio / 10, 1.0),
    algorithm: 'blind_variance',
    execution_time_ms: Date.now() - start,
  };
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const startTime = Date.now();

  try {
    // Parse request with timeout
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timeout')), TIMEOUT_MS)
    );

    const body: DetectionRequest = await Promise.race([
      req.json(),
      timeoutPromise,
    ]) as DetectionRequest;

    const { samples, sample_rate, detector_type = 'energy' } = body;

    // Validate input
    if (!samples || samples.length === 0) {
      throw new Error('No samples provided');
    }

    if (samples.length > MAX_SAMPLES * 2) {
      console.warn(`Truncating ${samples.length} samples to ${MAX_SAMPLES}`);
    }

    // Execute detection based on type
    let result: DetectionResponse;

    switch (detector_type) {
      case 'energy':
        result = energyDetection(samples, sample_rate);
        break;

      case 'cyclo':
        result = cyclostationaryDetection(samples, sample_rate);
        break;

      case 'blind':
        result = blindDetection(samples, sample_rate);
        break;

      case 'all':
        // Run all detectors and fuse results
        const energyResult = energyDetection(samples, sample_rate);
        const cycloResult = cyclostationaryDetection(samples, sample_rate);
        const blindResult = blindDetection(samples, sample_rate);

        const detections = [energyResult, cycloResult, blindResult].filter(r => r.detected);
        const avgConfidence = detections.length > 0
          ? detections.reduce((sum, r) => sum + r.confidence, 0) / detections.length
          : 0;

        result = {
          detected: detections.length >= 2, // Majority vote
          confidence: avgConfidence,
          algorithm: 'multi_algorithm_fusion',
          snr_estimate: energyResult.snr_estimate,
          signal_power: energyResult.signal_power,
          noise_floor: energyResult.noise_floor,
          execution_time_ms: Date.now() - startTime,
        };
        break;

      default:
        throw new Error(`Unknown detector type: ${detector_type}`);
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
    console.error('Detection error:', error);

    return new Response(
      JSON.stringify({
        detected: false,
        confidence: 0,
        algorithm: 'error',
        error: error.message || 'Detection failed',
        execution_time_ms: Date.now() - startTime,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      }
    );
  }
});
