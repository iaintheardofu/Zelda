import { serve } from 'https://deno.land/std@0.177.0/http/server.ts';

// Resource constraints
const MAX_SAMPLES = 2048; // Limit to prevent memory issues
const TIMEOUT_MS = 30000; // 30 second timeout

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface ClassificationRequest {
  samples: number[];
  sample_rate: number;
  center_freq?: number;
}

interface ClassificationResponse {
  modulation: string;
  confidence: number;
  alternatives: Array<{ type: string; probability: number }>;
  features: {
    amp_mean?: number;
    amp_std?: number;
    amp_kurt?: number;
    phase_std?: number;
    zero_crossing_rate?: number;
  };
  execution_time_ms: number;
}

// Feature extraction (simplified, memory-efficient)
function extractFeatures(samples: number[]): any {
  const N = Math.min(samples.length, MAX_SAMPLES);
  const limited = samples.slice(0, N);

  // Amplitude statistics
  const amplitudes = limited.map(Math.abs);
  const ampMean = amplitudes.reduce((a, b) => a + b, 0) / N;

  let ampVariance = 0;
  let ampFourthMoment = 0;
  for (let i = 0; i < N; i++) {
    const diff = amplitudes[i] - ampMean;
    ampVariance += diff * diff;
    ampFourthMoment += diff ** 4;
  }
  ampVariance /= N;
  const ampStd = Math.sqrt(ampVariance);
  const ampKurt = (ampFourthMoment / N) / (ampVariance ** 2) - 3;

  // Phase variation (using sign changes as proxy)
  let phaseChanges = 0;
  for (let i = 1; i < N; i++) {
    if (Math.sign(limited[i]) !== Math.sign(limited[i - 1])) {
      phaseChanges++;
    }
  }
  const phaseStd = phaseChanges / N;

  // Zero crossing rate
  let zeroCrossings = 0;
  for (let i = 1; i < N; i++) {
    if ((limited[i] >= 0 && limited[i - 1] < 0) || (limited[i] < 0 && limited[i - 1] >= 0)) {
      zeroCrossings++;
    }
  }
  const zcrRate = zeroCrossings / N;

  return {
    amp_mean: ampMean,
    amp_std: ampStd,
    amp_kurt: ampKurt,
    phase_std: phaseStd,
    zero_crossing_rate: zcrRate,
  };
}

// Rule-based classification (no ML required, memory efficient)
function classifyModulation(features: any): ClassificationResponse {
  const start = Date.now();

  const { amp_std, amp_kurt, phase_std, zero_crossing_rate } = features;

  // Normalized amplitude variation
  const ampVariation = amp_std / (features.amp_mean + 1e-10);

  // Classification rules based on features
  let modulation = 'UNKNOWN';
  let confidence = 0;
  const alternatives: Array<{ type: string; probability: number }> = [];

  // AM - High amplitude variation, low phase variation
  if (ampVariation > 0.3 && phase_std < 0.3) {
    modulation = 'AM';
    confidence = 0.85;
    alternatives.push({ type: 'DSB', probability: 0.1 });
    alternatives.push({ type: 'SSB', probability: 0.05 });
  }
  // FM - Low amplitude variation, high phase variation
  else if (ampVariation < 0.2 && phase_std > 0.4) {
    modulation = 'FM';
    confidence = 0.88;
    alternatives.push({ type: 'PM', probability: 0.08 });
    alternatives.push({ type: 'FSK', probability: 0.04 });
  }
  // BPSK - Low amplitude variation, moderate phase variation
  else if (ampVariation < 0.15 && phase_std > 0.25 && phase_std < 0.45) {
    modulation = 'BPSK';
    confidence = 0.82;
    alternatives.push({ type: 'QPSK', probability: 0.12 });
    alternatives.push({ type: '8PSK', probability: 0.06 });
  }
  // QPSK - Low amplitude variation, moderate-high phase variation
  else if (ampVariation < 0.18 && phase_std > 0.35 && phase_std < 0.55) {
    modulation = 'QPSK';
    confidence = 0.87;
    alternatives.push({ type: '8PSK', probability: 0.08 });
    alternatives.push({ type: 'OQPSK', probability: 0.05 });
  }
  // QAM - Moderate amplitude and phase variation
  else if (ampVariation > 0.2 && ampVariation < 0.35 && phase_std > 0.3) {
    if (amp_kurt > 0) {
      modulation = '16QAM';
      confidence = 0.75;
      alternatives.push({ type: '64QAM', probability: 0.15 });
    } else {
      modulation = '64QAM';
      confidence = 0.72;
      alternatives.push({ type: '16QAM', probability: 0.18 });
    }
    alternatives.push({ type: 'OFDM', probability: 0.10 });
  }
  // FSK - High zero crossing rate
  else if (zero_crossing_rate > 0.3) {
    modulation = 'FSK';
    confidence = 0.78;
    alternatives.push({ type: 'MSK', probability: 0.12 });
    alternatives.push({ type: 'GFSK', probability: 0.10 });
  }
  // 8PSK - High phase variation, low amplitude variation
  else if (ampVariation < 0.15 && phase_std > 0.5) {
    modulation = '8PSK';
    confidence = 0.80;
    alternatives.push({ type: 'QPSK', probability: 0.12 });
    alternatives.push({ type: '16PSK', probability: 0.08 });
  }
  // OFDM - High kurtosis, complex pattern
  else if (amp_kurt > 1.5 || amp_kurt < -1.5) {
    modulation = 'OFDM';
    confidence = 0.70;
    alternatives.push({ type: '64QAM', probability: 0.18 });
    alternatives.push({ type: 'DSSS', probability: 0.12 });
  }
  // Default to QPSK (most common)
  else {
    modulation = 'QPSK';
    confidence = 0.60;
    alternatives.push({ type: 'BPSK', probability: 0.20 });
    alternatives.push({ type: '8PSK', probability: 0.15 });
    alternatives.push({ type: 'FM', probability: 0.05 });
  }

  return {
    modulation,
    confidence,
    alternatives: alternatives.slice(0, 3),
    features,
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

    const body: ClassificationRequest = await Promise.race([
      req.json(),
      timeoutPromise,
    ]) as ClassificationRequest;

    const { samples, sample_rate, center_freq } = body;

    // Validate input
    if (!samples || samples.length === 0) {
      throw new Error('No samples provided');
    }

    if (!sample_rate || sample_rate <= 0) {
      throw new Error('Invalid sample rate');
    }

    if (samples.length > MAX_SAMPLES * 2) {
      console.warn(`Truncating ${samples.length} samples to ${MAX_SAMPLES}`);
    }

    // Extract features
    const features = extractFeatures(samples);

    // Classify modulation
    const result = classifyModulation(features);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
    console.error('Classification error:', error);

    return new Response(
      JSON.stringify({
        modulation: 'UNKNOWN',
        confidence: 0,
        alternatives: [],
        features: {},
        error: error.message || 'Classification failed',
        execution_time_ms: Date.now() - startTime,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400,
      }
    );
  }
});
