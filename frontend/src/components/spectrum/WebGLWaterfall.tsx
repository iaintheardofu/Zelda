'use client';

import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';

interface WebGLWaterfallProps {
  width?: number;
  height?: number;
  fftSize?: number;
  historySize?: number;
  className?: string;
}

export interface WebGLWaterfallRef {
  addFFTData: (fftData: number[]) => void;
  clear: () => void;
}

export const WebGLWaterfall = forwardRef<WebGLWaterfallRef, WebGLWaterfallProps>(
  ({ width = 1024, height = 512, fftSize = 1024, historySize = 200, className = '' }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGLRenderingContext | null>(null);
    const programRef = useRef<WebGLProgram | null>(null);
    const textureRef = useRef<WebGLTexture | null>(null);
    const waterfallDataRef = useRef<Uint8Array>(new Uint8Array(fftSize * historySize * 4));
    const currentLineRef = useRef(0);

    // Initialize WebGL
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const gl = canvas.getContext('webgl', {
        alpha: false,
        antialias: false,
        preserveDrawingBuffer: true,
      });

      if (!gl) {
        console.error('WebGL not supported');
        return;
      }

      glRef.current = gl;

      // Vertex shader
      const vertexShaderSource = `
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 vTexCoord;
        void main() {
          gl_Position = vec4(position, 0.0, 1.0);
          vTexCoord = texCoord;
        }
      `;

      // Fragment shader with cyberpunk color palette
      const fragmentShaderSource = `
        precision mediump float;
        varying vec2 vTexCoord;
        uniform sampler2D uSampler;

        vec3 heatmapColor(float value) {
          // Cyberpunk color palette: dark blue -> cyan -> magenta -> yellow
          vec3 color;

          if (value < 0.25) {
            // Dark blue to blue
            float t = value * 4.0;
            color = mix(vec3(0.0, 0.0, 0.1), vec3(0.0, 0.0, 1.0), t);
          } else if (value < 0.5) {
            // Blue to cyan
            float t = (value - 0.25) * 4.0;
            color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);
          } else if (value < 0.75) {
            // Cyan to magenta
            float t = (value - 0.5) * 4.0;
            color = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0), t);
          } else {
            // Magenta to yellow/white
            float t = (value - 0.75) * 4.0;
            color = mix(vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 0.5), t);
          }

          return color;
        }

        void main() {
          float value = texture2D(uSampler, vTexCoord).r;
          gl_FragColor = vec4(heatmapColor(value), 1.0);
        }
      `;

      // Compile shaders
      const vertexShader = gl.createShader(gl.VERTEX_SHADER)!;
      gl.shaderSource(vertexShader, vertexShaderSource);
      gl.compileShader(vertexShader);

      if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
        console.error('Vertex shader error:', gl.getShaderInfoLog(vertexShader));
        return;
      }

      const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER)!;
      gl.shaderSource(fragmentShader, fragmentShaderSource);
      gl.compileShader(fragmentShader);

      if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
        console.error('Fragment shader error:', gl.getShaderInfoLog(fragmentShader));
        return;
      }

      // Link program
      const program = gl.createProgram()!;
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return;
      }

      programRef.current = program;
      gl.useProgram(program);

      // Create quad vertices
      const positions = new Float32Array([
        -1, -1,  // bottom left
         1, -1,  // bottom right
        -1,  1,  // top left
         1,  1,  // top right
      ]);

      const texCoords = new Float32Array([
        0, 1,  // bottom left
        1, 1,  // bottom right
        0, 0,  // top left
        1, 0,  // top right
      ]);

      // Position buffer
      const positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

      const positionLocation = gl.getAttribLocation(program, 'position');
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

      // Texture coord buffer
      const texCoordBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

      const texCoordLocation = gl.getAttribLocation(program, 'texCoord');
      gl.enableVertexAttribArray(texCoordLocation);
      gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);

      // Create texture
      const texture = gl.createTexture();
      textureRef.current = texture;
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      // Set initial texture data
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        fftSize,
        historySize,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        waterfallDataRef.current
      );

      // Set viewport and clear
      gl.viewport(0, 0, width, height);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Initial render
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      return () => {
        if (gl) {
          gl.deleteTexture(texture);
          gl.deleteBuffer(positionBuffer);
          gl.deleteBuffer(texCoordBuffer);
          gl.deleteProgram(program);
          gl.deleteShader(vertexShader);
          gl.deleteShader(fragmentShader);
        }
      };
    }, [width, height, fftSize, historySize]);

    // Add FFT data function
    const addFFTData = (fftData: number[]) => {
      const gl = glRef.current;
      const texture = textureRef.current;
      if (!gl || !texture || fftData.length !== fftSize) return;

      const data = waterfallDataRef.current;
      const currentLine = currentLineRef.current;

      // Scroll waterfall up by one line (copy data)
      for (let y = 0; y < historySize - 1; y++) {
        for (let x = 0; x < fftSize; x++) {
          const srcIdx = ((y + 1) * fftSize + x) * 4;
          const dstIdx = (y * fftSize + x) * 4;
          data[dstIdx] = data[srcIdx];
          data[dstIdx + 1] = data[srcIdx + 1];
          data[dstIdx + 2] = data[srcIdx + 2];
          data[dstIdx + 3] = data[srcIdx + 3];
        }
      }

      // Add new line at bottom
      for (let x = 0; x < fftSize && x < fftData.length; x++) {
        // Normalize power: -100 dBm to -20 dBm range
        const normalizedPower = Math.max(0, Math.min(1, (fftData[x] + 100) / 80));
        const intensity = Math.floor(normalizedPower * 255);

        const idx = ((historySize - 1) * fftSize + x) * 4;
        data[idx] = intensity;     // R channel (used by shader)
        data[idx + 1] = intensity; // G channel
        data[idx + 2] = intensity; // B channel
        data[idx + 3] = 255;       // Alpha
      }

      // Update texture
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        fftSize,
        historySize,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        data
      );

      // Render
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      currentLineRef.current = (currentLine + 1) % historySize;
    };

    // Clear waterfall
    const clear = () => {
      const gl = glRef.current;
      const texture = textureRef.current;
      if (!gl || !texture) return;

      // Clear data
      waterfallDataRef.current.fill(0);
      currentLineRef.current = 0;

      // Update texture
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        fftSize,
        historySize,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        waterfallDataRef.current
      );

      // Render
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      addFFTData,
      clear,
    }));

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={className}
      />
    );
  }
);

WebGLWaterfall.displayName = 'WebGLWaterfall';
