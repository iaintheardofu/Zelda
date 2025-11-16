export function ZeldaLogo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Magenta Lightning Bolt with Glow */}
      <g filter="url(#glow)">
        <path
          d="M13 2L3 14h8l-1 8 10-12h-8l1-8z"
          fill="url(#magentaGradient)"
          stroke="#FF1166"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>

      {/* Gradient Definition */}
      <defs>
        <linearGradient id="magentaGradient" x1="3" y1="2" x2="13" y2="22" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#FF1166" />
          <stop offset="50%" stopColor="#AA00FF" />
          <stop offset="100%" stopColor="#FF1166" />
        </linearGradient>

        {/* Glow Filter */}
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}

export function ZeldaLogoAnimated({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Animated Magenta Lightning Bolt */}
      <g filter="url(#glowAnimated)">
        <path
          d="M13 2L3 14h8l-1 8 10-12h-8l1-8z"
          fill="url(#magentaGradientAnimated)"
          stroke="#FF1166"
          strokeWidth="0.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <animate
            attributeName="opacity"
            values="1;0.6;1"
            dur="2s"
            repeatCount="indefinite"
          />
        </path>
      </g>

      <defs>
        <linearGradient id="magentaGradientAnimated" x1="3" y1="2" x2="13" y2="22" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#FF1166">
            <animate attributeName="stop-color" values="#FF1166;#AA00FF;#FF1166" dur="3s" repeatCount="indefinite" />
          </stop>
          <stop offset="50%" stopColor="#AA00FF" />
          <stop offset="100%" stopColor="#FF1166">
            <animate attributeName="stop-color" values="#FF1166;#AA00FF;#FF1166" dur="3s" repeatCount="indefinite" />
          </stop>
        </linearGradient>

        <filter id="glowAnimated" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
    </svg>
  );
}
