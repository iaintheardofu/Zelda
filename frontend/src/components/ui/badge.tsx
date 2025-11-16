import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-sm border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 font-orbitron uppercase tracking-wider',
  {
    variants: {
      variant: {
        default:
          'border-transparent bg-primary text-primary-foreground shadow glow-cyan',
        secondary:
          'border-transparent bg-secondary text-secondary-foreground glow-purple',
        destructive:
          'border-transparent bg-destructive text-destructive-foreground shadow glow-red',
        outline: 'text-foreground border-primary glow-cyan',
        success:
          'border-neon-green bg-neon-green/20 text-neon-green glow-green',
        warning:
          'border-neon-orange bg-neon-orange/20 text-neon-orange',
        danger:
          'border-neon-red bg-neon-red/20 text-neon-red glow-red',
        info:
          'border-neon-cyan bg-neon-cyan/20 text-neon-cyan glow-cyan',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
