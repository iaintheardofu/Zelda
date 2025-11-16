import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 font-orbitron uppercase tracking-wider',
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground shadow hover:bg-primary/90 glow-cyan',
        destructive:
          'bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90 glow-red',
        outline:
          'border border-primary text-primary bg-background shadow-sm hover:bg-primary/10 hover:glow-cyan',
        secondary:
          'bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80 glow-purple',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
        neon: 'bg-transparent border-2 border-neon-cyan text-neon-cyan hover:bg-neon-cyan/10 glow-cyan',
        danger: 'bg-neon-red/20 border border-neon-red text-neon-red hover:bg-neon-red/30 glow-red',
        success: 'bg-neon-green/20 border border-neon-green text-neon-green hover:bg-neon-green/30 glow-green',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-8',
        xl: 'h-12 rounded-md px-10 text-base',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
