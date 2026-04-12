import { Info, AlertTriangle, Lightbulb, AlertOctagon } from 'lucide-react'
import { ReactNode } from 'react'

type CalloutType = 'info' | 'warning' | 'tip' | 'critical'

interface CalloutBoxProps {
  type: CalloutType
  title?: string
  children: ReactNode
}

const config: Record<CalloutType, {
  icon: typeof Info
  bgClass: string
  borderClass: string
  textClass: string
  iconClass: string
  defaultTitle: string
}> = {
  info: {
    icon: Info,
    bgClass: 'bg-[var(--callout-info-bg)]',
    borderClass: 'border-[var(--callout-info-border)]',
    textClass: 'text-[var(--callout-info-text)]',
    iconClass: 'text-[var(--callout-info-border)]',
    defaultTitle: 'Note',
  },
  warning: {
    icon: AlertTriangle,
    bgClass: 'bg-[var(--callout-warning-bg)]',
    borderClass: 'border-[var(--callout-warning-border)]',
    textClass: 'text-[var(--callout-warning-text)]',
    iconClass: 'text-[var(--callout-warning-border)]',
    defaultTitle: 'Warning',
  },
  tip: {
    icon: Lightbulb,
    bgClass: 'bg-[var(--callout-tip-bg)]',
    borderClass: 'border-[var(--callout-tip-border)]',
    textClass: 'text-[var(--callout-tip-text)]',
    iconClass: 'text-[var(--callout-tip-border)]',
    defaultTitle: 'Tip',
  },
  critical: {
    icon: AlertOctagon,
    bgClass: 'bg-[var(--callout-critical-bg)]',
    borderClass: 'border-[var(--callout-critical-border)]',
    textClass: 'text-[var(--callout-critical-text)]',
    iconClass: 'text-[var(--callout-critical-border)]',
    defaultTitle: 'Critical',
  },
}

export function CalloutBox({ type, title, children }: CalloutBoxProps) {
  const c = config[type]
  const Icon = c.icon

  return (
    <div className={`my-6 rounded-xl border-l-4 ${c.borderClass} ${c.bgClass} p-5`}>
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 shrink-0 mt-0.5 ${c.iconClass}`} />
        <div className="flex-1 min-w-0">
          <p className={`font-semibold text-sm mb-1 ${c.textClass}`}>
            {title || c.defaultTitle}
          </p>
          <div className={`text-sm leading-relaxed ${c.textClass} opacity-90`}>
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}
