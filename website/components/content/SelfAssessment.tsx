'use client'

import { useState } from 'react'
import { ChevronDown, ChevronUp, GraduationCap, Eye } from 'lucide-react'

interface Question {
  question: string
  answer: string
  hints?: string[]
}

interface SelfAssessmentProps {
  questions: Question[]
}

export function SelfAssessment({ questions }: SelfAssessmentProps) {
  return (
    <div className="my-10">
      <div className="flex items-center gap-2 mb-6">
        <GraduationCap className="w-5 h-5 text-accent-orange" />
        <h2 className="text-xl font-bold text-text-primary">Self-Assessment Questions</h2>
      </div>
      <div className="space-y-3">
        {questions.map((q, i) => (
          <QuestionCard key={i} index={i + 1} question={q} />
        ))}
      </div>
    </div>
  )
}

function QuestionCard({ index, question }: { index: number; question: Question }) {
  const [revealed, setRevealed] = useState(false)
  const [hintsShown, setHintsShown] = useState(0)

  return (
    <div className="rounded-xl border border-border-primary bg-bg-surface overflow-hidden">
      {/* Question */}
      <div className="p-5">
        <div className="flex items-start gap-3">
          <span className="w-7 h-7 rounded-lg bg-accent-orange-subtle flex items-center justify-center
            text-xs font-bold text-accent-orange shrink-0">
            {index}
          </span>
          <p className="text-sm font-medium text-text-primary leading-relaxed pt-0.5">
            {question.question}
          </p>
        </div>
      </div>

      {/* Hints */}
      {question.hints && question.hints.length > 0 && !revealed && (
        <div className="px-5 pb-3">
          {question.hints.slice(0, hintsShown).map((hint, i) => (
            <div key={i} className="ml-10 mb-2 text-xs text-text-tertiary bg-bg-surface-hover rounded-lg px-3 py-2">
              <span className="font-medium text-accent-amber">Hint {i + 1}:</span> {hint}
            </div>
          ))}
          {hintsShown < question.hints.length && (
            <button
              onClick={() => setHintsShown(hintsShown + 1)}
              className="ml-10 text-xs text-accent-amber hover:text-accent-orange transition-colors"
            >
              Show hint {hintsShown + 1} of {question.hints.length}
            </button>
          )}
        </div>
      )}

      {/* Answer */}
      {revealed ? (
        <div className="border-t border-border-primary bg-accent-green-subtle/30 p-5">
          <div className="flex items-start gap-3">
            <div className="w-7 shrink-0" />
            <div className="text-sm text-text-primary leading-relaxed">
              {question.answer}
            </div>
          </div>
          <button
            onClick={() => setRevealed(false)}
            className="ml-10 mt-3 flex items-center gap-1 text-xs text-text-tertiary hover:text-text-secondary transition-colors"
          >
            <ChevronUp className="w-3.5 h-3.5" />
            Hide answer
          </button>
        </div>
      ) : (
        <button
          onClick={() => setRevealed(true)}
          className="w-full flex items-center justify-center gap-2 py-3 text-xs font-medium
            text-accent-blue hover:text-accent-blue-hover border-t border-border-primary
            bg-bg-surface/50 hover:bg-accent-blue-subtle/30 transition-colors"
        >
          <Eye className="w-3.5 h-3.5" />
          Reveal Answer
        </button>
      )}
    </div>
  )
}
