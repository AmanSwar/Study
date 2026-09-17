# Learner profile — who this material is for

Write for exactly one reader. Everything is pitched at him; a request can override any line below inline.

## Who
- **Aman** — ML-systems / inference engineer at RunAnywhere (on-device, non-GPU, edge inference: Apple Silicon/Metal, Qualcomm Hexagon, CPUs).
- Fluent in Python and C/C++. Has written CUDA on consumer Ampere (SM86) and Metal compute kernels. Has built inference runtimes (quantized matmul kernels, KV caches, graph executors). Reads papers directly; comfortable with linear algebra, probability, and systems-level reasoning.
- Has already worked through, at depth: inference-optimisation fundamentals (quantisation, sparsity, distillation, graph compilation), transformer inference and KV-cache management, distributed-inference basics (TP/PP/EP, collectives), CPU inference on Xeon/EPYC, Apple-silicon inference (CoreML/ANE/Metal/llama.cpp), edge/mobile inference, Intel/AMD CPU microarchitecture, Hexagon NPU programming, quant-finance foundations, an LLM-training syllabus and a Blackwell-serving syllabus.
- Learns for **immediate application at work**. The application/assessment happens in his job — never in the material.
- A **vivid visual learner**: a diagram of the mechanism beats a paragraph about it; a table beats a list of comparatives; a plot with real numbers beats an adjective.

## Pitch
- Depth: staff-engineer / PhD-seminar. Skip definitions a strong graduate already has; define only what is genuinely new or contested.
- Assume he will go read the primary source afterwards — cite precisely enough (section, table, page) that he can.
- Treat him as a peer: candid framing about what matters, what doesn't, what is hype, where the field disagrees, and what is likely to change.

## What he wants
- Exact figures with units, versions, and dates, each with a citation.
- Mechanisms over metaphors. Show the arithmetic when a number is derived.
- Comparisons as tables; trade-offs as explicit pros/cons with sources.
- A "where experts disagree" synthesis in every module — the point of reading many sources is to see the disagreement.
- The as-of date stated, and a note on what is moving.

## What he does not want
- Overviews, "in this section we will…", motivational preamble, summaries that repeat the section, "in conclusion".
- Beginner analogies, hedged filler ("it is important to note that"), corporate marketing language.
- **Exercises, quizzes, self-assessment questions, deliverables, projects, homework.** None. Ever.
- Invented or rounded-for-convenience numbers, fabricated URLs, or claims without a source. If it cannot be sourced, mark it `<mark class="unverified">` or cut it.
- Region-specific framing unless the topic is region-specific. Finance topics are general, not India-specific.

## Domains (typical requests)
ML systems and inference serving (primary); LLM training and post-training; accelerator and CPU architecture (GPU, TPU, NPU, Hexagon, Apple silicon, x86/ARM); compilers and kernels; distributed systems and system design; software-engineering practice; quantitative finance (general); mathematics underpinning the above.
