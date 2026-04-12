# aman.study

**Live site: [aman-study.vercel.app](https://aman-study.vercel.app)**

PhD-level study material across 4 tracks, 104 modules.

## Tracks

### ML Systems Engineering (39 modules + 5 appendices)
Production inference systems — quantization, transformers, distributed inference, CPU/GPU/Apple Silicon/Edge AI, voice AI, and RAG infrastructure.

### Intel & AMD CPU Architecture (27 modules + 5 appendices)
CPU design deep dive — x86 architecture, Xeon Scalable, AMD EPYC, performance engineering, and production inference engines.

### Qualcomm Hexagon NPU (10 modules)
On-device inference — Hexagon VLIW architecture, HVX vector programming, HTA/HMX tensor accelerators, and mobile optimization.

### From Zero to Quant (28 chapters)
Quantitative trading for ML engineers — financial markets, alpha research, backtesting, portfolio construction, risk management, and live execution on Indian markets.

## Tech Stack

- **Framework**: Next.js 16 (App Router, TypeScript)
- **Styling**: Tailwind CSS 4
- **Math**: KaTeX
- **Diagrams**: Interactive SVGs with Framer Motion
- **Markdown**: react-markdown + remark-gfm + remark-math
- **Deployment**: Vercel

## Development

```bash
cd website
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Structure

```
.
├── computer science/          # Source markdown files
│   ├── MLsys/                 # 39 modules across 10 parts
│   ├── intel/                 # 27 modules across 7 parts
│   └── qualcomm/              # 10 modules
├── From_Zero_to_Quant/        # 28 chapters
└── website/                   # Next.js application
    ├── app/                   # 139 routes
    ├── components/            # UI, content, diagram components
    ├── content/               # Track metadata
    └── lib/                   # Markdown loader, types
```
