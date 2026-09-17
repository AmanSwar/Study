# AI Inference Hardware — outline

As of 2026-09-17. Kind: deep dive, flat, 6 pages. Sources: `sources.json` (50 sources, S1-S50; coverage "well" on all 14 subtopics — nothing "thin" or "none").

**Revision note (this pass):** Aman rejected the prior plan's exclusion of three areas (next-gen roadmap, official cloud pricing, training-workload benchmarking) and asked for all three folded in. A supplemental research pass added S30-S50 to close exactly those three gaps, and all three are now "well" covered. This revision keeps the page count at 6 (the deep-dive guide's hard maximum) and absorbs the new material by expanding scope and word budgets on the pages where it belongs, rather than adding a 7th page. Total estimate rose from ~53,000 to ~60,500 words across the same six pages; no page exceeds the 12,000-word cap.

## Reading order and rationale

1. **`01-nvidia-hopper-blackwell`** — NVIDIA Hopper (H100/H200), Blackwell (B200/GB200 NVL72), and the Rubin roadmap. Goes first: NVIDIA is the reference architecture (Transformer Engine, NVLink) the other vendors are implicitly compared against, and its own MLPerf/marketing claims (the "30x" number) need to be decomposed once, early. **Change this pass:** the Rubin/Vera Rubin roadmap section is now a real, citable subsection (Vera CPU + Rubin GPU platform, NVLink6, Vera Rubin NVL72, Rubin CPX) grounded in NVIDIA's own newsroom and product pages (S30-S33), not a bounded unverified aside — word budget raised 9,000 → 10,500.
2. **`02-amd-instinct-cdna`** — AMD Instinct MI300X/MI325X/MI350X/MI355X (CDNA3/CDNA4), now extended through the CDNA5-based MI400 series (MI455X/MI430X) and the Helios rack, grounded in AMD's own product pages and IR press release (S34-S37), including AMD's own comparison chart against Rubin (presented as AMD's claim here, adjudicated on page 6). Word budget raised 8,500 → 10,000.
3. **`03-google-tpu-v5-v7`** — Google TPU v5e/v5p/v6e (Trillium)/v7 (Ironwood). Unchanged from the prior plan: a structurally different design (systolic-array MXU, torus interconnect, pod-scale-first), placed after the two GPU vendors so the ICI-mesh contrast is legible.
4. **`04-aws-trainium-inferentia`** — AWS Trainium1/2 and Inferentia1/2 (NeuronCore v1-v3). Unchanged architecturally; the scope note is sharpened to say plainly that AWS has zero submissions to MLPerf Inference, InferenceMAX, *and* MLPerf Training (confirmed this pass), with the full treatment on page 5.
5. **`05-inference-benchmarking-throughput-mechanics`** — MLPerf Inference/InferenceMAX methodology (unchanged), **plus MLPerf Training methodology and cross-vendor time-to-train results**, added this pass as a supporting data point, not a scope change: same standards body, a genuinely different measurement paradigm (time-to-train vs. tokens/sec), with results for NVIDIA/AMD/Google and AWS's documented non-participation. Placed after all four hardware pages and before the comparison page, same as before. Word budget raised 9,000 → 11,500.
6. **`06-hardware-selection-model-fit`** — cross-vendor comparison page, last per the deep-dive convention. **Change this pass:** now also carries official cloud on-demand/committed-use pricing across AWS/GCP/Azure (a genuine $/hour dimension, distinct from the existing $/million-tokens data) and a "next-gen preview" section adjudicating the new Rubin-vs-MI455X paper-spec disagreement. Still introduces no new chip specs or benchmark methodology of its own — synthesizes pages 1-5 only. Word budget raised 9,500 → 11,500.

Pages 1-5 are each independently readable (no forward references required); page 6 is written last-to-read by design since it cites facts established on the other five.

## What is deliberately excluded and why

Per Aman's explicit request, none of the three previously-excluded areas (next-gen roadmap, official cloud pricing, training-workload benchmarking) are excluded any longer — all three are now folded into the pages above with real citations. What remains out of scope, and why:

- **Exercises, quizzes, deliverables, or assessments** — none, per the learner profile (a standing rule, unrelated to this revision).
- **The generation *after* next** (NVIDIA Rubin Ultra/Feynman, AMD MI500/UAL256) as confirmed specs — genuinely one step further than what any vendor has published verbatim. S35 (SemiAnalysis) gives directional MI500/UAL256 context (256-chip UALink domain, "late 2027") and is used only as a brief forward pointer inside page 2's MI400 topic; Rubin Ultra/Feynman have no fetched primary source at all and are not mentioned. This is a narrower, later boundary than the "next-gen roadmap" Aman asked for (Rubin and MI400/Helios, both now well-sourced and fully included) — it is not a re-exclusion of that request.
- **Deep treatment of training mechanics** (MFU derivation, checkpointing, optimizer-state sharding) — Aman's own framing for this request was explicit: the training-benchmark addition is "a supporting data point alongside the inference focus, not a scope change away from inference." Page 5 covers MLPerf Training's *methodology and results* (how time-to-train is scored, what NVIDIA/AMD/Google/AWS's standing is) but does not re-derive training-specific arithmetic the way pages 1-4 do for inference-serving arithmetic.
- **AWS's own P5/P5e/P5en/Trn1/Trn2 pricing as a directly-fetched vendor page** — unlike Inf2 (S40) and unlike GCP/Azure, these AWS pages load pricing via a JS widget that plain `curl` cannot execute, and the ~480MB AWS bulk Price List API was impractical to fetch/parse in this pass. The pricing itself is *not* excluded — page 6 uses cross-checked secondary aggregators (Vantage.sh S42, Spare Cores S43, Techzine/DCD S44) instead, with one specific Vantage.sh data point (p5e.48xlarge) and one (trn2.48xlarge) explicitly flagged and excluded as internally inconsistent in favor of the corroborated alternative.

## Where coverage is thin

Nothing in `sources.json.coverage` is marked "thin" or "none" this pass (all 14 subtopics are "well" covered) — but three specific facts are worth flagging as genuinely thin/absent rather than a gap the writers should paper over:

- **AWS Trainium/Inferentia has zero third-party benchmark datapoints anywhere**: not in MLPerf Inference, not in InferenceMAX (explicitly "planned within two months" per S25, still not delivered as of this research), and not in MLPerf Training (confirmed via WebSearch — AWS does not appear in any v5.0/v5.1 Training participant list). Page 4 and page 5 both state this plainly; AWS's own peak-FLOPS marketing claims are presented as AWS's claims, not corroborated ones.
- **Google's most recent *official* Trillium-vs-v5p training comparison is the MLPerf Training v4.1 round (Nov 2024, S50)** — no v5.0/v5.1 equivalent blog was found, so page 5's Google training data point is one round older than NVIDIA's and AMD's.
- **The Rubin-vs-MI455X spec discrepancy is unresolved by design**, not by omission: AMD's own comparison chart (S37) states Rubin at 35 PFLOPS FP4 / 22.0TB/s, while NVIDIA's own materials (S30) state 50 PFLOPS NVFP4 for "inference" and give no HBM4 bandwidth figure at all. Neither vendor states a common basis (dense vs. sparse, training vs. inference peak), so page 6 treats this as an open disagreement (one of four), not something to adjudicate to a single number.

## Source-map summary

- 50 sources total: 27 official (vendor spec pages/data sheets/docs/pricing pages/press releases), 4 standard (MLCommons), 11 blog (vendor engineering blogs), and 8 analysis (independent benchmarking/microarchitecture/pricing-aggregator outlets — Chips and Cheese, SemiAnalysis x3, Artificial Analysis, Junyi Hou, Introl, Vantage.sh, Spare Cores).
- All 14 subtopics are covered "well"; nothing is marked "thin" or "none."
- 6 sourced disagreements (up from 4): TPU-vs-GPU cost (D1), AMD-vs-NVIDIA who-wins (D2), vendor MLPerf vs. independent benchmarks (D3), what NVIDIA's "30x" figure actually measures (D4), whose next-gen flagship wins on paper — Rubin vs. MI455X (D5, new), and whether the specific $0.39/chip-hour TPU claim holds up against Google's own pricing page (D6, new) — each assigned to exactly one page.

Top 8 sources (four repeat from the prior pass; four are new this pass):
1. **S1/S2 — NVIDIA H100/H200 product pages** (NVIDIA, 2026): vendor primary spec-of-record for Hopper FLOPS, memory, and interconnect.
2. **S3/S4 — NVIDIA GB200 NVL72 product page + Blackwell architecture page** (NVIDIA, 2026): primary spec sheet for the rack-scale Blackwell reference unit and its chiplet/NVLink design.
3. **S30/S31 — NVIDIA Rubin newsroom press release + Vera Rubin platform page** (NVIDIA, 2026, new): primary source for the Rubin/Vera Rubin roadmap — Vera CPU, Rubin GPU, NVLink6, 2H 2026 availability.
4. **S7-S10 — AMD Instinct MI300X/MI325X/MI350X/MI355X data sheets** (AMD, 2025): vendor data sheets, the primary spec source for all four current Instinct SKUs.
5. **S36/S37 — AMD Helios rack product page + MI400 series product page** (AMD, 2026, new): primary source for MI455X/MI430X and Helios, including AMD's own comparison chart vs. Vera Rubin.
6. **S12-S15 — Google Cloud TPU v5e/v5p/v6e/v7 docs** (Google, 2026): live vendor system-architecture documentation, per-chip and per-pod specs.
7. **S17-S19, S21 — AWS Neuron/Trainium2/Trn2/Inf2 architecture docs** (AWS, 2026): vendor architecture documentation, the authoritative per-chip and per-instance spec source.
8. **S38/S39/S41 — Google Cloud TPU/GPU pricing pages + Azure Retail Prices API** (2026, new): official, directly-fetched cloud on-demand/committed-use pricing — the primary evidence resolving D6 and grounding page 6's new pricing table.

Also load-bearing: **S23-S28** (MLCommons MLPerf Inference v5.0/v5.1, SemiAnalysis InferenceMAX, Artificial Analysis, Junyi Hou's TPU-TCO fact-check) for the inference-benchmark and cost-per-token evidence; **S45-S50** (new — MLCommons MLPerf Training v5.0/v5.1, NVIDIA/AMD/Google training blogs) for the training-benchmark supporting data point.

## Page list

| # | id | shortTitle | est. words | sources | disagreements |
|---|----|-----------|-----------|---------|----------------|
| 1 | 01-nvidia-hopper-blackwell | NVIDIA Hopper/Blackwell | 10,500 | S1,S2,S3,S4,S5,S6,S30,S31,S32,S33 | D4 |
| 2 | 02-amd-instinct-cdna | AMD Instinct CDNA3/4 | 10,000 | S7,S8,S9,S10,S11,S34,S35,S36,S37 | — |
| 3 | 03-google-tpu-v5-v7 | Google TPU v5-v7 | 8,500 | S12,S13,S14,S15,S16 | — |
| 4 | 04-aws-trainium-inferentia | AWS Trainium/Inferentia | 8,500 | S17,S18,S19,S20,S21,S22 | — |
| 5 | 05-inference-benchmarking-throughput-mechanics | Benchmark Mechanics | 11,500 | S23,S24,S25,S26,S27,S45,S46,S47,S48,S49,S50 | D3 |
| 6 | 06-hardware-selection-model-fit | Hardware Selection | 11,500 | S3,S9,S15,S18,S25,S27,S28,S29,S30,S31,S37,S38,S39,S40,S41,S42,S43,S44 | D1, D2, D5, D6 |

Total estimated: ~60,500 words across 6 pages (up from ~53,000; every page still within the 6,000-12,000 band).
