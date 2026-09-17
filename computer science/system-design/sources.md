# Sources — System design from classical internet-scale systems to AI inference platforms and AI training infrastructure

As of 2026-09-17 · 66 sources · course

## Coverage

**Well covered (26):** request-path, storage-fundamentals, consistency-models, consensus-coordination, distributed-file-object-storage, batch-stream-processing, microservices-api-design, observability-slo, security-fundamentals, google-search, netflix, twitch, uber-dispatch, api-gateway-edge, rate-limits-quotas, billing-metering, request-routing-admission, streaming-protocols, agent-harness-sandboxes, prompt-caching-tiers, safety-moderation, checkpointing-storage-cluster, cluster-scheduling-orchestration, fault-tolerance-elasticity, rl-training-infra, rl-frameworks-comparison

**Thin (20):** caching-invalidation, messaging-streams, resilience-patterns, backpressure-rate-limiting, gmail, youtube-vitess, whatsapp, twitter-timeline, instagram, stripe-payments, airbnb, auth-identity, gpu-backend-handoff, data-privacy-retention, llm-observability, multi-tenancy-isolation, capacity-cost-management, data-pipelines-training, post-training-pipelines, frontier-synthesis

**Not covered (1):** incident-patterns

## Subtopics

- `request-path` — Scalability and the request path: DNS, load balancers, reverse proxies, CDNs, edge
- `caching-invalidation` — Caching at every layer and invalidation
- `storage-fundamentals` — Relational vs NoSQL, replication, partitioning/sharding, transactions, indexes
- `consistency-models` — Consistency models, CAP/PACELC
- `consensus-coordination` — Consensus and coordination: Paxos/Raft, leases, ZooKeeper/etcd
- `distributed-file-object-storage` — Distributed file/object storage: GFS, HDFS, S3-style, Tectonic
- `messaging-streams` — Messaging and streams: Kafka-style logs, queues, exactly-once, idempotency
- `batch-stream-processing` — Batch and stream processing: MapReduce -> Spark/Flink lineage
- `microservices-api-design` — Microservices vs monolith, API design, service mesh
- `backpressure-rate-limiting` — Backpressure and rate limiting as general mechanisms
- `observability-slo` — Observability (metrics/logs/traces), SLOs, capacity planning
- `resilience-patterns` — Failure modes and resilience patterns: timeouts, retries, circuit breakers, bulkheads
- `security-fundamentals` — Security fundamentals: authn/authz, secrets, zero trust
- `google-search` — Google Search: crawl -> index -> serve, GFS/Bigtable/Spanner lineage
- `gmail` — Gmail infrastructure
- `youtube-vitess` — YouTube: upload -> transcode -> CDN -> adaptive streaming -> recommendations, plus Vitess
- `netflix` — Netflix: Open Connect CDN, encoding ladder, chaos engineering, recommendation systems
- `twitch` — Twitch: live ingest, low-latency HLS, chat at scale
- `whatsapp` — WhatsApp: Erlang, messaging at scale
- `twitter-timeline` — Twitter/X timeline: fan-out on write vs read
- `instagram` — Instagram: feeds, storage, sharding
- `stripe-payments` — Stripe: ledgers, idempotency keys, API design
- `uber-dispatch` — Uber: dispatch, geo-indexing (H3), real-time systems
- `airbnb` — Airbnb: search, booking, SOA migration
- `api-gateway-edge` — Edge and API gateway for AI platforms
- `auth-identity` — Identity and auth: API keys, OAuth, orgs/workspaces, service accounts
- `rate-limits-quotas` — Rate limits and quotas for LLM APIs
- `billing-metering` — Token metering, credits, and billing: ledger design, idempotency, holds, reconciliation
- `request-routing-admission` — Request routing, admission control, queueing, SLO tiers, fallbacks
- `gpu-backend-handoff` — The interface between the platform and the multi-node GPU backend
- `streaming-protocols` — Streaming: SSE/WebSockets, partial results, cancellation, resumption
- `agent-harness-sandboxes` — Tool use, agent harnesses, sandboxes for code execution, session/conversation state
- `prompt-caching-tiers` — Platform-level prompt/prefix caching tiers and billing implications
- `safety-moderation` — Safety and moderation pipelines: pre/post filters, classifiers, policy routing
- `data-privacy-retention` — Data: conversation/log storage, retention, privacy/tenancy, consent for training
- `llm-observability` — Observability specific to LLM serving: per-token metrics, tracing, cost attribution
- `multi-tenancy-isolation` — Multi-tenancy and isolation on inference platforms
- `capacity-cost-management` — Capacity and cost management for inference platforms
- `incident-patterns` — Incident patterns specific to LLM serving platforms
- `data-pipelines-training` — Training data pipelines: crawl/ingest -> filtering/dedup -> tokenization -> sharded datasets
- `checkpointing-storage-cluster` — Storage and checkpointing at scale; cluster network topology
- `cluster-scheduling-orchestration` — Cluster scheduling and orchestration: Slurm, Kubernetes, Ray, MAST
- `fault-tolerance-elasticity` — Fault tolerance and elasticity: checkpoint/restart, stragglers, node health
- `post-training-pipelines` — Post-training pipelines: SFT/preference data flows, evaluation gates, model registries
- `rl-training-infra` — RL training infrastructure: environments as services, rollout generation, trainer/rollout decoupling
- `rl-frameworks-comparison` — Open-source RL framework architectures: verl, OpenRLHF, SkyRL, AReaL, slime, NeMo-RL
- `frontier-synthesis` — Frontier and synthesis: what is unsettled and changing across classical, inference, and training systems

## Disagreements found

### Strong, synchronously-replicated consistency vs. eventual consistency for large-scale storage

- **Position A**: Two-phase commit over Paxos groups, with a globally synchronized clock (TrueTime), is worth its latency cost because it gives externally-consistent transactions at global scale; Spanner was built specifically because Bigtable's eventual cross-datacenter replication and lack of cross-row transactions caused frequent complaints from application teams. [S4]
- **Position B**: For always-on, latency-critical services, ACID and cross-datacenter strong consistency are the wrong trade: Dynamo deliberately gives up consistency during failures/partitions to remain 'always writeable', pushing conflict resolution to reads and the application, because Amazon's experience showed ACID data stores 'tend to have poor availability.' [S5, S11]

### Is Raft's understandability advantage over Paxos decisive in practice?

- **Position A**: Raft's decomposition (leader election / log replication / safety) is easier to teach and implement correctly: a 43-student study found participants answered Raft questions better than Paxos questions after learning both. [S1]
- **Position B**: Production-critical systems that predate or coexist with Raft's adoption still run mature Paxos implementations rather than migrating: Chubby (behind Bigtable) and Spanner's spanservers both use Paxos, suggesting operational maturity of an existing implementation outweighs a pedagogical understandability advantage once a system is already built and hardened. [S3, S4]

### Fixed-window vs. sliding-window rate limiting

- **Position A**: Fixed windows are simpler and cheaper to implement (a counter that resets on a clock boundary), but allow up to 2x the nominal limit in a burst that straddles a window boundary. [S43]
- **Position B**: Sliding windows evaluate the trailing interval continuously and correctly reject the same boundary-straddling burst, at the cost of tracking request timestamps rather than a single counter. [S43]

### How should token-based LLM rate limits be enforced when true cost is only known after generation?

- **Position A**: Estimate pessimistically at request start and adjust the estimate as the request streams: Anthropic's ITPM is estimated at the beginning of a request and adjusted during it to reflect actual input tokens used, and OTPM counts only tokens actually generated in real time. [S36]
- **Position B**: Admit optimistically and reconcile after the fact: Envoy AI Gateway checks whether processing a request would exceed the limit using current state, processes it, and only then counts the resulting token usage toward the total -- a design that can let a burst of concurrent requests each pass the pre-check before any of their usage lands. [S40]

### Should cached tokens count against a customer's rate limit?

- **Position A**: Cached input tokens should be excluded from the throughput limit: for most Claude models, cache_read_input_tokens do not count toward ITPM, so heavy prefix reuse (e.g. 80% cache hit rate) multiplies effective throughput headroom well beyond the nominal limit. [S36]
- **Position B**: Cached input tokens should still count toward the platform's throughput limit: OpenAI's documentation states cached input tokens still count toward tokens-per-minute limits even though they are billed at a discounted rate. [S38]

### Should safety moderation be a pre-generation blocking gate or a post-generation signal?

- **Position A**: Moderation scores should be treated as policy signals for the calling application, not an automatic blocking decision; results should be reviewed before showing output to a user, and streamed moderation scores only arrive after the full output is generated. [S44, S50]
- **Position B**: Safety should be a continuous pipeline spanning policy design, training-time shaping of model behavior, pre-deployment evaluation gates, and real-time classifier enforcement across the full lifecycle -- not a single bolt-on filter step around generation. [S45]

### Dedicated per-tenant GPU pools vs. shared pooled GPUs with per-key limits for inference multi-tenancy

- **Position A**: Dedicated GPU clusters per model and customer tier eliminate noisy-neighbor variance, at the cost of lower aggregate utilization when a given tenant's traffic is bursty or idle. [S47]
- **Position B**: Shared GPU pools behind a gateway that enforces per-virtual-key budgets and rate limits (LiteLLM's model) maximize utilization across tenants, accepting some risk of cross-tenant interference that must be bounded by the rate-limit/queueing layer instead of by physical isolation. [S39, S40]

### Synchronous vs. asynchronous trainer/rollout coupling in RL post-training infrastructure

- **Position A**: Fully asynchronous, microservice-decoupled generation and training (with bounded staleness via version rejection, depth bounding, or importance-sampling correction) delivers large speedups: AReaL reports 2.77x over synchronous systems with comparable or better final performance, and PipelineRL's per-forward-pass weight swap keeps staleness near zero without ever blocking generation. [S60, S63]
- **Position B**: Bounded-queue, Ray-orchestrated designs (verl's 3D-HybridEngine resharding, NeMo-RL, SkyRL) keep training and rollout more tightly coupled through explicit resharding or a capacity-limited buffer, trading some of the asynchronous speedup for a simpler mental model and easier debugging of a single dataflow graph. [S56, S57, S62, S63]

### HPC batch scheduler vs. cloud-native orchestrator for GPU training clusters

- **Position A**: A centralized scheduler with a simple heartbeat model (Slurm's slurmctld/slurmd) is the traditional HPC answer: one controller assigns jobs to nodes and arbitrates a pending-work queue, with no declarative reconciliation loop. [S65]
- **Position B**: Cloud-native orchestrators (the Borg-to-Kubernetes lineage, and Ray's actor/task model) favor declarative desired-state reconciliation and finer-grained resource primitives (priority, quota, admission control in Borg; tasks/actors/object store in Ray), and are what modern RL frameworks (8 of 16 surveyed) build their trainer/rollout orchestration on top of instead of Slurm. [S9, S64, S63]

## Sources

### S1 · paper · Stanford University — In Search of an Understandable Consensus Algorithm (Extended Version) (2014, Extended version, published 2014-05-20)

https://raft.github.io/raft.pdf · accessed 2026-09-17

*Primary paper introducing Raft, the consensus algorithm underlying etcd, Consul, CockroachDB, and most modern replicated logs.*

_Covers: consensus-coordination_

Facts:
- Raft separates leader election, log replication, and safety as independent subproblems (§4)
- A 43-student user study found Raft significantly easier to understand than Paxos (Abstract)
- A cluster of 5 servers tolerates the failure of any 2 (§2)
- Election timeouts are randomized from a fixed interval, e.g. 150-300ms, to avoid repeated split votes (§5.2)
- A log entry is committed once a majority of the cluster has replicated it; a minority of slow servers does not affect performance (§2)
- Raft uses a strong leader: log entries flow only from leader to followers, simplifying management (Abstract)
- Raft's membership-change mechanism uses joint consensus with overlapping majorities of old and new configurations so the cluster keeps operating during reconfiguration (Abstract)

### S2 · paper · Google — The Google File System (2003, SOSP '03)

https://research.google.com/archive/gfs-sosp2003.pdf · accessed 2026-09-17

*Primary paper on GFS, the distributed file system underlying Bigtable and the architectural template for HDFS.*

_Covers: distributed-file-object-storage, storage-fundamentals_

Facts:
- GFS clusters store hundreds of terabytes across thousands of disks on over a thousand machines, accessed concurrently by hundreds of clients (Abstract)
- Chunk size is 64 MB, much larger than typical filesystem block sizes (§2.5)
- GFS uses a single master architecture; clients never read/write file data through the master, only metadata (§2.3-2.4)
- Each chunk is replicated on multiple chunkservers (default 3 replicas), with configurable replication levels per namespace region (§2.3)
- The master maintains less than 64 bytes of metadata per 64MB chunk, keeping the entire metadata working set in memory (§2.6.1)
- Atomic record append lets many clients append concurrently without extra client-side synchronization, at an offset chosen by GFS (§3.3)
- Mutation order is controlled via leases: the master grants a chunk lease to one replica (the primary) with a 60-second initial timeout, renewable via piggybacked HeartBeat messages (§3.1)
- Ideal elapsed time to push B bytes to R replicas is B/T + RL, with typical network throughput T of 100 Mbps and latency L well under 1ms, so 1MB distributes in about 80ms (§3.2)

### S3 · paper · Google — Bigtable: A Distributed Storage System for Structured Data (2006, OSDI '06)

https://www.usenix.org/legacy/event/osdi06/tech/chang/chang.pdf · accessed 2026-09-17

*Primary paper on Bigtable, Google's wide-column store underlying web indexing, Google Earth, Google Finance, and part of the Spanner storage stack.*

_Covers: storage-fundamentals, google-search_

Facts:
- Bigtable's data model is a sparse, distributed, persistent multi-dimensional sorted map indexed by (row, column, timestamp) (§2)
- Row ranges are dynamically partitioned into tablets, the unit of distribution and load balancing; tables start as one tablet and split automatically to ~100-200MB by default (§4, §5.1)
- Bigtable uses the Chubby lock service (Paxos-based, five replicas) to ensure at most one active master and to store bootstrap/schema/ACL data (§4)
- Data is stored on GFS using the immutable SSTable file format with a 64KB block size and an in-memory block index (§4)
- A tablet server typically manages 10 to a thousand tablets (§5.1)
- Measured Chubby unavailability affected Bigtable server-hours by only 0.0047% on average across 14 clusters/11 Chubby instances (worst single cluster: 0.0326%) (§4)
- Bigtable does not support general cross-row transactions, only single-row atomic read-modify-write (§3)

### S4 · paper · Google — Spanner: Google's Globally-Distributed Database (2012, OSDI '12)

https://research.google.com/archive/spanner-osdi2012.pdf · accessed 2026-09-17

*Primary paper on Spanner, the first system to provide externally-consistent distributed transactions at global scale via the TrueTime API; the architectural successor to Bigtable/Megastore at Google.*

_Covers: storage-fundamentals, consistency-models, consensus-coordination, google-search, gmail_

Facts:
- Spanner shards data across many sets of Paxos state machines in datacenters spread all over the world (§1)
- At least 300 applications within Google use Megastore, including Gmail, Picasa, Calendar, Android Market, and AppEngine, despite its relatively low performance, because its data model is simpler to manage (§2.3)
- A zone has one zonemaster and between one hundred and several thousand spanservers; each spanserver is responsible for 100 to 1000 tablets (§2)
- Each spanserver implements a single Paxos state machine on top of each tablet, with long-lived leases defaulting to 10 seconds; every Paxos write is logged twice (tablet log and Paxos log) (§2.1)
- A directory (bucket of contiguous keys sharing a prefix) is the unit of data placement and geographic replication; a 50MB directory can typically be moved between Paxos groups in a few seconds (§2.2)
- Spanner exposes a novel TrueTime API that directly exposes clock uncertainty, keeping it generally under 10ms using GPS and atomic clock references (Abstract, §1)
- Spanner is the first system to provide externally-consistent distributed transactions at global scale, with commit timestamps reflecting serialization order (Abstract)

Conflicts: Bigtable (S3) provides only eventually-consistent replication across datacenters and no cross-row transactions; Spanner (S4) explicitly built synchronous cross-datacenter replication and general-purpose transactions to fix this, at the cost of running two-phase commit over Paxos.

### S5 · paper · Amazon — Dynamo: Amazon's Highly Available Key-value Store (2007, SOSP '07)

https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf · accessed 2026-09-17

*Primary paper on Dynamo, the eventually-consistent key-value store whose design (consistent hashing, vector clocks, quorum reads/writes, gossip) became the blueprint for Cassandra, Riak, Voldemort, and DynamoDB.*

_Covers: storage-fundamentals, consistency-models_

Facts:
- Dynamo deliberately sacrifices consistency under certain failure scenarios to achieve high availability, using an 'always writeable' design that never rejects customer updates (Abstract, §2.3)
- SLAs at Amazon are expressed at the 99.9th percentile, not the mean or median, because averages don't reflect the experience of the whole customer base (§2.2)
- Data is partitioned and replicated using consistent hashing with virtual nodes; conflicting writes are reconciled via vector clocks (§4.2, §4.4)
- Consistency among replicas during updates is maintained by a quorum-like technique and a decentralized replica synchronization protocol using Merkle trees for anti-entropy (Table 1)
- Initial Dynamo instances target a scale of up to hundreds of storage hosts (§2.1)
- Membership and failure detection use a gossip-based protocol, avoiding a centralized registry (Table 1)
- Dynamo compares itself explicitly to Bigtable: Dynamo targets pure key/value access requiring high write availability where updates are never rejected, even during partitions (§3.2)

Conflicts: Dynamo (S5) explicitly rejects the relational/ACID model used by Spanner (S4) for the same class of always-on shopping-cart-like services, arguing 'experience at Amazon has shown that data stores that provide ACID guarantees tend to have poor availability.'

### S6 · paper · Google — MapReduce: Simplified Data Processing on Large Clusters (2004, OSDI '04)

https://research.google.com/archive/mapreduce-osdi04.pdf · accessed 2026-09-17

*Primary paper defining the MapReduce programming model and runtime, the ancestor of Hadoop and the batch-processing lineage that Spark and Flink respond to.*

_Covers: batch-stream-processing_

Facts:
- The map function has type (k1,v1) -> list(k2,v2); reduce has type (k2,list(v2)) -> list(v2) (§2.2)
- Input is split into M pieces of 16-64MB (user-controllable); the master assigns idle workers to map or reduce tasks (§3.1)
- On worker failure, completed and in-progress map tasks are reset to idle and rescheduled because their output is on local disk (inaccessible); completed reduce tasks need not be redone since output is in the global file system (§3.3)
- Hundreds of MapReduce programs have been implemented and upwards of one thousand MapReduce jobs run on Google's clusters every day (Abstract)
- Typical clusters consist of hundreds/thousands of dual-processor x86 machines with 2-4GB RAM, connected by 100Mbps-1Gbps Ethernet, with commodity IDE disks (§3)

### S7 · paper · Google — Zanzibar: Google's Consistent, Global Authorization System (2019, USENIX ATC '19)

https://www.usenix.org/system/files/atc19-pang.pdf · accessed 2026-09-17

*Primary paper on Zanzibar, Google's global relationship-based authorization system used by Calendar, Cloud, Drive, Maps, Photos, and YouTube; the model behind the open-source SpiceDB/OpenFGA authz systems.*

_Covers: security-fundamentals_

Facts:
- Zanzibar stores more than two trillion access control lists and performs millions of authorization checks per second (§1)
- Zanzibar has maintained 95th-percentile latency under 10ms and availability above 99.999% over 3 years of production use (Abstract)
- ACLs are relation tuples of the form <object>#<relation>@<user>, and can be group tuples where the user is itself a userset, supporting nested group membership (§2.1)
- Zanzibar provides external consistency for authorization decisions via a globally distributed database with 'zookies', letting a client ensure a check reflects ACL data no older than a specified change (§1)
- Zanzibar responds to more than 95% of authorization checks within 10 milliseconds and has maintained more than 99.999% availability for the last 3 years (§1)

### S8 · paper · Yahoo! — ZooKeeper: Wait-free Coordination for Internet-scale Systems (2010, USENIX ATC '10)

https://www.usenix.org/legacy/event/atc10/tech/full_papers/Hunt.pdf · accessed 2026-09-17

*Primary paper on ZooKeeper, the coordination kernel behind leader election, service discovery, and distributed locks in Kafka, HBase, and countless other systems; the direct ancestor to etcd's design goals.*

_Covers: consensus-coordination_

Facts:
- ZooKeeper guarantees FIFO client ordering of all operations and linearizable writes, but reads are served locally by any server and are not linearized, since read workloads dominate at ratios of 2:1 to 100:1 (Abstract, §1)
- ZooKeeper implements a leader-based atomic broadcast protocol called Zab to guarantee update linearizability (§1)
- Data is organized as znodes in a hierarchical namespace like a filesystem; znodes are typed regular or ephemeral (deleted when the creating session ends) (§2.1)
- Watches let clients receive one-time notifications of znode changes without polling; this is the primitive used to build leader election and locks on top (§2.1)
- ZooKeeper is described as a wait-free coordination kernel deliberately exposing simple primitives (not blocking locks) so applications can build their own higher-level coordination recipes (Abstract)

### S9 · paper · Google — Large-scale cluster management at Google with Borg (2015, EuroSys '15)

https://research.google.com/pubs/archive/43438.pdf · accessed 2026-09-17

*Primary paper on Borg, Google's cluster manager and the direct design ancestor of Kubernetes; grounds cluster scheduling, admission control, and priority/quota mechanisms used across both classical services and AI training clusters.*

_Covers: cluster-scheduling-orchestration, observability-slo_

Facts:
- Borg cells run up to tens of thousands of machines; median cell size is about 10K machines (§2.2)
- Borg's workload has two main parts: long-running latency-sensitive services (e.g. Gmail, Docs, web search, BigTable) and batch jobs taking seconds to days (§2.1)
- In a representative cell, prod jobs get ~70% of CPU resources allocated and ~60% of CPU usage; ~55% of memory allocated and ~85% of memory usage (§2.1)
- Admission control uses priority and quota: quota is checked at admission time (not scheduling time) and jobs with insufficient quota are rejected immediately upon submission (§2.5)
- Every task gets a stable 'Borg name service' (BNS) name recorded in Chubby that load balancers use to find where to route requests (§2.6)
- A preempted task typically gets a SIGTERM notice about 80% of the time before SIGKILL (§2.3)

### S10 · paper · LinkedIn — Kafka: a Distributed Messaging System for Log Processing (2011, NetDB '11 workshop)

https://notes.stephenholiday.com/Kafka.pdf · accessed 2026-09-17

*Primary paper by Kafka's creators, describing the design decisions (log-structured storage, pull model, stateless broker) that define the messaging/streaming lineage used across Part A and referenced across AI training data pipelines.*

_Covers: messaging-streams_

Facts:
- Each topic partition corresponds to a physical log implemented as a set of ~1GB segment files; a message is only exposed to consumers after being flushed (§3.1)
- Kafka relies on the OS page cache instead of an application-level cache, so a restarted broker keeps a warm cache (§3.1)
- Kafka uses a pull model (consumers pull from brokers) rather than push, which allows a consumer to deliberately rewind and re-consume data, unlike a queue (§3.1)
- The broker is stateless: it does not track which messages each consumer has read; a message is deleted after a fixed retention period, typically 7 days (§3.1)
- Messages are addressed by logical offset in the log rather than an explicit message id, avoiding auxiliary index structures (§3.1)
- Kafka uses sendfile() to transfer bytes from a log segment file to a socket without extra copies or kernel-user boundary crossings for efficient transfer (§3.1)

### S11 · blog · DBMS Musings (personal blog; author later at Yale/UMD) — Problems with CAP, and Yahoo's little known NoSQL system (2010, Published 2010-04-23; formalized as PACELC in Computer, Feb 2012)

http://dbmsmusings.blogspot.com/2010/04/problems-with-cap-and-yahoos-little.html · accessed 2026-09-17

*Primary source (original post) for the PACELC framework, which is the formulation Aman's learner profile and content standards expect over a bare CAP theorem treatment.*

_Covers: consistency-models_

Facts:
- PACELC asks: if there is a partition (P), how does the system trade off availability and consistency (A and C); else (E), in normal operation, how does it trade off latency and consistency (L and C)? (post body)
- PA/EL systems like Dynamo give up consistency for availability during a partition, and give up consistency for latency in normal operation (post body)
- PC/EC systems (fully ACID databases) refuse to give up consistency and pay both the availability and latency costs to achieve it (post body)
- PC/EL systems like Yahoo's PNUTS give up consistency for latency in normal operation but do not give up additional consistency during a partition (they give up availability instead) (post body)
- Systems that give up consistency for availability during partitions also tend to give up consistency for latency absent partitions, which is what CAP alone fails to explain (post body)

### S12 · textbook · O'Reilly Media — Designing Data-Intensive Applications (2026, 2nd edition, published February 2026, 650 pages)

https://www.oreilly.com/library/view/designing-data-intensive-applications/9781098119058/ · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*The standard graduate-level textbook synthesizing replication, partitioning, transactions, consistency, batch/stream processing into one coherent framework; content-standards-mandated grounding for Part A. Table of contents page returned 403 to automated fetch; citations should reference chapter numbers from the well-known 1st-edition structure (Replication ch.5, Partitioning ch.6, Transactions ch.7, Consistency and Consensus ch.9, Batch Processing ch.10, Stream Processing ch.11) until a writer can access a preview copy to confirm 2nd-edition renumbering.*

_Covers: storage-fundamentals, consistency-models, consensus-coordination, batch-stream-processing, messaging-streams_

_No facts extracted — see authority note above._

### S13 · paper · Google — BeyondCorp: Design to Deployment at Google (2016, USENIX ;login: vol. 41, no. 1, Spring 2016)

https://research.google.com/pubs/archive/44860.pdf · accessed 2026-09-17

*Primary Google paper on BeyondCorp, the origin of the 'zero trust' architecture pattern now standard in enterprise and platform security design.*

_Covers: security-fundamentals_

Facts:
- BeyondCorp gates access to applications based on device state and user identity rather than physical location or network, treating both internal and external networks as completely untrusted (Overview)
- The core components are the Trust Inferer, Device Inventory Service, Access Control Engine, Access Policy, Gateways, and Resources (Overview)
- The Trust Inferer continuously analyzes and annotates device state, setting the maximum trust tier a device may access and downgrading trust on missing security patches or infection signals (Components section)
- Resources are accessed only via gateways (SSH servers, web proxies, 802.1x-enabled networks) that enforce a minimum trust tier or assign a VLAN (Overview)
- Google's Device Inventory Service ingested billions of deltas from over 15 data sources at roughly three million per day, totaling over 80 terabytes (Device Inventory Service section)

### S14 · textbook · Google / O'Reilly — Site Reliability Engineering, Chapter 4: Service Level Objectives (2016, sre.google/sre-book (freely hosted))

https://sre.google/sre-book/service-level-objectives/ · accessed 2026-09-17

*Google's own canonical definitions of SLI/SLO/SLA, the vocabulary content-standards requires for the observability/SLO module.*

_Covers: observability-slo_

Facts:
- An SLI is a carefully defined quantitative measure of some aspect of the level of service provided (chapter body)
- An SLO is a target value or range for a service level measured by an SLI (chapter body)
- An SLA is an explicit or implicit contract with consequences for meeting or missing the SLOs it contains; missing an SLO without consequence means it was only an SLO, not an SLA (chapter body)
- Example SLI/SLO: 99% (averaged over 1 minute) of Get RPC calls will complete in less than 100ms (chapter body)
- Google Compute Engine's published availability target is 'three and a half nines' -- 99.95% (chapter body)

### S15 · repo · Netflix — Netflix/Hystrix wiki: How it Works (2018, v1.5.18 (last stable release; project in maintenance mode))

https://github.com/Netflix/Hystrix/wiki/How-it-Works · accessed 2026-09-17

*Netflix's own documentation of Hystrix, the circuit-breaker/bulkhead library that popularized the resilience patterns (timeouts, circuit breakers, bulkheads, fallbacks) now standard in service-mesh and gateway layers.*

_Covers: resilience-patterns_

Facts:
- Hystrix isolates points of access to remote systems to stop cascading failure and enable resilience where failure is inevitable (README)
- The circuit breaker trips from CLOSED to OPEN once request volume exceeds circuitBreakerRequestVolumeThreshold and the error percentage exceeds circuitBreakerErrorThresholdPercentage; after circuitBreakerSleepWindowInMilliseconds it allows one trial request through in a HALF-OPEN state (How it Works page)
- With thread-pool isolation, clients execute on separate threads so the calling thread (e.g. a Tomcat thread) can time out and walk away from a slow dependency (How it Works page)
- Semaphore isolation cannot enforce timeouts: if a semaphore-isolated dependency becomes latent, the parent thread stays blocked until the underlying network call itself times out (How it Works page)

### S16 · official · NGINX / F5 — Using nginx as HTTP load balancer (2026, live documentation)

https://nginx.org/en/docs/http/load_balancing.html · accessed 2026-09-17

*Official reference-implementation documentation for the four canonical load-balancing algorithms (round robin, least connections, ip-hash, weighted) at the reverse-proxy layer.*

_Covers: request-path_

Facts:
- Round robin (the default) has no session affinity: each subsequent request from the same client can land on a different server (doc body)
- Least-connected (least_conn) sends new requests to whichever upstream server currently has the fewest active connections, avoiding overloading a slow busy server (doc body)
- ip_hash uses the client IP as a hash key so a given client is always routed to the same server (except when that server is down), providing sticky sessions (doc body)
- A weight=3 directive on one server out of three sends 3 of every 5 requests to it (proportional weighted round robin) (doc body)

### S17 · official · Cloudflare — Content Delivery Network (CDN) Reference Architecture (2026, live documentation)

https://developers.cloudflare.com/reference-architecture/cdn-reference-architecture · accessed 2026-09-17

*Official CDN architecture documentation from a major CDN operator, covering origin/edge/PoP topology, tiered caching, and TTL-based freshness/invalidation.*

_Covers: request-path, caching-invalidation_

Facts:
- Cloudflare's edge network spans hundreds of cities worldwide, reaching 95% of the world's Internet-connected population within 50 milliseconds (doc body)
- With Tiered Cache, certain data centers act as reverse proxies to the origin for other data centers, increasing cache-hit rate and reducing origin load (doc body)
- Cache Reserve adds a higher cache tier with longer retention using R2 object storage; when the TTL on content in Cache Reserve expires, the content must be revalidated against the origin (doc body)

### S18 · paper · UC Berkeley — Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing (2012, USENIX NSDI '12)

https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf · accessed 2026-09-17

*Primary paper introducing Spark/RDDs, the system that succeeded MapReduce for iterative and interactive workloads and the direct conceptual ancestor of the batch/stream lineage Aman's brief asks for.*

_Covers: batch-stream-processing_

Facts:
- RDDs achieve fault tolerance via lineage (recording the transformations used to build a dataset) rather than data replication or logging updates to mutable state (§1)
- RDDs restrict writes to coarse-grained transformations (map, filter, join) rather than fine-grained updates to shared state, which is what makes cheap fault tolerance possible (§2.3)
- Spark is up to 20x faster than Hadoop for iterative applications, speeds up a real-world data analytics report by 40x, and can scan a 1TB dataset interactively with 5-7s latency (Abstract)
- RDDs let a user persist a dataset in memory across queries via a persist() call, avoiding recomputation when the same data is reused across operations (§2.2.1)

### S19 · official · Vitess (CNCF graduated project) — What is Vitess? (2026, vitess.io live docs)

https://vitess.io/docs/overview/whatisvitess/ · accessed 2026-09-17

*Official documentation for Vitess, the MySQL-sharding middleware built at YouTube/Google and now run by Slack, Square, JD.com; the canonical citation for YouTube's database layer.*

_Covers: youtube-vitess, storage-fundamentals_

Facts:
- Vitess served all of YouTube's database traffic for over five years (doc body)
- VTGate is a proxy that accepts MySQL protocol connections and routes queries to the appropriate shards; VTTablet is an agent alongside each MySQL instance managing queries, connections, and replication (doc body)
- A consistent data store (etcd, ZooKeeper, or Consul) maintains Vitess cluster topology state (doc body)
- Each MySQL connection carries a memory overhead between 256KB and almost 3MB depending on release; Vitess is built to handle thousands of connections cheaply on top of that constraint (doc body)

### S20 · blog · Netflix — Serving 100 Gbps from an Open Connect Appliance (2017, published 2017-09-29)

https://netflixtechblog.com/serving-100-gbps-from-an-open-connect-appliance-cdb51dda3b99 · accessed 2026-09-17

*Netflix's own account of the FreeBSD/NVMe engineering that lets a single CDN appliance saturate 100Gbps -- the numbers behind Open Connect, Netflix's purpose-built CDN.*

_Covers: netflix, request-path_

Facts:
- The stated goal and eventual result was serving 100 Gbps from a single FreeBSD-based Open Connect Appliance using NVMe-based storage (post body)
- Progression of measured throughput across hardware/software iterations: 40 Gbps (CPU-limited, Xeon E5-2697v2) -> 52 Gbps (PCIe Gen3 x8-limited) -> ~70-80 Gbps (prototype hardware) -> over 90 Gbps first achieved -> 90 Gbps sustained with 100% TLS traffic (post body)
- Serving entirely from memory rather than NVMe disk dropped performance from 40 Gbps CPU-limited to only 22 Gbps CPU-limited, an explicit memory-vs-disk trade-off (post body)
- 10-20% of the most popular titles are served from memory; the rest from NVMe SSD, exploiting a long-tail content-popularity distribution (post body)
- Detailed TCP statistics collection for every connection consumed 5-10% of CPU, so Netflix samples only a small percentage of connections (post body)

### S21 · blog · Netflix — Per-Title Encode Optimization (2017, published 2017-04-19)

https://netflixtechblog.com/per-title-encode-optimization-7e99442b62a2 · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*Netflix's own explanation of per-title encoding, the technique that replaced a single fixed bitrate ladder with a per-video-complexity-optimized ladder; the source could not be fetched directly (Cloudflare/medium anti-bot blocking persisted through direct fetch, browser-UA curl, and a reader-proxy retry) so facts below are drawn from the WebSearch tool's cached summary of the same page, not independently verified against the full text.*

_Covers: netflix_

Facts:
- Per-title encoding customizes the resolution/bitrate ladder to each title's complexity (motion, detail, colorfulness) instead of using one fixed ladder for all content (post summary (via search snippet))
- Netflix estimates about a 20% bitrate reduction from per-title encoding, and about 30% from doing it per-scene (post summary (via search snippet))

### S22 · official · principlesofchaos.org — Principles of Chaos Engineering (2019, v1 (2019 update))

https://principlesofchaos.org/ · accessed 2026-09-17

*The canonical published definition of chaos engineering, co-authored by the Netflix engineers who built Chaos Monkey and the Simian Army.*

_Covers: netflix, resilience-patterns_

Facts:
- Chaos Engineering is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions in production (landing page)
- Advanced principle: build a hypothesis around measurable steady-state output, not internal system attributes (advanced principles)
- Advanced principle: chaos experiments should run directly on production traffic to guarantee authenticity and relevance (advanced principles)
- Advanced principle: minimize blast radius -- some short-term negative impact is allowed but must be minimized and contained (advanced principles)

### S23 · blog · Netflix — System Architectures for Personalization and Recommendation (2013, netflixtechblog.com)

https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8 · accessed 2026-09-17

*Netflix's own account of its offline/nearline/online computation split for recommendations, the clearest published statement of that three-tier trade-off in the industry.*

_Covers: netflix_

Facts:
- Netflix runs recommendation computation across three modes: offline (batch, no latency limit), nearline (event-driven, near real-time), and online (synchronous, request-time, latency-SLA-bound) (post body)
- Online computation is explicitly bound by availability and response-time SLAs, which limits the complexity of algorithms that can run there (post body)
- Offline computation will not react quickly to new data, leading to staleness that can degrade the member experience (post body)
- Manhattan is Netflix's internal near-real-time event-flow computation framework, described by Netflix as similar to Storm but addressing different concerns; Hermes is the internal pub/sub layer notifying subscribers when a query result is ready, supporting HDFS, S3, or Cassandra as backing stores (post body)

### S24 · blog · Twitch — Ingesting Live Video Streams at Global Scale (2022, published 2022-04-26)

https://blog.twitch.tv/en/2022/04/26/ingesting-live-video-streams-at-global-scale/ · accessed 2026-09-17

*Twitch's own account of its live-ingest routing architecture (Intelligest, IRS, Capacitor, The Well), grounding the low-latency-live-video case study.*

_Covers: twitch_

Facts:
- Twitch operates nearly a hundred points of presence (PoPs) across geographic regions for stream ingest (post body)
- Named components: an Intelligest media proxy in each PoP that terminates live streams and queries the routing service; the Intelligest Routing Service (IRS) running in AWS as the stateful controller; Capacitor monitoring compute-resource fluctuations at origins; The Well monitoring backbone network status/utilization (post body)
- Twitch ingests RTMP and WebRTC and translates them to an internal canonical protocol across its backbone; PoPs previously used HAProxy for load balancing (post body)
- Twitch's static, precomputed routing solution cannot react to unexpected real-time traffic/system fluctuations, motivating a move to real-time dynamic routing (post body)

### S25 · blog · Twitch — Twitch Engineering: An Introduction and Overview (2015, published 2015-12-18)

https://blog.twitch.tv/en/2015/12/18/twitch-engineering-an-introduction-and-overview-a23917b71a25/ · accessed 2026-09-17

*Twitch's own numbers for chat, video, and API scale, and the evolution of its chat backend (Python -> Go) -- primary source for the live-video-plus-chat-at-scale case study.*

_Covers: twitch_

Facts:
- Twitch has peaked at over 2 million concurrent video streams and 30,000+ simultaneous streamers (post body)
- Chat delivers over 10 billion messages a day; web APIs average over 50,000 requests per second (post body)
- Chat architecture: Edge speaks IRC over raw TCP and WebSockets; Pubsub distributes messages internally across Edge nodes in a hierarchical fanout (post body)
- Twitch's chat implementation moved through NodeJS (failed to scale, hit a core bug), then Python, then to Go (post body)
- Twitch deliberately runs a large number of bare-metal PoPs rather than pure cloud, citing higher achievable video quality; it is also moving an increasing share of services to AWS to reduce operational overhead -- an explicit build-vs-buy trade-off stated by the same team (post body)

### S26 · blog · WhatsApp — 1 million is so 2011 (2012, published 2012-01-06)

https://blog.whatsapp.com/1-million-is-so-2011 · accessed 2026-09-17

*WhatsApp's own announcement of pushing a single server to 2 million TCP connections on Erlang/FreeBSD -- the primary source for the Erlang-messaging-at-scale case study.*

_Covers: whatsapp_

Facts:
- WhatsApp pushed a single server to over 2 million concurrent TCP connections (post body)
- The server ran FreeBSD 8.2-STABLE amd64 with Erlang R14B03 (erts-5.8.4), 24 CPU cores, 103GB physical memory, on Intel Xeon X5675 @ 3.07GHz (post body)
- At 2 million connections, CPU utilization was 37.9% user / 13.6% system / 41.9% idle -- there was still headroom to spare (post body)
- Kernel tuning parameters used: kern.ipc.maxsockets=2400000, kern.maxfiles=3000000, kern.maxfilesperproc=2700000, net.inet.tcp.tcbhashsize=524288 (post body)

### S27 · blog · Instagram — Sharding & IDs at Instagram (2012, instagram-engineering.com)

https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*Instagram's own explanation of its Postgres logical-sharding scheme and 64-bit ID generation, the primary source for the Instagram storage case study. The origin domain refused direct connections (ECONNREFUSED) and reader-proxy retries returned errors, so facts below are drawn from the WebSearch tool's summary of the same page, not independently verified against full text.*

_Covers: instagram_

Facts:
- Instagram evaluated NoSQL but chose to shard data across several thousand logical shards mapped onto far fewer physical PostgreSQL servers, using Postgres schemas to make this easy to script/administrate (post summary (via search snippet))
- Instagram's custom 64-bit ID scheme allocates 41 bits to time, 13 bits to shard ID, and 10 bits to a per-shard sequence, generated via a PL/pgSQL function inside each shard (post summary (via search snippet))

### S28 · blog · Stripe — Designing robust and predictable APIs with idempotency (2017, stripe.com/blog)

https://stripe.com/blog/idempotency · accessed 2026-09-17

*Stripe's own explanation of idempotency keys, the payments-API pattern the request explicitly says the AI-billing module should reuse.*

_Covers: stripe-payments, billing-metering_

Facts:
- Stripe's mutating endpoints accept an Idempotency-Key header; a retried request with the same key returns the cached result of the first request, including cached 5xx errors (post body)
- Stripe's own client libraries retry automatically on failure using an idempotency key with exponential backoff and jitter (post body)
- A key that is currently in-flight is a distinct third state (neither absent nor complete); treating it as only two states turns a legitimate retry into a duplicate write (post body (via extraction))

### S29 · blog · Uber — H3: Uber's Hexagonal Hierarchical Spatial Index (2018, published 2018-06-27)

https://www.uber.com/en-EG/blog/h3/ · accessed 2026-09-17

*Uber's own explanation of H3, the geospatial indexing library it built and open-sourced for dispatch/pricing/ETA, and why hexagons beat squares/postal codes for this workload.*

_Covers: uber-dispatch_

Facts:
- H3 supports sixteen resolutions; each finer resolution cell has one-seventh the area of the coarser resolution (post body)
- The globe is tiled from an icosahedron with 122 base cells (10 per face) and 12 unavoidable pentagons at the icosahedron vertices, since perfect hexagonal tiling of a sphere is impossible (post body)
- H3 indexes are 64-bit integers (post body)
- Hexagons have only one distance between a cell centerpoint and each neighbor, unlike squares which have two distinct neighbor distances requiring multiple coefficient sets; postal codes and operator-drawn zones have unstable, arbitrary shapes unsuitable for analysis (post body)

### S30 · blog · Uber — Uber Engineering's Ringpop (2016, published 2016-02-04)

https://www.uber.com/en-ES/blog/ringpop-open-source-nodejs-library · accessed 2026-09-17

*Uber's own explanation of Ringpop, the SWIM-gossip/consistent-hashing library underlying application-layer sharding for its dispatch (DISCO) and geospatial services.*

_Covers: uber-dispatch, consensus-coordination_

Facts:
- Ringpop implements a SWIM gossip protocol variant over TCP, computes membership/ring checksums, and retains down members in its member list (post body)
- Nodes repeatedly ping each other so every node learns of every other node's existence and status, letting work be divided automatically without a centralized coordinator (post body)
- Ringpop's consistent-hash ring uses FarmHash as its hash function and a red-black tree for the ring structure, with a uniform number of replica points added per node (post body)
- Because consistent hashing minimizes reassignment on membership change, adding/removing a node does not require rebalancing all object data, which reduces latency (post body)

### S31 · blog · Airbnb (The Airbnb Tech Blog) — Building Services at Airbnb, Part 1 (2017, published 2017-12-12)

https://medium.com/airbnb-engineering/building-services-at-airbnb-part-1-c4c1d8fa811b · accessed 2026-09-17

*Airbnb's own account of migrating from a monolithic Rails service to SOA, including its Thrift-based Service IDL and the tooling built to make services fast to bootstrap -- the primary source for the monolith-to-microservices case study.*

_Covers: airbnb, microservices-api-design_

Facts:
- Airbnb explicitly frames the move from a monolithic Rails service to SOA as not without challenges, alongside building new products at the same time (post body)
- Airbnb built a Thrift-based Service IDL and a Thrift-over-HTTP protocol as its inter-service communication layer, rather than adopting an existing framework wholesale, because replacing the HTTP stack outright would have caused major disruption (post body)
- Several new Java services went from inception to production traffic in only three weeks using Airbnb's 'make-me-a-service' scaffolding tool, an estimated saving of 2-3 weeks of engineering time per service (post body)

### S32 · official · Google Search Central — In-Depth Guide to How Google Search Works (2026, live documentation)

https://developers.google.com/search/docs/fundamentals/how-search-works · accessed 2026-09-17

*Google's own official description of the crawl -> index -> serve pipeline; the base citation for the Google Search case study (architectural specifics beyond this are not publicly disclosed by Google, per notes).*

_Covers: google-search_

Facts:
- Google Search works in three stages: crawling (downloading text/images/video via automated crawlers), indexing (analyzing content and storing it in the Google index), and serving (returning relevant results to a query) (doc body)
- The Google Search index covers hundreds of billions of webpages and is well over 100,000,000 gigabytes in size (doc body)
- During crawling, Google renders the page and runs any JavaScript found, using a recent version of Chrome, similar to how a browser renders pages (doc body)
- Ranking relevancy is determined by hundreds of factors, and Google states explicitly that it does not accept payment to rank pages higher (doc body)
- Indexing is not guaranteed -- not every page Google crawls and processes is added to the index (doc body)

### S33 · talk · Twitter (talk given at QCon San Francisco) — Timelines at Scale (2013, QCon SF 2012, recorded/published 2013-04-03)

https://www.infoq.com/presentations/Twitter-Timeline-Scalability · accessed 2026-09-17

*The most-cited primary talk by a named Twitter VP of Engineering describing Twitter's fan-out-on-write timeline architecture; InfoQ's page carries only the abstract/metadata, not a transcript, so detailed figures are corroborated via S34.*

_Covers: twitter-timeline_

Facts:
- The talk's stated subject is the architecture Twitter used to handle thousands of events per second -- tweets, social graph mutations, and direct messages (InfoQ abstract)
- Speaker was Raffi Krikorian, Senior Director of Applications Services at Twitter, at QCon San Francisco (recorded/published April 2013) (InfoQ metadata)

### S34 · analysis · High Scalability (third-party analysis site) — The Architecture Twitter Uses to Deal with 150M Active Users, 300K QPS, a 22 MB/S Firehose, and Send Tweets in Under 5 Seconds (2013, published 2013-07-08)

https://highscalability.com/the-architecture-twitter-uses-to-deal-with-150m-active-users/ · accessed 2026-09-17

*Third-party analysis that transcribes and attributes specific numbers to Raffi Krikorian's 'Timelines at Scale' talk (S33); used here as analysis-tier corroboration, not as the sole source for any figure.*

_Covers: twitter-timeline_

Facts:
- At the time of the talk, Twitter had 150M worldwide active users and served 300K QPS for home timelines against an ingest rate of about 4,000-5,000 tweets/sec average (7K/sec peak, >12K/sec during large events) (article body, attributed to Krikorian's talk)
- Reads dominate writes by roughly 50x: 300K QPS spent reading timelines vs. only 6,000 requests/sec spent on writes (article body, attributed to Krikorian's talk)
- A home timeline is cached in a Redis cluster capped at 800 entries, replicated 3x across machines and across datacenters (article body, attributed to Krikorian's talk)
- On a tweet's arrival, a fanout daemon looks up the tweet author's followers and writes the tweet ID into each follower's Redis-cached timeline; Twitter targets under 5 seconds for this to reach followers, versus up to 5 minutes in the naive worst case for a celebrity with 31M followers (article body, attributed to Krikorian's talk)
- Named systems: Gizzard (abstracts SQL transactions, provides global replication) built on Flock (maintains follower/following lists), Timeline/Tweet/User/Social-Graph services, Gizmoduck (user object service), Tweetypie (tweet object service), Early Bird (modified Lucene for search), Zipkin (Dapper-style tracing) (article body, attributed to Krikorian's talk)

### S35 · official · OpenAI — Rate limits (OpenAI API) (2026, live documentation)

https://developers.openai.com/api/docs/guides/rate-limits · accessed 2026-09-17

*OpenAI's own definition of its rate-limit dimensions and usage tiers, the canonical reference for the platform rate-limits/quotas module.*

_Covers: rate-limits-quotas_

Facts:
- Rate limits are enforced independently across RPM, RPD, TPM, TPD, IPM, and audio minutes per minute; exceeding any one triggers a 429 (doc body)
- Usage tiers are keyed to cumulative dollars paid: Free (allowed geography), Tier 1 ($5 paid), Tier 2 ($50), Tier 3 ($100), Tier 4 ($250), Tier 5 ($1,000); monthly usage caps range from $100/month up to $200,000/month at Tier 5 (doc body)
- Vector stores are separately rate-limited at 300 requests/minute per store (doc body)
- Unsuccessful (429) requests still count against the per-minute limit, so blindly resending a request will not work; clients should honor Retry-After and add jitter (doc body)
- OpenAI recommends ramping traffic gradually: once traffic reaches 1M input TPM, increase no more than 50% every 15 minutes (doc body)

### S36 · official · Anthropic — Rate limits (Claude API) (2026, live documentation)

https://platform.claude.com/docs/en/api/rate-limits · accessed 2026-09-17

*Anthropic's own definition of rate limits vs. spend limits and its token-bucket, cache-aware ITPM enforcement -- directly relevant to the billing/rate-limit design module and its interaction with prompt caching.*

_Covers: rate-limits-quotas, prompt-caching-tiers, billing-metering_

Facts:
- Spend limits cap monthly cost; rate limits cap request rate over a period -- these are explicitly two different mechanisms with different error signatures (doc body)
- Rate limiting uses the token bucket algorithm: capacity is continuously replenished up to a maximum rather than reset at fixed intervals (doc body)
- Monthly spend caps by tier: Start $500, Build $1,000, Scale $200,000; Custom tier has no cap (doc body)
- For most models, only uncached input tokens (input_tokens + cache_creation_input_tokens) count toward the ITPM rate limit; cache_read_input_tokens do not, so caching effectively raises throughput -- e.g. a 2,000,000 ITPM limit with an 80% cache hit rate can process ~10,000,000 total input tokens/minute (doc body)
- OTPM is evaluated in real time on actual tokens generated; the max_tokens request parameter does not count against OTPM, so setting it high carries no rate-limit downside (doc body)
- Reaching the spend cap returns a 429 rate_limit_error with no retry-after header and an enforced_spend_limit_reached error_code, distinguishing it from an ordinary rate limit which always carries retry-after (doc body)
- Per-workspace rate limits can be set below the org-wide limit to protect other workspaces from overuse; unused workspace allocation is available to other workspaces, but the org-wide limit always applies even if workspace limits sum to more (doc body)

Conflicts: OpenAI (S35) enforces rate limits primarily via fixed-window-like per-minute/per-day counters across 6 independent dimensions (RPM/RPD/TPM/TPD/IPM/audio); Anthropic (S36) uses a continuously-replenishing token bucket across 3 dimensions (RPM/ITPM/OTPM) and explicitly excludes cached tokens from the input-token count -- two different philosophies for the same generate-now-know-cost-later problem.

### S37 · official · Anthropic — Prompt caching (Claude API) (2026, live documentation)

https://platform.claude.com/docs/en/build-with-claude/prompt-caching · accessed 2026-09-17

*Anthropic's own mechanism description for prefix caching, including breakpoint semantics, TTL economics, and lookback window -- the primary source for the platform prefix-caching module.*

_Covers: prompt-caching-tiers_

Facts:
- Pricing multipliers: 5-minute cache write = 1.25x base input; 1-hour cache write = 2x base input; cache read = 0.1x base input (0.025x on some newer models) (doc body)
- A cache write is created only at an explicit breakpoint (a hash of the prefix ending there); reads walk backward up to 20 positions looking for a matching prefix hash written by a prior request (doc body)
- Up to 4 explicit cache breakpoints are allowed per request; cache prefixes are built in order tools -> system -> messages, and a change at one level invalidates that level and everything after it (doc body)
- Default cache lifetime is 5 minutes, measured from the start of the write/read request, not the end of the response -- a 4-minute-long response leaves only about 1 minute for a follow-up request to hit the cache (doc body)
- For concurrent requests, a cache entry only becomes available after the first response begins, so parallel requests racing on a cold cache will all miss (doc body)
- Minimum cacheable prompt length varies from 512 tokens (some models) up to 4,096 tokens (others); shorter prompts marked cacheable are processed without caching and without error (doc body)

### S38 · official · OpenAI — Prompt caching (OpenAI API) (2026, live documentation)

https://developers.openai.com/api/docs/guides/prompt-caching · accessed 2026-09-17

*OpenAI's own mechanism description for prompt caching (automatic, prefix-based, no explicit breakpoints) -- lets a writer contrast OpenAI's automatic-caching design against Anthropic's explicit-breakpoint design (S37).*

_Covers: prompt-caching-tiers_

Facts:
- OpenAI's cache lookup walks the longest-to-shortest prefix of the incoming request looking for a matching prefix already resident on a serving machine, rather than requiring an explicit cache_control marker (doc body)
- Cache reads are discounted up to 90%; on GPT-5.6+ minimum cacheable length is 1,024 tokens, cache writes cost 1.25x base rate and reads cost 0.1x base rate (doc body)
- On GPT-5.6+, a cache persists 30 minutes after its most recent write or reuse; earlier models retain caches roughly 5-10 minutes up to one hour (in-memory) or up to 24 hours (24h explicit mode) (doc body)
- Manual cache clearing is not available, and cached input tokens still count toward tokens-per-minute rate limits -- an explicit contrast with Anthropic's exclusion of cached tokens from ITPM (S36) (doc body)

Conflicts: OpenAI (S38) counts cached input tokens toward its TPM rate limit; Anthropic (S36/S37) explicitly excludes cache_read_input_tokens from ITPM for most models, meaning heavy cache use raises effective throughput headroom on Anthropic but not on OpenAI.

### S39 · official · LiteLLM — Life of a Request (LiteLLM Proxy architecture) (2026, docs.litellm.ai live docs)

https://docs.litellm.ai/docs/proxy/architecture · accessed 2026-09-17

*LiteLLM's own architecture doc for its open-source LLM gateway/proxy -- the request explicitly names LiteLLM as a reference open-source gateway.*

_Covers: api-gateway-edge, billing-metering, request-routing-admission_

Facts:
- Request flow: virtual-key auth/budget check (cache lookup, DB fallback on miss) -> parallel-request rate limiter (rpm/tpm at global/key/user/team levels) -> LiteLLM Router (load balancing, fallbacks, retries) -> SDK translation to OpenAI format -> provider call (doc body)
- After the response returns to the client, spend logging, rate-limit accounting, and logging callbacks all run as asynchronous background tasks -- no database write sits in the request's critical path (doc body)
- Virtual keys support hard limits (tokens, requests, upstream-billing caps), soft limits with alerts, model whitelists, and per-key rate limits; spend is tracked per key, and if attached, rolls up to a user_id and team_id (doc body)

### S40 · official · Envoy AI Gateway (CNCF-adjacent, backed by Tetrate/Bloomberg/others) — Usage-based Rate Limiting (Envoy AI Gateway / Agent Router docs) (2026, v0.1 docs)

https://aigateway.envoyproxy.io/docs/0.1/capabilities/usage-based-ratelimiting/ · accessed 2026-09-17

*Envoy AI Gateway's own description of token-based rate limiting layered on Envoy's Global Rate Limit API -- the request explicitly names Envoy AI Gateway as a reference gateway.*

_Covers: rate-limits-quotas, request-routing-admission_

Facts:
- The gateway automatically extracts token usage (input/output/total) from LLM responses that follow the OpenAI schema format, including via response transformers for non-OpenAI backends like AWS Bedrock (doc body)
- Enforcement is two-phase: on request arrival, the gateway checks whether processing would exceed the configured token limit and rejects with 429 if so; otherwise it processes the request and only then counts the resulting token usage toward the total (doc body)
- Custom cost metrics can be computed with CEL expressions, e.g. to weight cached tokens differently from live tokens (doc body)
- Example config limits: 1,000 total tokens/hour/user for GPT-4, 5,000 total tokens/hour/user for GPT-3.5-turbo (doc example config)

Conflicts: Envoy AI Gateway (S40) admits a request before knowing its true token cost, then reconciles after the fact -- an optimistic-admission design distinct from Anthropic's estimate-then-adjust-during-the-request approach (S36).

### S41 · official · Kubernetes SIGs — Introduction (Kubernetes Gateway API Inference Extension) (2026, live docs)

https://gateway-api-inference-extension.sigs.k8s.io/ · accessed 2026-09-17

*Official docs for the Kubernetes Gateway API Inference Extension, the project explicitly named in the request as a reference open-source gateway for inference-aware routing.*

_Covers: request-routing-admission, api-gateway-edge, gpu-backend-handoff_

Facts:
- An InferencePool represents a set of endpoints running a model-server framework; the Endpoint Picker (EPP) is the Inference Router implementation that fetches metrics from pool endpoints to pick the one that best meets configured objectives (doc body)
- The architecture reuses Envoy's External Processing (ext-proc) protocol to extend any ext-proc-and-Gateway-API-capable gateway into an inference-aware gateway, rather than building a bespoke gateway (doc body)
- Model-server-aware, metrics-driven load balancing is claimed to reduce serving latency and improve accelerator utilization versus generic L7 balancing (doc body)

### S42 · blog · Kubernetes (kubernetes.io/blog) — Introducing Gateway API Inference Extension (2025, published 2025-06-05)

https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/ · accessed 2026-09-17

*The Kubernetes project's own announcement explaining why LLM inference traffic breaks generic HTTP load balancing.*

_Covers: request-routing-admission, api-gateway-edge_

Facts:
- LLM inference sessions are long-running, resource-intensive, and partially stateful -- a single GPU-backed model server may keep multiple sessions active with in-memory token caches, unlike typical short-lived stateless web requests (post body)
- Traditional load balancers that route on HTTP path or round-robin lack model-identity and request-criticality awareness (e.g. interactive chat vs. batch), which the Inference Extension adds as first-class routing signals (post body)
- The named routing signals introduced are KV-cache utilization and queue depth on candidate model servers (post title/body)

### S43 · official · Cloudflare — Rate limiting (Cloudflare AI Gateway) (2026, live documentation)

https://developers.cloudflare.com/ai-gateway/features/rate-limiting/ · accessed 2026-09-17

*Cloudflare's own description of fixed vs. sliding window rate limiting at the AI-gateway layer -- the request explicitly names Cloudflare as a reference AI gateway.*

_Covers: rate-limits-quotas_

Facts:
- Fixed window resets at time boundaries (e.g. 12:00-12:10), so 10 requests at 12:09 plus 10 at 12:11 both succeed (20 total) even against a 10-per-10-minutes limit (doc body)
- Sliding window evaluates the trailing time interval continuously, so the same 20-request burst across the 12:09/12:11 boundary would fail under a 10-per-10-minutes sliding limit (doc body)
- Exceeding the configured rate returns HTTP 429 Too Many Requests and the request is not processed (doc body)

Conflicts: Cloudflare (S43) documents fixed-window rate limiting as materially weaker than sliding-window (allowing 2x burst at a window boundary) using the same 10-per-10-minute example -- a concrete, quantified version of the classic fixed-vs-sliding-window trade-off.

### S44 · official · OpenAI — Moderation (OpenAI API) (2026, live documentation)

https://developers.openai.com/api/docs/guides/moderation · accessed 2026-09-17

*OpenAI's own moderation-classifier taxonomy and integration guidance -- the primary source for the safety/moderation pipeline module's classifier design.*

_Covers: safety-moderation_

Facts:
- The moderation endpoint classifies against 13 categories: harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic (doc body)
- The response returns a boolean flagged, per-category booleans, per-category confidence scores 0-1, and which input types (text/image) each category applied to (doc body)
- OpenAI explicitly recommends treating moderation scores as policy signals rather than an automatic blocking decision, and reviewing results before showing generated output to a user or acting on it (doc body)

### S45 · blog · Anthropic — Building safeguards for Claude (2025, anthropic.com/news)

https://www.anthropic.com/news/building-safeguards-for-claude · accessed 2026-09-17

*Anthropic's own description of its safety pipeline across policy, training, pre-deployment testing, and real-time enforcement -- lets a writer contrast a training-time-integrated safety design against OpenAI's classifier-only moderation API (S44).*

_Covers: safety-moderation_

Facts:
- The pipeline has four stages: policy development (Usage Policy, Unified Harm Framework, Policy Vulnerability Testing with external experts), training integration (fine-tuning teams shape model refusal behavior), pre-deployment testing (safety/risk/bias evaluations), and real-time detection/enforcement (post body)
- Classifiers are prompted or specially fine-tuned Claude models used to detect policy violations in real time (post body)
- Pre-deployment risk assessments explicitly cover CBRN (chemical, biological, radiological, nuclear) and cyber-harm domains as a distinct high-risk category from general safety evaluations (post body)

Conflicts: OpenAI (S44) ships moderation as a separate scoring endpoint applied around generation and explicitly recommends against automatic blocking; Anthropic (S45) embeds safety classifiers throughout training and real-time enforcement as one continuous pipeline rather than a bolt-on filter.

### S46 · official · Stripe — Usage-based billing / Recording usage for billing (Stripe) (2026, live documentation)

https://docs.stripe.com/billing/usage-based · accessed 2026-09-17

*Stripe's own architecture for usage-based billing (meters, meter events, async aggregation) -- the metering/billing pattern the request says the AI-billing module should reuse.*

_Covers: billing-metering_

Facts:
- Metronome (a Stripe product) is the recommended path for new usage-based billing integrations, providing real-time metering, flexible pricing, and continuous running-balance tracking rather than invoice-time-only totals (doc body)
- The legacy Billing Meters path sends meter events to a Stripe endpoint and Stripe aggregates them at the end of the billing period; Stripe explicitly processes meter events asynchronously, so aggregated usage summaries may lag recently received events (doc body)
- Metronome is positioned as more suitable than Billing Meters for products where customers generate large numbers of events per second or per day (doc body)

### S47 · blog · Fireworks AI (claims as reported by a third party) — Fireworks AI Deep Dive: Inference at Scale (third-party summary of Fireworks' own claims) (2026, n/a)

https://www.gmicloud.ai/en/blog/fireworks-ai-inference-at-scale · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*The request names Fireworks AI as a reference inference platform; a direct Fireworks AI engineering blog post with this level of detail could not be located and fetched in the time budget, so these figures are Fireworks' own public claims as relayed by a third party and should be re-verified against fireworks.ai's own blog before being cited as fact in a module.*

_Covers: multi-tenancy-isolation, capacity-cost-management_

Facts:
- Fireworks reportedly runs dedicated GPU clusters per model/customer tier rather than sharing GPU resources across customers, to avoid noisy-neighbor variance (summary (unverified against Fireworks' own blog))
- Fireworks reportedly processes 200,000 queries/second and 10+ trillion tokens/day at peak, across tens of GPU clouds, using a custom inference engine (FireAttention) with continuous batching, paged attention, and prefill disaggregation (summary (unverified against Fireworks' own blog))

Conflicts: If accurate, Fireworks' claimed per-customer dedicated-cluster model (isolation-first) is the opposite of the shared-pool-plus-virtual-key model that gateways like LiteLLM (S39) and Envoy AI Gateway (S40) are built around (utilization-first); this is a real fork in multi-tenancy philosophy worth verifying against a primary Fireworks source.

### S48 · official · Anthropic — Code execution tool (Claude API) (2026, tool versions code_execution_20250825 / 20260120 / 20260521)

https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool · accessed 2026-09-17

*Anthropic's own specification of its hosted code-execution sandbox: resource limits, container lifecycle, networking posture, and pricing -- the most detailed public spec of an agent-harness sandbox from a frontier lab.*

_Covers: agent-harness-sandboxes_

Facts:
- Each container runs Python 3.11 on a Linux x86_64 sandbox with 5 GiB RAM, 5 GiB disk, and 1 CPU (Containers / Runtime environment)
- Internet access is completely disabled inside the container; no outbound network requests are permitted, so only pre-installed libraries (pandas, numpy, scipy, matplotlib, etc.) are available -- packages cannot be installed at runtime (Containers / Networking and security)
- Containers expire 30 days after creation; after about 5 minutes of inactivity a container is checkpointed and can be restored by sending its ID again within the 30-day window; an expired container returns an error and cannot be restored (Container reuse)
- Without web search/fetch, code execution is billed by execution time (minimum 5 minutes per invocation), with 1,550 free hours/month per organization and $0.05/hour per container beyond that; execution time bills even if the tool isn't called when files are attached, since files preload onto the container (Usage and pricing)
- Code execution is free (no execution-time charge) when used together with the web_search or web_fetch tools, beyond standard token costs (Usage and pricing)
- A pause_turn stop reason signals the API paused a long-running turn; the caller can resubmit the response as-is to let Claude continue, or modify content to interrupt (Errors / pause_turn stop reason)
- Container data (execution artifacts, uploaded files, outputs) is retained up to 30 days server-side; files pushed through the Files API persist until explicitly deleted (Data retention)

### S49 · blog · OpenAI — Unlocking the Codex harness: how we built the App Server (2026, openai.com/index)

https://openai.com/index/unlocking-the-codex-harness/ · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*OpenAI's own architectural account of the Codex agent harness (threads, sandboxing, MCP tool wiring) -- the primary requested source for the agent-harness-backends module; direct WebFetch returned HTTP 403 in this session, so facts below are drawn from the WebSearch tool's cached summary rather than the full page text.*

_Covers: agent-harness-sandboxes_

Facts:
- The Codex App Server owns the agent loop and sandboxed execution; the harness is defined as the surrounding execution system that manages conversation state, streamed activity, and tool interaction, while applications own product context, business rules, and tools (post summary (via search snippet))
- A thread is a Codex conversation between a user and an agent; Codex creates, resumes, forks, and archives threads and persists their event history (post summary (via search snippet))
- A useful harness must preserve/resume long-running work, decide what context to retain or compact, discover and validate tools, execute inside a sandbox, stream observable events, pause for approval, recover after interruption, and verify outcomes -- stated as the harness's core responsibility list (post summary (via search snippet))

### S50 · official · OpenAI — Streaming API responses (OpenAI API) (2026, live documentation)

https://developers.openai.com/api/docs/guides/streaming-responses · accessed 2026-09-17

*OpenAI's own SSE event taxonomy and an explicit stated limitation on moderating streamed output -- grounds the streaming-protocol module.*

_Covers: streaming-protocols_

Facts:
- The Responses API streams typed lifecycle events over SSE: response.created, response.in_progress, response.output_text.delta, response.completed, response.failed, plus per-item events for function calls, code interpreter, and file search (doc body)
- Streaming output makes moderation harder because partial completions are harder to evaluate than a full response; if moderation scores are requested, they only arrive after the full output is generated and are not included with partial deltas (doc body)

### S51 · official · OpenAI — Background mode (OpenAI API) (2026, live documentation)

https://developers.openai.com/api/docs/guides/background · accessed 2026-09-17

*OpenAI's own mechanism for stream resumption after a client disconnect -- directly grounds the request's 'resumption' requirement in the streaming module.*

_Covers: streaming-protocols_

Facts:
- A response created with both background:true and stream:true can be resumed after a disconnect: the client tracks the sequence_number from each streamed event and reconnects with GET /v1/responses/{id}?stream=true&starting_after={sequence_number} (doc body)
- Resumption is only possible if the background response was originally created with stream=true (doc body)
- Response data is temporarily stored to disk for roughly 10 minutes to support asynchronous execution and polling (doc body)

### S52 · blog · Anthropic — Clio: Privacy-preserving insights into real-world AI use (2024, anthropic.com/research)

https://www.anthropic.com/research/clio · accessed 2026-09-17

*Anthropic's own account of Clio, the production system it uses to analyze real conversation data for safety/observability while preserving privacy -- directly grounds the data-privacy-retention and LLM-observability modules' treatment of conversation-log analysis.*

_Covers: data-privacy-retention, llm-observability_

Facts:
- Clio was used to analyze 1 million claude.ai conversations across Free and Pro tiers; web/mobile app development was the top category at over 10% of conversations, education over 7%, business strategy nearly 6% (post body)
- Clio's pipeline has four stages: extracting facets (topic/language metadata), semantic clustering by theme, cluster description with private information excluded, and building hierarchies for exploration; a minimum user/conversation threshold prevents small clusters (which might identify individuals) from surfacing (post body)
- Claude itself verifies that cluster summaries do not contain overly specific or identifying information before they are shown to human analysts, and only these higher-level clusters -- not raw conversations -- are visible to analysts (post body)
- Clio's classification correlated with the separate Trust and Safety classification system at r = 0.71, and was used operationally to find false positives/negatives in that safety system and to detect a coordinated spam network using similar prompt structures (post body)

### S53 · paper · Meta — The Llama 3 Herd of Models (2024, arXiv:2407.21783)

https://arxiv.org/pdf/2407.21783 · accessed 2026-09-17

*Meta's own paper describing the hardware, storage, network topology, reliability data, and post-training pipeline that powered Llama 3 405B pre-training -- the single most detailed public account of frontier-scale training infrastructure, explicitly named as required grounding in the request.*

_Covers: checkpointing-storage-cluster, cluster-scheduling-orchestration, fault-tolerance-elasticity, data-pipelines-training, post-training-pipelines_

Facts:
- Llama 3 405B trains on up to 16K H100 GPUs (700W TDP, 80GB HBM3 each) on Meta's Grand Teton platform, 8 GPUs + 2 CPUs per server, connected via NVLink within a server; jobs are scheduled with MAST, Meta's global-scale training scheduler (§3.3.1)
- Storage runs on Tectonic, Meta's general-purpose distributed file system, built on 7,500 SSD-equipped servers offering 240PB of storage, 2TB/s sustainable throughput and 7TB/s peak; per-GPU checkpoint state ranges 1MB-4GB and must be written during highly bursty checkpoint writes that saturate the storage fabric for short durations (§3.3.1)
- The RoCE-based 24K-GPU cluster uses a three-layer Clos network: 16 GPUs per rack (2 servers) behind a single ToR switch; 192 racks form a pod of 3,072 GPUs with full bisection bandwidth; 8 pods form the 24K-GPU cluster via aggregation switches at a 1:7 oversubscription ratio (§3.3.1)
- 4D parallelism (tensor, pipeline, context, data/FSDP) is ordered [TP, CP, PP, DP] from innermost (highest bandwidth need) to outermost (most latency-tolerant); achieved BF16 MFU is 38-43% across the 8K/16K-GPU pre-training configurations (§3.3.2, Table 4)
- Over a 54-day pre-training snapshot, there were 466 job interruptions (47 planned, 419 unexpected); ~78% of unexpected interruptions were attributed to confirmed or suspected hardware issues, with faulty GPUs the single largest category at 30.1% and GPU HBM3 memory failures at 17.2% (§3.3.4, Table 5)
- Despite 16K-GPU-scale failure rates, Meta achieved higher than 90% effective training time (time spent on useful training over elapsed time), at a cost of at least one training interruption per day (§3.3.4)
- Web data curation applies URL-level, document-level (global MinHash), and aggressive line-level (ccNet-style, removing lines repeated >6 times per 30M-document bucket) de-duplication, plus fasttext/RoBERTa-based quality classifiers and a 176-language fasttext language-ID model for multilingual data (§3.1.1)
- Post-training is 6 rounds of SFT -> reward modeling -> rejection sampling -> DPO; PagedAttention is adopted for rejection sampling and gives over 2x throughput improvement there (§4.1, §4.2.2)

### S54 · paper · ByteDance / Peking University — MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs (2024, USENIX NSDI '24)

https://www.usenix.org/system/files/nsdi24-jiang-ziheng.pdf · accessed 2026-09-17

*ByteDance's production system paper for training LLMs beyond 10,000 GPUs -- the second of the two named 'cluster papers' (with Philly) grounding the training-infrastructure fault-tolerance and observability modules.*

_Covers: fault-tolerance-elasticity, cluster-scheduling-orchestration, checkpointing-storage-cluster_

Facts:
- MegaScale achieves 55.2% Model FLOPs Utilization (MFU) training a 175B-parameter model on 12,288 GPUs, a 1.34x improvement over Megatron-LM (Abstract)
- The design principle is full-stack algorithm-system co-design across model block, optimizer, computation/communication overlap, operator optimization, data pipeline, and network performance, because failures and stragglers are the norm rather than the exception at this scale (Abstract, §1)
- MegaScale's largest production AI cluster exceeds 10,000 GPUs; over the years of operation it has repaired and recovered the training process for a production run over 100 times in the presence of failures while the loss continued to converge (§1)
- The reliability toolset includes heartbeat messages for real-time anomaly detection, optimized checkpointing/recovery to reduce interruptions, and a 3D parallel-training visualization tool that shows data dependencies between ranks for diagnosing stragglers (§1)

### S55 · paper · Microsoft Research — Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads (Philly) (2019, USENIX ATC '19)

https://www.usenix.org/conference/atc19/presentation/jeon · accessed 2026-09-17 · **not fetched directly — see authority/notes**

*Microsoft's Philly cluster-manager trace study, explicitly named in the request as one of the three cluster papers (with Borg and MegaScale) that ground GPU-cluster scheduling design; could not be fetched as a full PDF within the session's time budget, so no verbatim facts are recorded here -- a writer should fetch https://www.usenix.org/system/files/atc19-jeon.pdf directly before citing it.*

_Covers: cluster-scheduling-orchestration_

_No facts extracted — see authority note above._

### S56 · paper · ByteDance / University of Hong Kong — HybridFlow: A Flexible and Efficient RLHF Framework (2024, arXiv:2409.19256, EuroSys '25)

https://arxiv.org/pdf/2409.19256 · accessed 2026-09-17

*The academic paper behind verl (the open-source name for HybridFlow) -- the primary architectural justification for verl's hybrid single-controller/multi-controller design, one of the six RL frameworks the request asks to compare.*

_Covers: rl-training-infra, rl-frameworks-comparison_

Facts:
- Existing RLHF frameworks force a choice between a single-controller paradigm (flexible but inefficient due to dispatch overhead to billion-parameter distributed models) and a multi-controller paradigm (efficient but inflexible, since changing one node's dataflow requires changing all dependent nodes) (§1)
- HybridFlow uses single-controller coordination between nodes (for flexible inter-node data resharding) and multi-controller execution within a node (for efficient intra-node computation) (§1)
- The 3D-HybridEngine reshards the actor model between training and generation phases with zero memory redundancy and significantly reduced communication overhead (§1)
- HybridFlow demonstrates 1.53x to 20.57x throughput improvement running various RLHF algorithms compared to state-of-the-art baselines (Abstract)
- The RLHF dataflow is modeled as a DAG where each node is computation of a neural network (actor/critic/reference/reward) and each edge is a data dependency, frequently a many-to-many multicast resharding operation between differently-parallelized models (§2.1)

### S57 · repo · verl-project — volcengine/verl README (2026, main branch)

https://github.com/volcengine/verl · accessed 2026-09-17

*Primary repository README for verl, the open-source implementation of HybridFlow and one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison_

Facts:
- verl's 3D-HybridEngine eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases (README)
- Supported RL algorithms include PPO, GRPO, DAPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, and DrGPO (README)
- verl integrates vLLM, SGLang, and HF Transformers for generation, and FSDP/FSDP2/Megatron-LM for training, via modular APIs (README)
- verl reports roughly a 1.4x speedup compared to its own previous version (v0.3.0.post1) (README changelog)

### S58 · repo · OpenRLHF — OpenRLHF README (2026, main branch)

https://github.com/OpenRLHF/OpenRLHF · accessed 2026-09-17

*Primary repository README for OpenRLHF, described by its own maintainers as the first RLHF framework built on a Ray + vLLM distributed architecture; one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison_

Facts:
- OpenRLHF uses Ray to separate the Actor, Reward, Reference, and Critic models across different GPUs, enabling scalable training for models up to 70B+ parameters (README)
- RLHF training spends 80% of its time on sample generation; OpenRLHF uses vLLM with Auto Tensor Parallelism and Pipeline Parallelism for that generation step (README)
- Training uses DeepSpeed ZeRO-3, deepcompile, AutoTP, and RingAttention, enabling large-model training without a heavyweight framework while working directly with HuggingFace models (README)
- OpenRLHF supports PPO, REINFORCE++, REINFORCE++-baseline, GRPO, RLOO, Dr.GRPO, and FlashREINFORCE, decoupled from single-turn vs. multi-turn agent execution mode (README)
- A Hybrid Engine Scheduling design lets all models and vLLM engines share GPU resources, minimizing idle time and maximizing utilization (README)

### S59 · repo · NovaSky-AI — SkyRL README (2026, main branch)

https://github.com/NovaSky-AI/SkyRL · accessed 2026-09-17

*Primary repository README for SkyRL, a modular full-stack RL training library; one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison_

Facts:
- skyrl-train is described as a modular, performant training framework for RL, emphasizing modularity throughout rather than a monolithic trainer (README)
- SkyRL-SQL-7B, a model the team trained on just 653 samples, is reported to outperform both GPT-4o and o4-mini on its target task (README)

### S60 · repo · inclusionAI (Ant Group-affiliated) — AReaL README (2026, main branch, AReaL 2.0)

https://github.com/inclusionAI/AReaL · accessed 2026-09-17

*Primary repository README for AReaL, a fully asynchronous RL training system; one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison, rl-training-infra_

Facts:
- AReaL 2.0 refactored into a microservice architecture with independent training, inference, agent, and weight-update services -- a fully asynchronous RL paradigm that completely decouples generation from training (README)
- Staleness is controlled via a max_head_offpolicyness parameter; setting it to 0 recovers a synchronous algorithm (README / docs)
- AReaL reports a 2.77x speedup (boba^2 configuration) over synchronous systems with comparable or superior training performance; AReaL-lite achieves 90% of AReaL's performance with 80% fewer lines of code (README)
- AReaL supports 16 algorithms including GRPO, GSPO, PPO, DAPO, LitePPO, Dr.GRPO, REINFORCE++, RLOO, SAPO, IcePop, KPop, M2PO, DPO, reward modeling, SFT, and distillation (README)

### S61 · repo · Zhipu AI / THUDM — slime README (2026, main branch)

https://github.com/THUDM/slime · accessed 2026-09-17

*Primary repository README for slime, an SGLang-native post-training framework for RL scaling and the RL framework behind the GLM model family; one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison, rl-training-infra_

Facts:
- slime has three components: a training module (reads completed rollouts from the buffer, pushes updated weights after training), a rollout module (generates data via SGLang, optionally wrapped with multi-turn/tool-call/sandbox logic), and a data buffer bridging the two (README)
- slime integrates Megatron for training and SGLang for rollout/inference, passing SGLang arguments through directly with a --sglang- prefix so upstream serving optimizations remain available without a wrapper layer (README)
- slime is the RL framework behind the GLM-5.x, GLM-4.7, GLM-4.6, and GLM-4.5 model releases, and also supports Qwen, DeepSeek-V3, and Llama 3 (README)

### S62 · repo · NVIDIA — NVIDIA-NeMo/RL README (2026, main branch)

https://github.com/NVIDIA-NeMo/RL · accessed 2026-09-17

*Primary repository README for NeMo-RL, NVIDIA's post-training/RL library built on Ray; one of the six RL frameworks the request names for comparison.*

_Covers: rl-frameworks-comparison, cluster-scheduling-orchestration_

Facts:
- NeMo RL provides resource management via Ray for scalable, flexible deployment across hardware configurations, supporting both small experiments and massive multi-GPU, multi-node deployments (README)
- Two training backends: DTensor (PyTorch-native TP/SP/PP/CP/FSDP2) for hackable research prototypes, and Megatron (NVIDIA's 6D-parallelism framework) for maximum scale (README)
- Supported algorithms include GRPO/GSPO/DAPO, SFT (with LoRA), DPO, and on-policy distillation; the framework emphasizes environment isolation with process isolation between RL Actors (README)

### S63 · blog · Hugging Face — Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries (2026, hf.co/blog)

https://huggingface.co/blog/async-rl-training-landscape · accessed 2026-09-17

*The single most direct comparison available of trainer/rollout decoupling strategies across the RL-infra ecosystem, explicitly surveying verl, NeMo-RL, SkyRL, AReaL, and slime (four of the request's six named frameworks) side by side plus 11 others.*

_Covers: rl-frameworks-comparison, rl-training-infra_

Facts:
- 16 frameworks surveyed: AReaL, ART, Atropos, MILES, NeMo-RL, OAT, open-instruct, PipelineRL, PRIME-RL, ROLL, SkyRL, SLIME, TorchForge, Tunix, verl, verifiers-rl (post body)
- The dominant architecture disaggregates inference and training GPUs into two pools connected by a weight-synchronization protocol: an inference pool running vLLM/SGLang continuously and a training pool running the optimizer continuously (post body)
- Staleness-management strategies fall into three orthogonal categories: hard version-rejection of samples too far behind current policy, depth-bounding via a fixed-capacity queue between generation and training, and importance-sampling correction (often clipped/truncated) for stale samples that do get used (post body)
- 8 of the 16 surveyed frameworks (verl, SkyRL, NeMo-RL, SLIME, MILES, ROLL, OAT, open-instruct) use Ray for orchestration; verifiers-rl, PipelineRL, ART, and AReaL instead use native Python asyncio/threading; Atropos uses HTTP microservices; TorchForge uses the Monarch actor model (post body)
- Weight-sync latency ranges from ~100-500ms for standard NCCL broadcast down to ~20ms with bucketing strategies; PipelineRL is unique in swapping weights between forward passes with only a ~1-10ms gap, never interrupting an in-flight sequence (post body)
- For a 32B model generating 32K tokens/rollout across 512 sequences on a single H100 at ~1.2K tok/s, generation alone takes about 3.7 hours -- the concrete numeric illustration of why the generation step, not training, is the bottleneck that motivates async RL infra (post body)

Conflicts: The survey (S63) documents three genuinely different staleness-tolerance philosophies in production RL frameworks (hard rejection, bounded depth, importance-sampling correction) rather than one converged answer, and a further fork between Ray-based (8/16) and native-async (4/16) orchestration -- both are live, unsettled design choices rather than solved problems.

### S64 · official · Ray / Anyscale — Ray Core Walkthrough (2026, docs.ray.io live docs)

https://docs.ray.io/en/latest/ray-core/walkthrough.html · accessed 2026-09-17

*Ray's own description of its core primitives (tasks, actors, object store) -- Ray is the orchestration substrate underneath 8 of the 16 RL frameworks surveyed in S63 and is explicitly named in the request.*

_Covers: cluster-scheduling-orchestration, rl-frameworks-comparison_

Facts:
- Ray solves distributed Python scaling with a small set of essential primitives -- tasks, actors, and objects -- for building and scaling distributed applications (doc body)
- Ray actors are stateful workers that preserve state between method calls, and an actor executes its method calls serially in the order received, preserving consistency (doc body)
- Ray's distributed object store manages data across the cluster via automatic storage of task/actor return values, explicit placement via ray.put(), and passing references between components to avoid unnecessary data copying (doc body)

### S65 · official · SchedMD (Slurm) — Slurm Workload Manager Overview (2026, live documentation)

https://slurm.schedmd.com/overview.html · accessed 2026-09-17

*SchedMD's own architectural description of Slurm, the HPC scheduler explicitly named in the request as a comparison point against Kubernetes/Ray for training-cluster orchestration.*

_Covers: cluster-scheduling-orchestration_

Facts:
- Slurm's three key functions are: allocating exclusive/non-exclusive access to compute nodes for a duration; providing a framework to start, execute, and monitor parallel work on allocated nodes; and arbitrating resource contention via a pending-work queue (doc body)
- Architecture is centralized: slurmctld runs on a management node (with optional failover backup) and monitors resources/work; each compute node runs a slurmd daemon that waits for, executes, and reports on work, like a remote shell (doc body)
- An optional database daemon stores accounting data across multiple clusters, and an optional REST API daemon enables remote interaction with Slurm over standard HTTP (doc body)

Conflicts: Slurm (S65) centralizes scheduling in a single slurmctld with heartbeat-based node reporting, a much simpler control plane than Kubernetes' declarative reconciliation model or Ray's actor-based scheduling (S64) -- the classic HPC-batch-scheduler vs. cloud-native-orchestrator trade-off that training-infra teams have to choose between.

### S66 · blog · Prime Intellect — Environments Hub: A Community Hub To Scale RL To Open AGI (2026, primeintellect.ai/blog)

https://www.primeintellect.ai/blog/environments · accessed 2026-09-17

*Prime Intellect's own announcement of the Environments Hub and the verifiers spec -- directly grounds the 'environments as services' requirement for RL training infrastructure.*

_Covers: rl-training-infra_

Facts:
- Environments are defined as declaring the world, rules, and feedback loop of state, action, and reward, packaged so effort can focus on task-specific components (datasets, tools/harnesses, reward functions) while reusing shared infrastructure for evaluation or RL training (post body)
- Environments are natively supported in Prime Intellect's scalable prime-rl trainer, and sandboxes plug directly into Verifier Environments for secure code execution (post body)
- In a private beta the week before launch, over 30 researchers and companies contributed environments to the Hub (post body)

## Notes

SCOPE: this map covers a 4-part, 32-40-module course spanning classical distributed systems, 11 company case studies, a full AI-inference-platform stack, and AI-training/RL infrastructure -- four largely disjoint literatures. 66 sources were kept (above the course guide's 30-60 band) because each of the four parts is close to its own deep-dive in scope; breadth was prioritized over trimming once every kept source carried real, distinct facts. FETCH FAILURES: netflixtechblog.com and instagram-engineering.com sit behind Cloudflare/anti-bot protection that blocked direct WebFetch and a browser-UA curl; a reader-proxy (r.jina.ai) recovered full text for 3 of 5 attempted Netflix posts (S20, S23) but not for the per-title-encoding post (S21) or the Instagram sharding post (S27), and openai.com/index/unlocking-the-codex-harness (S49) returned HTTP 403 on every attempt -- all three are marked fetched:false with facts drawn from WebSearch's cached summaries only; a writer with browser access should re-fetch and verify before treating those quotes as exact. GMAIL has no dedicated engineering source: Google has not published a Gmail-specific architecture account comparable to the GFS/Bigtable/Spanner papers; the only grounding is Gmail's appearance as a Megastore consumer (S4) and as a Borg prod workload example (S9), so any Gmail module must say this plainly and lean on the Spanner/Bigtable lineage rather than invent Gmail-specific mechanics. PHILLY (S55) could not be fetched as a full PDF in the session's time budget -- URL is recorded but carries zero facts; fetch https://www.usenix.org/system/files/atc19-jeon.pdf before citing it. DDIA (S12) is a textbook whose O'Reilly table-of-contents page returned 403; chapter numbers given for the 2nd edition (Kleppmann & Riccomini, Feb 2026) are inferred from the well-known 1st-edition structure and should be confirmed against a physical/preview copy before a writer cites a specific chapter number. TWITTER (S33/S34): the primary talk (InfoQ listing) carries only an abstract; all the specific numbers are corroborated only through a third-party analysis (High Scalability) that itself attributes every figure to the talk -- treat detailed Twitter numbers as one step removed from the primary source. FIREWORKS (S47) is the weakest source in the set: no Fireworks-authored engineering blog post at this level of technical detail could be located in the time budget, so the figures are third-party-relayed claims about Fireworks, not independently confirmed -- flag this explicitly if a module leans on it, or replace it with a Baseten/Together/Modal/Anyscale primary source if a writer can locate one. INCIDENT-PATTERNS has no dedicated source at all (coverage: none) -- the planner should either scope an incident-patterns module narrowly around what the Llama 3 (S53) and MegaScale (S54) reliability sections already document (interruption root-cause taxonomies, heartbeat/checkpoint recovery) or fold it into the fault-tolerance-elasticity module rather than schedule it standalone. FRONTIER-SYNTHESIS (Part E) is marked thin/no-dedicated-source by design: it is meant to be built from the 9 cross-cutting disagreements above plus each part's fastest-moving edges (DDIA's Feb-2026 2nd edition, the RL-framework fork documented in S63, the still-unsettled multi-tenancy philosophy in S39/S40/S47) rather than from new primary sources -- the planner should still schedule it.
