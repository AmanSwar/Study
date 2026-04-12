# MODULE 28 — RLHF & Alignment Systems

## 1. Introduction & Learning Objectives

Reinforcement Learning from Human Feedback (RLHF) represents the final critical stage in large language model development, transforming capable base models into safe, aligned, and user-friendly assistants. RLHF requires simultaneous optimization of four distinct models (SFT model, reward model, policy model, reference model) across complex training loops, making it a systems-intensive endeavor. This module analyzes RLHF infrastructure comprehensively, covering the classical PPO-based pipeline and modern direct preference optimization (DPO) approaches that eliminate reinforcement learning complexity while maintaining alignment quality.

**Learning Objectives:**
- Understand RLHF pipeline architecture (SFT → RM → PPO phases)
- Analyze memory and computational requirements across four simultaneous models
- Master PPO (Proximal Policy Optimization) implementation details
- Comprehend DPO (Direct Preference Optimization) mathematics and advantages
- Understand GRPO (Group Relative Policy Optimization) for efficiency
- Implement reward model inference at production scale
- Design and evaluate alignment testing frameworks

## 2. RLHF Pipeline Overview

### 2.1 Classical RLHF Architecture

RLHF follows a multi-stage pipeline, each optimizing different aspects of model behavior:

**Stage 1: Supervised Fine-Tuning (SFT)**
```
Input: Base model (7B parameters)
Process:
- Fine-tune on high-quality instruction-following examples
- Examples: {question, expert_response}
- Standard supervised learning with cross-entropy loss

Output: SFT model (foundation for reward model training)
```

**Stage 2: Reward Model (RM) Training**
```
Input: SFT model, preference pairs {response_A, response_B}
Process:
- Train classifier: P(preferred | response)
- Binary classification on preference data
- Examples: pairs of SFT model outputs, human-labeled preferences

Output: Reward model (scorer for response quality)
```

**Stage 3: PPO (Proximal Policy Optimization)**
```
Input: SFT model, reward model
Process:
- Policy: SFT model (generates responses)
- Reward function: Trained reward model
- Optimize policy via PPO to maximize reward signal
- Maintain KL divergence from reference model (prevent drift)

Output: RLHF-aligned model
```

**Pipeline Diagram:**
```
Base Model (7B)
    ↓
Stage 1: SFT fine-tuning on high-quality data
    ↓
SFT Model
    ↓↘ (used as initialization)
Stage 2: RM training on preference pairs
    ↓ (scores responses)
Reward Model
    ↓
Stage 3: PPO policy optimization
    ↓ (policy generates responses)
Policy Model (RLHF-aligned)
    ↓
Final: Instruction-following, aligned model
```

### 2.2 RLHF Infrastructure Complexity

RLHF simultaneously manages four distinct models during policy training:

**Four Models in PPO Training Loop:**

1. **Policy Model (πθ)**: Actively trained
   - Generates responses for current prompts
   - Updated via PPO gradients
   - Parameters: P (same as base model size)

2. **Reward Model (R)**: Fixed during PPO
   - Scores generated responses
   - Determines optimization signal
   - Parameters: P (trained separately)

3. **Reference Model (πθ_ref)**: Fixed during PPO
   - Frozen copy of policy for KL divergence computation
   - Prevents policy from diverging too far from SFT
   - Parameters: P (copy of policy at start)

4. **Value Model (V)**: Trained alongside policy
   - Estimates expected future reward
   - Reduces variance in gradient estimates
   - Parameters: P (often same architecture as policy)

**Memory Budget Analysis (4 A100-80GB GPUs):**

```
Per-GPU allocation (assuming 4-GPU cluster):

Policy Model (πθ):
- Parameters (FP16): 7B × 2 bytes / 4 GPUs = 3.5 GB
- Gradients (FP16): 3.5 GB
- Optimizer state (Adam): 7B × 8 bytes / 4 GPUs = 14 GB
- Total policy: ~21 GB

Reward Model (R) inference:
- Parameters (FP16): 7B × 2 bytes / 4 GPUs = 3.5 GB
- Activations for batch: 2 GB
- Total RM: ~5.5 GB

Reference Model (πθ_ref) inference:
- Parameters (FP16): 7B × 2 bytes / 4 GPUs = 3.5 GB
- Activations for batch: 2 GB
- Total ref: ~5.5 GB

Value Model:
- Can share weights with policy
- If separate: additional 3.5 GB parameters

Activations during PPO:
- Generated tokens: 2-3 GB
- Trajectory buffer: 5-10 GB

GPU Total: 21 + 5.5 + 5.5 + 10 ≈ 42 GB (fits on 80GB A100)
```

**Timeline: PPO Training Step**

```
t=0:     Sample prompts from dataset
         [0-50ms CPU overhead]

t=1:     Policy generates responses (πθ)
         [500-1000ms: depends on sequence length]

t=2:     Reward model scores responses (R)
         [500ms: parallel inference]

t=3:     Reference model computes logits (πθ_ref)
         [500ms: parallel inference]

t=4:     Compute advantages via bootstrapping
         [50-100ms: CPU computation]

t=5:     PPO optimization step (value + policy)
         [1000-1500ms: backward pass]

Total per step: ~3-4 seconds

Typical training: 10-20 steps per epoch
1 epoch ≈ 30-80 seconds
Training duration: 100 epochs ≈ 1-2 hours per 10K examples
```

## 3. Supervised Fine-Tuning (SFT) Stage

### 3.1 SFT Dataset Composition

SFT stage trains the foundation for downstream RLHF:

**Data Quality Requirements:**
```
High-quality instruction-following examples:
{
  "instruction": "Explain quantum entanglement",
  "input": "",
  "output": "Quantum entanglement is a phenomenon where..."
}

Characteristics needed for good SFT:
1. Diverse task types (QA, summarization, code, reasoning)
2. High execution quality (correct, helpful, safe)
3. Adequate length (diverse example complexities)
4. Consistent formatting (enables pattern learning)

Typical dataset size: 10K-100K high-quality examples
Cost: $500K-$2M (human annotation at scale)
```

**SFT Training Procedure:**
```python
def sft_training_loop(model, train_loader, num_epochs=3):
    """
    Standard supervised fine-tuning
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * num_epochs
    )

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in train_loader:
            inputs = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            # Forward pass
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

# SFT metrics to monitor:
# - Validation loss (should decrease)
# - Task-specific metrics (BLEU for summarization, etc.)
# - Human evaluation (optional, on small validation set)
```

## 4. Reward Model Training

### 4.1 Reward Model Architecture & Training

Reward models learn to predict human preference over model responses:

**RM Training Setup:**
```
Input: Pairs of model-generated responses
Format: {prompt, response_A, response_B, preference (A or B)}

Model architecture:
- Base: Transformer encoder (often 6-8 layers of base model size)
- Classification head: Linear layer outputting single scalar score

Training objective:
- Maximize probability of correct preference prediction
- Binary cross-entropy loss between predicted and actual preference

Loss function:
L = -[y_true × log(sigmoid(score_A - score_B)) +
       (1 - y_true) × log(1 - sigmoid(score_A - score_B))]

Where:
- score_A: RM output for response_A
- score_B: RM output for response_B
- y_true: 1 if A preferred, 0 if B preferred
```

**RM Implementation:**
```python
class RewardModel(torch.nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()

        self.base_model = base_model  # Transformer encoder
        self.score_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        """
        Compute reward score for input sequence
        """

        # Encode sequence
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Take final token representation
        # Shape: [batch_size, hidden_size]
        final_hidden = hidden_states[:, -1, :]

        # Score
        reward = self.score_head(final_hidden)

        return reward

def rm_training_step(model, batch, optimizer):
    """
    Single training step for reward model
    """

    input_ids_a = batch['input_ids_a'].cuda()
    input_ids_b = batch['input_ids_b'].cuda()
    labels = batch['preference'].cuda()  # 1 if A preferred, 0 if B

    attention_mask_a = batch['attention_mask_a'].cuda()
    attention_mask_b = batch['attention_mask_b'].cuda()

    # Get rewards for both responses
    reward_a = model(input_ids_a, attention_mask_a)  # [batch, 1]
    reward_b = model(input_ids_b, attention_mask_b)  # [batch, 1]

    # Compute preference prediction
    # P(A preferred) = sigmoid(score_a - score_b)
    score_diff = reward_a - reward_b  # [batch, 1]
    predicted_prefs = torch.sigmoid(score_diff)

    # Binary cross-entropy loss
    loss = torch.nn.functional.binary_cross_entropy(
        predicted_prefs.squeeze(),
        labels.float()
    )

    # Optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

### 4.2 RM Validation & Calibration

**Accuracy Measurement:**
```python
def evaluate_reward_model(rm, validation_loader):
    """
    Evaluate RM accuracy on preference prediction
    """

    rm.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in validation_loader:
            input_ids_a = batch['input_ids_a'].cuda()
            input_ids_b = batch['input_ids_b'].cuda()
            labels = batch['preference'].cuda()

            reward_a = rm(input_ids_a).squeeze()
            reward_b = rm(input_ids_b).squeeze()

            # Predict preference: A if reward_a > reward_b
            predicted = (reward_a > reward_b).long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

# Expected RM accuracy: 60-80% (depends on dataset noise)
# Low accuracy (<55%): Indicates dataset issues or inadequate training
# High accuracy (>90%): May indicate overfitting on specific distribution
```

## 5. PPO (Proximal Policy Optimization)

### 5.1 PPO Algorithm Fundamentals

PPO is the standard RL algorithm for RLHF, balancing stability and efficiency:

**Policy Gradient Theorem:**
```
Objective (undiscounted):
J(π) = E_τ~π [R(τ)]

Where:
- τ: trajectory (sequence of states and actions)
- R(τ): total return (sum of rewards)

Policy gradient:
∇J(π) = E_τ~π [∇log π(a|s) × Q(s,a)]

Where:
- Q(s,a): action-value function (expected return from action a in state s)
- ∇log π(a|s): gradient of policy log-probability
```

**PPO Clipped Objective:**
```
PPO addresses policy divergence via clipping:

L^CLIP(θ) = Ê_t [min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t): probability ratio
- Â_t: advantage estimate
- ε: clipping parameter (typical: 0.2)

Effect: Prevents probability ratio from deviating too far from 1
- When r_t > 1+ε: gradient zeroed (no further improvement)
- When r_t < 1-ε: gradient zeroed (no further degradation)
- Otherwise: normal policy gradient
```

**KL Divergence Penalty:**
```
RLHF-specific modification:

L = L^CLIP(θ) - β × KL(π_θ || π_ref)

Where:
- β: KL penalty coefficient (typical: 0.01-0.05)
- KL divergence ensures policy doesn't drift from SFT
- Balances task performance vs alignment stability

KL computation:
KL(π_θ || π_ref) = Σ π_ref(a|s) × log(π_ref(a|s) / π_θ(a|s))

Practical approximation (importance-weighted):
KL ≈ (π_ref(a|s) / π_θ(a|s) - 1) / 2
```

### 5.2 PPO Training Loop

**PPO Implementation:**
```python
def ppo_training_step(
    policy_model,
    reference_model,
    reward_model,
    value_model,
    prompts,
    max_new_tokens=512,
    num_epochs=4,
    num_minibatches=8,
):
    """
    Single PPO training iteration
    """

    # Phase 1: Generate responses using policy
    with torch.no_grad():
        # Generate completions
        outputs = policy_model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences
        generated_text = tokenizer.batch_decode(generated_ids)

    # Phase 2: Compute rewards
    with torch.no_grad():
        rewards = reward_model(generated_ids)  # [batch, 1]
        rewards = rewards.squeeze()

    # Phase 3: Compute reference logits (for KL penalty)
    with torch.no_grad():
        ref_logits = reference_model(generated_ids).logits

    # Phase 4: Collect trajectories for PPO optimization
    trajectories = {
        'prompts': prompts,
        'generated_ids': generated_ids,
        'rewards': rewards,
        'ref_logits': ref_logits,
    }

    # Phase 5: Compute advantages (bootstrapped returns)
    with torch.no_grad():
        values = value_model(generated_ids).squeeze()

        # Compute returns (rewards are terminal only)
        returns = rewards.clone()

        # Advantage = Return - Value estimate
        advantages = returns - values

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Phase 6: PPO optimization (multiple epochs over trajectories)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)

    for ppo_epoch in range(num_epochs):
        # Shuffle trajectories
        indices = torch.randperm(len(prompts))

        for minibatch_idx in range(num_minibatches):
            # Select minibatch
            start = minibatch_idx * (len(prompts) // num_minibatches)
            end = start + (len(prompts) // num_minibatches)
            mb_indices = indices[start:end]

            mb_generated_ids = generated_ids[mb_indices]
            mb_rewards = rewards[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_ref_logits = ref_logits[mb_indices]

            # Forward pass with current policy
            policy_outputs = policy_model(mb_generated_ids)
            policy_logits = policy_outputs.logits

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(mb_ref_logits, dim=-1)

            # Extract log prob of generated token
            action_log_probs = log_probs.gather(
                -1, mb_generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_action_log_probs = ref_log_probs.gather(
                -1, mb_generated_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Probability ratio
            ratio = torch.exp(action_log_probs - ref_action_log_probs)

            # PPO clipped loss
            clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
            policy_loss = -torch.min(
                ratio * mb_advantages,
                clipped_ratio * mb_advantages
            ).mean()

            # KL divergence penalty
            kl_loss = (ratio - 1) / 2
            kl_penalty = 0.01 * kl_loss.mean()

            # Value function loss
            values = value_model(mb_generated_ids).squeeze()
            value_loss = torch.nn.functional.mse_loss(values, mb_rewards)

            # Total loss
            total_loss = policy_loss + kl_penalty + 0.5 * value_loss

            # Optimization step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    return {
        'policy_loss': policy_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'value_loss': value_loss.item(),
        'mean_reward': rewards.mean().item(),
    }
```

### 5.3 PPO Memory Requirements

**Memory Breakdown During PPO:**

```
Policy model training:
- Parameters (FP16): 3.5 GB
- Gradients (FP16): 3.5 GB
- Optimizer state (Adam, per-GPU sharded): 14 GB
- Activation cache (trajectory buffer): 5-10 GB
Total: ~26-31 GB per GPU

Reference model (frozen):
- Parameters (FP16): 3.5 GB
- Activations: 2 GB
Total: ~5.5 GB per GPU

Reward model (frozen):
- Parameters (FP16): 3.5 GB
- Activations: 2 GB
Total: ~5.5 GB per GPU

Value model (separate):
- Parameters (FP16): 3.5 GB (if shared with policy, included above)
- Gradients: 3.5 GB (if shared)
Total: 0 GB additional (if shared)

Per-GPU total: ~38 GB (fits on A100-80GB)
```

**Optimization Techniques:**

```python
def optimize_ppo_memory():
    """Memory optimization strategies for PPO"""

    # 1. Share value model with policy (no separate parameters)
    # Effect: -3.5 GB parameters, -3.5 GB gradients

    # 2. Use gradient checkpointing
    with torch.checkpoint(policy_model, use_reentrant=False):
        outputs = policy_model(generated_ids)
    # Effect: -5-10 GB activation cache

    # 3. Reduce batch size for generation
    # Generate in multiple smaller batches, accumulate rewards
    # Effect: Reduces activation buffer size

    # 4. Use 8-bit optimizer (AdamW8bit)
    # Reduces optimizer state from 8 bytes to 2 bytes per parameter
    # Effect: -28 GB optimizer state per GPU

    # 5. Quantize reward/reference models (FP8 or INT8)
    # Effect: -1-2 GB per frozen model
```

## 6. Direct Preference Optimization (DPO)

### 6.1 DPO Motivation & Formulation

DPO (Rafailov et al., NeurIPS 2023) eliminates PPO complexity by directly optimizing preference predictions:

**Problem with Classical RLHF:**
```
PPO requires:
1. Reward model training (separate stage)
2. Policy, reference, value models simultaneously
3. Complex RL training loop
4. Careful hyperparameter tuning (KL penalty, clipping)
5. Total: 3-4 stages, months of engineering

Issues:
- Reward overoptimization (policy exploits RM errors)
- Non-stationary reward signal during training
- High variance in advantage estimates
```

**DPO Key Insight:**
```
Observation: Preference distribution p(y_w > y_l | x) follows specific form

Standard RL formulation:
p(y_w > y_l | x) ∝ exp(r(x, y_w) - r(x, y_l)) / Z(x)

Equivalently:
r(x, y) = β × log(π(y|x) / π_ref(y|x)) + const

Therefore:
- Can optimize π directly using preference pairs
- No separate reward model needed
- No RL required
```

**DPO Objective:**
```
Loss = -log σ(β × log(π_θ(y_w|x) / π_ref(y_w|x)) -
                    β × log(π_θ(y_l|x) / π_ref(y_l|x)))

Simplified:
Loss = -log σ(β × [log ratio_w - log ratio_l])

Where:
- y_w: preferred response
- y_l: dispreferred response
- β: temperature parameter (controls strength of preference)

Effect: Directly maximize probability of preference without RM
```

### 6.2 DPO Implementation

**DPO Training:**
```python
def dpo_training_step(
    model,
    reference_model,
    batch,
    beta=0.5,
    learning_rate=5e-6
):
    """
    Single DPO training step
    """

    input_ids = batch['prompt_input_ids'].cuda()
    w_input_ids = batch['w_input_ids'].cuda()  # Preferred
    l_input_ids = batch['l_input_ids'].cuda()  # Dispreferred

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Get policy logits
    policy_w_logits = model(w_input_ids).logits
    policy_l_logits = model(l_input_ids).logits

    # Get reference logits (frozen)
    with torch.no_grad():
        ref_w_logits = reference_model(w_input_ids).logits
        ref_l_logits = reference_model(l_input_ids).logits

    # Compute log probabilities (per-token)
    policy_w_log_probs = torch.nn.functional.log_softmax(policy_w_logits, dim=-1)
    policy_l_log_probs = torch.nn.functional.log_softmax(policy_l_logits, dim=-1)

    ref_w_log_probs = torch.nn.functional.log_softmax(ref_w_logits, dim=-1)
    ref_l_log_probs = torch.nn.functional.log_softmax(ref_l_logits, dim=-1)

    # Extract log prob of actual tokens
    # (sum across sequence for sequence-level probability)
    def get_sequence_log_prob(logits, token_ids):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        seq_log_probs = log_probs.gather(
            -1, token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return seq_log_probs.sum(dim=-1)

    policy_w_seq_log_prob = get_sequence_log_prob(policy_w_logits, w_input_ids)
    policy_l_seq_log_prob = get_sequence_log_prob(policy_l_logits, l_input_ids)

    with torch.no_grad():
        ref_w_seq_log_prob = get_sequence_log_prob(ref_w_logits, w_input_ids)
        ref_l_seq_log_prob = get_sequence_log_prob(ref_l_logits, l_input_ids)

    # Log probability ratios
    log_ratio_w = policy_w_seq_log_prob - ref_w_seq_log_prob
    log_ratio_l = policy_l_seq_log_prob - ref_l_seq_log_prob

    # DPO loss: maximize probability that π prefers w over l
    dpo_loss = -torch.nn.functional.logsigmoid(
        beta * (log_ratio_w - log_ratio_l)
    ).mean()

    # Backward pass
    dpo_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return dpo_loss.item()
```

### 6.3 DPO Memory & Efficiency

**Memory Comparison: PPO vs DPO**

| Aspect | PPO | DPO |
|--------|-----|-----|
| Models needed | 4 (policy, RM, ref, value) | 2 (policy, ref) |
| RM training | Separate stage | Not needed |
| GPU memory | 38 GB per GPU (4 models) | 20 GB per GPU (2 models) |
| Training time | 2-3x longer | 1x baseline |
| Stages | 3-4 (SFT → RM → PPO → eval) | 2 (SFT → DPO) |
| Code complexity | ~1000 LOC | ~200 LOC |
| Stability | Requires careful tuning | More stable |

**DPO Advantages:**
```
1. Simpler training pipeline
   - No separate reward model
   - No complex RL algorithm
   - Direct preference optimization

2. Better memory efficiency
   - 50% fewer model parameters
   - Eliminates value model
   - Straightforward GPU allocation

3. Training stability
   - Direct gradient signal
   - No reward overoptimization
   - Predictable convergence

4. Production benefits
   - Faster deployment (fewer stages)
   - Easier debugging
   - Reproducible results
```

### 6.4 DPO β Parameter Tuning

**Temperature Parameter (β) Analysis:**
```
DPO objective:
Loss = -log σ(β × [log(π/π_ref)_w - log(π/π_ref)_l])

Effect of β:
- β → 0: Loss becomes flat (no preference signal)
- β = 1: Standard KL-constrained preference
- β → ∞: Hard constraint (must get preferences correct)

Practical considerations:
- Too small β (< 0.1): Weak optimization signal, slow convergence
- Standard β (0.5-1.0): Balanced preference and KL
- Too large β (> 2.0): Over-focuses on preferences, may hurt performance

Empirical tuning:
β = 0.5: Conservative, maintains SFT quality
β = 1.0: Balanced, common default
β = 2.0: Aggressive, maximum preference optimization
```

## 7. GRPO & Recent Advances

### 7.1 GRPO (Group Relative Policy Optimization)

GRPO (DeepSeek-R1) improves on DPO for reasoning tasks:

**GRPO Key Ideas:**
```
Motivation: Standard DPO compares pairs
Issue: Doesn't leverage multi-response comparisons

GRPO approach:
- Sample multiple responses per prompt
- Compare within group (relative rankings)
- Direct policy gradient from preference ordering

Advantage: Better signal from comparing multiple candidates
Typical improvement: 2-5% on reasoning tasks vs DPO
```

**GRPO Implementation:**
```python
def grpo_training_step(
    model,
    reference_model,
    prompts,
    num_responses=4,
    beta=0.5
):
    """
    GRPO training with multiple responses per prompt
    """

    # Sample multiple responses per prompt
    with torch.no_grad():
        all_responses = []
        for _ in range(num_responses):
            responses = model.generate(
                prompts, max_new_tokens=512
            )
            all_responses.append(responses)

        # Score responses (using reward model or human preference)
        scores = score_responses(all_responses)

        # Rank responses
        rankings = torch.argsort(scores, descending=True)

    # Compute policy gradients based on ranking
    policy_loss = 0.0

    for i, prompt in enumerate(prompts):
        for j in range(num_responses):
            for k in range(j + 1, num_responses):
                # Compare response[j] (better) vs response[k] (worse)
                response_better = all_responses[rankings[i, j]][i]
                response_worse = all_responses[rankings[i, k]][i]

                # Get log probabilities
                policy_logits_better = model(response_better).logits
                policy_logits_worse = model(response_worse).logits

                with torch.no_grad():
                    ref_logits_better = reference_model(response_better).logits
                    ref_logits_worse = reference_model(response_worse).logits

                # Log probability ratios
                log_ratio_better = compute_seq_log_prob(
                    policy_logits_better, ref_logits_better
                )
                log_ratio_worse = compute_seq_log_prob(
                    policy_logits_worse, ref_logits_worse
                )

                # Policy gradient
                loss = -torch.nn.functional.logsigmoid(
                    beta * (log_ratio_better - log_ratio_worse)
                )
                policy_loss += loss

    return policy_loss / (num_responses * (num_responses - 1) / 2)
```

## 8. Reward Model Inference at Scale

### 8.1 Batch Reward Scoring

Reward models must score thousands of responses efficiently:

**Batched Reward Inference:**
```python
class ScalableRewardModel:
    def __init__(self, model, max_batch_size=256):
        self.model = model
        self.max_batch_size = max_batch_size

    def score_responses(self, responses, batch_size=None):
        """
        Efficient batch scoring of responses
        """

        if batch_size is None:
            batch_size = self.max_batch_size

        scores = []

        for i in range(0, len(responses), batch_size):
            batch = responses[i:i+batch_size]
            input_ids = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'].cuda()

            with torch.no_grad():
                batch_scores = self.model(input_ids).logits.squeeze(-1)

            scores.extend(batch_scores.cpu().tolist())

        return torch.tensor(scores)

    def rank_responses(self, prompt, responses):
        """
        Rank multiple responses for a prompt
        """

        scores = self.score_responses(responses)
        rankings = torch.argsort(scores, descending=True)

        return [
            {'response': responses[idx], 'score': scores[idx].item()}
            for idx in rankings
        ]

# Throughput: ~1000 scores/second on single A100
# Batching improves throughput: 2x with batch_size=256
```

### 8.2 Distributed Reward Scoring

For large-scale scoring:

```python
class DistributedRewardScoring:
    def __init__(self, model, num_gpus=4):
        self.model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=list(range(num_gpus))
        )
        self.num_gpus = num_gpus

    def score_batch(self, responses, distributed_batch_size=64):
        """
        Score across multiple GPUs
        """

        # Split across GPUs
        per_gpu_batch = distributed_batch_size // self.num_gpus

        all_scores = []

        for i in range(0, len(responses), distributed_batch_size):
            batch = responses[i:i+distributed_batch_size]

            # Prepare inputs
            input_ids = torch.stack([
                tokenizer(r, return_tensors='pt')['input_ids']
                for r in batch
            ]).cuda()

            # Forward pass (distributed)
            with torch.no_grad():
                batch_scores = self.model(input_ids)

            all_scores.extend(batch_scores.cpu().tolist())

        return torch.tensor(all_scores)

# With 4 GPUs: ~4000 scores/second
# Scales linearly with number of GPUs
```

## 9. Alignment Testing & Evaluation

### 9.1 Evaluation Framework

**Multi-Dimensional Evaluation:**

```python
class AlignmentEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate(self, model, test_prompts):
        """
        Comprehensive alignment evaluation
        """

        results = {
            'helpfulness': self.eval_helpfulness(model, test_prompts),
            'harmlessness': self.eval_harmlessness(model, test_prompts),
            'honesty': self.eval_honesty(model, test_prompts),
            'reasoning': self.eval_reasoning(model, test_prompts),
        }

        return results

    def eval_helpfulness(self, model, prompts):
        """
        Measure how helpful responses are
        Using automated metrics: ROUGE, BERTScore, etc.
        """
        scores = []

        for prompt in prompts:
            response = model.generate(prompt)

            # Automated metric
            score = compute_helpfulness_metric(response)
            scores.append(score)

        return {'mean': np.mean(scores), 'std': np.std(scores)}

    def eval_harmlessness(self, model, harmful_prompts):
        """
        Measure resistance to generating harmful content
        """
        harm_count = 0

        for prompt in harmful_prompts:
            response = model.generate(prompt)

            # Check if response contains harmful content
            if contains_harmful_content(response):
                harm_count += 1

        refusal_rate = 1 - (harm_count / len(harmful_prompts))

        return {'refusal_rate': refusal_rate}

    def eval_honesty(self, model, factual_prompts):
        """
        Measure factual accuracy
        """
        accuracy_scores = []

        for prompt, ground_truth in factual_prompts:
            response = model.generate(prompt)

            # Compute factual accuracy
            accuracy = check_factual_accuracy(response, ground_truth)
            accuracy_scores.append(accuracy)

        return {'accuracy': np.mean(accuracy_scores)}

    def eval_reasoning(self, model, reasoning_prompts):
        """
        Evaluate reasoning capability (MATH, GSM8K, etc.)
        """
        correct = 0

        for prompt, expected_answer in reasoning_prompts:
            response = model.generate(prompt)

            # Check if answer matches
            if extract_answer(response) == expected_answer:
                correct += 1

        accuracy = correct / len(reasoning_prompts)

        return {'accuracy': accuracy}
```

### 9.2 Human Evaluation

**Structured Human Evaluation Protocol:**

```python
class HumanEvaluationProtocol:
    def __init__(self, num_annotators=3):
        self.num_annotators = num_annotators
        self.annotations = []

    def prepare_evaluation_set(self, model_a, model_b, prompts, sample_size=100):
        """
        Prepare blind comparison between two models
        """

        evaluation_data = []

        for prompt in random.sample(prompts, sample_size):
            response_a = model_a.generate(prompt)
            response_b = model_b.generate(prompt)

            # Randomize order (to avoid bias)
            if random.random() > 0.5:
                response_a, response_b = response_b, response_a
                model_order = ['B', 'A']
            else:
                model_order = ['A', 'B']

            evaluation_data.append({
                'prompt': prompt,
                'response_1': response_a,
                'response_2': response_b,
                'ground_truth_order': model_order,
            })

        return evaluation_data

    def aggregate_annotations(self, annotations):
        """
        Aggregate multiple annotations with inter-rater agreement
        """

        agreements = 0
        total = len(annotations[0])

        for i in range(total):
            votes = [ann[i]['preference'] for ann in annotations]

            # Majority vote
            if votes.count(votes[0]) >= (len(votes) / 2 + 1):
                agreements += 1

        agreement_rate = agreements / total

        # Fleiss' kappa for inter-rater reliability
        kappa = compute_fleiss_kappa(annotations)

        return {
            'agreement_rate': agreement_rate,
            'fleiss_kappa': kappa,
        }
```

## 10. Summary & Production Deployment

### 10.1 RLHF vs DPO Comparison

| Criterion | RLHF (PPO) | DPO |
|-----------|-----------|-----|
| Complexity | High (4 models) | Low (2 models) |
| Training time | 3-4 weeks | 1-2 weeks |
| GPU memory | 38 GB per GPU | 20 GB per GPU |
| Accuracy | Baseline | +0.5-1% |
| Stability | Requires tuning | More stable |
| Production ready | Proven | Increasingly adopted |
| Recommended | Large-scale projects | Most new projects |

**Recommendation**: Use DPO for new projects. Use PPO only if:
- Explicit reward modeling is required
- Task requires complex multi-objective optimization
- Team has deep RL expertise

### 10.2 Training Timeline

**Classical RLHF (16 weeks):**
```
Week 1-2:   Prepare SFT data, annotation
Week 3-4:   SFT model training
Week 5-6:   Collect preference data for RM
Week 7-8:   Train reward model
Week 9-14:  PPO training (extended due to complexity)
Week 15-16: Evaluation and fine-tuning
```

**DPO Pipeline (8 weeks):**
```
Week 1-2:   Prepare SFT data, annotation
Week 3-4:   SFT model training
Week 5-6:   Collect preference data
Week 7:     DPO training (faster convergence)
Week 8:     Evaluation and deployment
```

**Speedup: 2× faster with DPO**

### 10.3 Hyperparameter Checklists

**DPO Training:**
- [ ] β parameter: 0.5-1.0 (test both, pick best on validation)
- [ ] Learning rate: 5e-6 to 5e-5
- [ ] Warm-up steps: 100-500
- [ ] Batch size: 32-64 per GPU
- [ ] Training epochs: 2-3
- [ ] KL divergence penalty: Monitor during training
- [ ] Validation: Every 100 steps

**Evaluation Metrics:**
- [ ] Task-specific accuracy (downstream evaluation)
- [ ] Human preference (small validation set, 100+ examples)
- [ ] Refusal rate (shouldn't exceed 10%)
- [ ] Reasoning performance (MATH, GSM8K, etc.)
- [ ] Factuality (on factual QA benchmarks)

### 10.4 Key Reading

- Ouyang et al. (2022). "Training language models to follow instructions with human feedback" (original RLHF)
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms" (PPO)
- Rafailov et al. (NeurIPS 2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- DeepSeek-R1 Team (2024). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (GRPO)

### 10.5 Production Checklist

- [ ] Complete RLHF or DPO training pipeline
- [ ] Validate model on standard benchmarks (MMLU, GSM8K, etc.)
- [ ] Conduct human evaluation (minimum 500 examples)
- [ ] Test safety/harmlessness extensively
- [ ] Set up monitoring for output quality
- [ ] Implement user feedback collection loop
- [ ] Plan continuous alignment improvement cycle
- [ ] Document model training procedure for reproducibility

---

**Module Completion Status**: Comprehensive coverage of RLHF and alignment systems from classical PPO-based approaches through modern DPO methods, with emphasis on systems design, memory efficiency, and practical deployment considerations.

**Cumulative Learning Outcome**: Mastery of the complete fine-tuning and alignment pipeline from SFT through LoRA/PEFT to RLHF/DPO, enabling deployment of production-grade instruction-following language models with systematic optimization across memory, compute, and training stability dimensions.
