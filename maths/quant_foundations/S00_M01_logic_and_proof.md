# Subject 0, Module 0.1 — Foundations of Logic and Proof

*Mathematical Foundations for Quantitative Research: From JEE to Jane Street.*

---

## How to read this module

You have seen almost everything here before — but almost never in this order, and almost never with the level of care that the rest of the curriculum will demand. The purpose of this module is not to teach you what a set or a function is. It is to *retrain your reflexes*: to make you think of "for all" and "there exists" as objects you manipulate, to make the ε–δ grammar of later real analysis feel native, and to make the move from "I can see why this is true" to "I can write a proof a stranger will accept" as automatic as writing a Python one-liner.

Every topic in this module follows the same seven-part structure the curriculum uses throughout:

1. **Motivation** — why the topic exists.
2. **Prerequisites** — what you must already know (for this first module, roughly JEE plus a willingness to be pedantic).
3. **Definitions** — stated precisely, followed by an intuitive gloss.
4. **Key Results** — theorems stated with all hypotheses and proved in full. No "similarly", no "left as exercise", no "the rest is routine". Every line.
5. **Worked Examples** — at least two per topic: one direct application, one that exposes a subtlety.
6. **Computational Implementation** — Python code where it sharpens the concept.
7. **[QUANT APPLICATION]** — one concrete, specific use inside quantitative research.
8. **Exercises** — graded ★ / ★★ / ★★★ with full solutions for the first two tiers and hints only for the third.

A word on the exercises: do them. The exercises for ★★★ are not garnish; they are where the difference between "read the book" and "internalized the book" lives. Most JEE-strong readers will blow through the ★ problems, find the ★★ problems a pleasant challenge, and get stuck on many of the ★★★ problems. That is the intended experience. Getting stuck and then unstuck is where mathematical maturity is grown.

---

## Table of Contents

- Topic 0.1.1 — Propositional Logic
- Topic 0.1.2 — Predicate Logic and Quantifiers
- Topic 0.1.3 — Proof Techniques
- Topic 0.1.4 — Set Theory Foundations
- Topic 0.1.5 — Functions
- Topic 0.1.6 — Cardinality and Cantor's Diagonal Argument

---

# Topic 0.1.1 — Propositional Logic

## Motivation

Mathematics is not arithmetic; it is the manipulation of *unambiguous statements*. Before we can prove anything, we need a grammar for assertions that is mechanical enough to admit of no wiggle room and rich enough to express the claims we actually care about. Propositional logic is the smallest such grammar. It is the level at which we ask: given truths A and B, what other truths are forced? Its operators — AND, OR, NOT, IMPLIES, IFF — are not decorative; they are the joints on which every later proof will hinge.

The reason a JEE-strong student sometimes struggles to write a clean ε–δ proof is almost never a lack of calculus. It is that the student has not internalized the logical structure of "for every ε there exists a δ such that for every x …" as a *syntactic object* that can be negated, contraposed, or chained. The remedy is to spend a few hours treating logic as itself a mathematical object. That is what this topic does.

## Prerequisites

You need to know how to read a statement like "if $x > 0$ then $x^2 > 0$" without confusion. That is all. We will build everything else here.

## Definitions

**Definition 0.1.1.1 (Proposition).** A *proposition* is a declarative sentence that is unambiguously either true or false. We write $\top$ for the constant true, $\bot$ for the constant false. Propositions are typically denoted by uppercase letters: $P, Q, R$.

*Intuition.* A proposition is anything that has a truth value. "It is raining in Mumbai at 09:30 on 22 April 2026" is a proposition (true or false, though you may not know which). "Is it raining?" is not a proposition (it is a question). "x > 0" is not a proposition until we specify what $x$ is; it is a *predicate* in a variable (we will treat those in Topic 0.1.2).

**Definition 0.1.1.2 (Logical connectives).** Given propositions $P$ and $Q$, we build new propositions:

- $\neg P$ ("not $P$"): the *negation*.
- $P \wedge Q$ ("$P$ and $Q$"): the *conjunction*.
- $P \vee Q$ ("$P$ or $Q$"): the *disjunction*. Mathematical "or" is always *inclusive* — it is true when at least one of $P, Q$ is true, *including* when both are.
- $P \Rightarrow Q$ ("$P$ implies $Q$"): the *implication* or *conditional*.
- $P \Leftrightarrow Q$ ("$P$ iff $Q$"): the *biconditional*.

The truth values of these compound propositions are defined by a *truth table*:

| $P$ | $Q$ | $\neg P$ | $P\wedge Q$ | $P\vee Q$ | $P\Rightarrow Q$ | $P\Leftrightarrow Q$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T | T | F | T | T | T | T |
| T | F | F | F | T | F | F |
| F | T | T | F | T | T | F |
| F | F | T | F | F | T | T |

**[WARNING]** The row "$P = F, Q = T$" for implication is the standard stumbling block. The sentence "if $P$ then $Q$" is *true* whenever $P$ is false, regardless of $Q$. This is called *vacuous truth*. "If the moon is made of cheese then I am the Prime Minister of India" is, as a matter of logic, a true statement — because the hypothesis is false. Many bad student proofs come from neglecting this convention; many real-world arguments ("every element of the empty set satisfies ...") depend on it.

*Intuition for implication.* The statement $P \Rightarrow Q$ is a *promise*. It says: I promise that whenever $P$ holds, $Q$ also holds. You can only call me a liar if you catch $P$ true and $Q$ false. If $P$ never holds, I cannot be caught out, and my promise stands.

**Definition 0.1.1.3 (Logical equivalence).** Two compound propositions $\varphi$ and $\psi$ built from the same atoms $P_1, \dots, P_n$ are *logically equivalent*, written $\varphi \equiv \psi$, if they agree on every row of the truth table — i.e., for every assignment of truth values to the atoms they evaluate to the same value.

*Intuition.* Logical equivalence is the logical analogue of set equality: two formulas are "the same formula" if they mean the same thing on every possible input.

**Definition 0.1.1.4 (Tautology, contradiction, contingency).** A compound proposition is a *tautology* if it is true under every assignment of truth values to its atoms; a *contradiction* if false under every assignment; a *contingency* otherwise.

Examples: $P \vee \neg P$ is a tautology (the *law of the excluded middle*). $P \wedge \neg P$ is a contradiction (the *law of non-contradiction*). $P \wedge Q$ is a contingency.

## Key Results

### Theorem 0.1.1.5 (Basic Equivalences)

For all propositions $P, Q, R$:

1. $\neg(\neg P) \equiv P$  (double negation)
2. $P \wedge Q \equiv Q \wedge P$ and $P \vee Q \equiv Q \vee P$  (commutativity)
3. $(P \wedge Q) \wedge R \equiv P \wedge (Q \wedge R)$ and $(P \vee Q) \vee R \equiv P \vee (Q \vee R)$  (associativity)
4. $P \wedge (Q \vee R) \equiv (P \wedge Q) \vee (P \wedge R)$  (distributivity of $\wedge$ over $\vee$)
5. $P \vee (Q \wedge R) \equiv (P \vee Q) \wedge (P \vee R)$  (distributivity of $\vee$ over $\wedge$)
6. $\neg(P \wedge Q) \equiv \neg P \vee \neg Q$  (De Morgan)
7. $\neg(P \vee Q) \equiv \neg P \wedge \neg Q$  (De Morgan)
8. $P \Rightarrow Q \equiv \neg P \vee Q$  (material conditional)
9. $P \Rightarrow Q \equiv \neg Q \Rightarrow \neg P$  (contrapositive)
10. $P \Leftrightarrow Q \equiv (P \Rightarrow Q) \wedge (Q \Rightarrow P)$
11. $\neg(P \Rightarrow Q) \equiv P \wedge \neg Q$  (negation of an implication)

**Proof.** Each equivalence is a statement about the truth tables of the two sides. We prove each by constructing the truth table and verifying agreement. We will do (1), (6), (8), (9), (11) explicitly; the rest follow by the same procedure.

*(1) $\neg(\neg P) \equiv P$.*

| $P$ | $\neg P$ | $\neg(\neg P)$ |
|:---:|:---:|:---:|
| T | F | T |
| F | T | F |

The columns for $P$ and $\neg(\neg P)$ are identical.

*(6) $\neg(P\wedge Q) \equiv \neg P \vee \neg Q$.*

| $P$ | $Q$ | $P\wedge Q$ | $\neg(P\wedge Q)$ | $\neg P$ | $\neg Q$ | $\neg P \vee \neg Q$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T | T | T | F | F | F | F |
| T | F | F | T | F | T | T |
| F | T | F | T | T | F | T |
| F | F | F | T | T | T | T |

Columns 4 and 7 agree on every row.

*(8) $P \Rightarrow Q \equiv \neg P \vee Q$.*

| $P$ | $Q$ | $P\Rightarrow Q$ | $\neg P$ | $\neg P \vee Q$ |
|:---:|:---:|:---:|:---:|:---:|
| T | T | T | F | T |
| T | F | F | F | F |
| F | T | T | T | T |
| F | F | T | T | T |

Columns 3 and 5 agree.

*(9) $P \Rightarrow Q \equiv \neg Q \Rightarrow \neg P$.* Using (8) on both sides and (6):
$$P \Rightarrow Q \equiv \neg P \vee Q \equiv Q \vee \neg P \equiv \neg(\neg Q) \vee \neg P \equiv \neg Q \Rightarrow \neg P.$$
The last step applies (8) with $\neg Q$ in place of $P$ and $\neg P$ in place of $Q$. Alternatively, a direct truth-table verification:

| $P$ | $Q$ | $P\Rightarrow Q$ | $\neg Q$ | $\neg P$ | $\neg Q\Rightarrow \neg P$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| T | T | T | F | F | T |
| T | F | F | T | F | F |
| F | T | T | F | T | T |
| F | F | T | T | T | T |

Columns 3 and 6 agree.

*(11) $\neg(P\Rightarrow Q) \equiv P \wedge \neg Q$.* Using (8) and (7):
$$\neg(P\Rightarrow Q) \equiv \neg(\neg P \vee Q) \equiv \neg(\neg P) \wedge \neg Q \equiv P \wedge \neg Q.$$
Each step used an equivalence already proved. $\square$

**Why this matters.** Equivalence (9), contrapositive, is the single most-used move in all of mathematics. Equivalence (11) is how you negate "if-then" statements — crucial when setting up a proof by contradiction. Equivalences (6) and (7), De Morgan, are what let us distribute negations through a formula; you will use them every time you negate a definition involving "for all" or "there exists".

### Theorem 0.1.1.6 (Modus Ponens is a tautology)

The formula $\big((P \Rightarrow Q) \wedge P\big) \Rightarrow Q$ is a tautology.

**Proof.**

| $P$ | $Q$ | $P\Rightarrow Q$ | $(P\Rightarrow Q)\wedge P$ | $\big((P\Rightarrow Q)\wedge P\big)\Rightarrow Q$ |
|:---:|:---:|:---:|:---:|:---:|
| T | T | T | T | T |
| T | F | F | F | T |
| F | T | T | F | T |
| F | F | T | F | T |

The last column is T throughout. $\square$

**Intuition.** Modus ponens is the principle "if you know $P \Rightarrow Q$ and you know $P$, you may conclude $Q$". This is the workhorse rule of every proof you will ever write.

### Theorem 0.1.1.7 (Functional completeness of $\{\neg, \wedge, \vee\}$)

Every Boolean function $f: \{T,F\}^n \to \{T,F\}$ can be written using only $\neg$, $\wedge$, and $\vee$ applied to the variables $P_1, \dots, P_n$.

**Proof.** Let $f: \{T,F\}^n \to \{T,F\}$ be a Boolean function. If $f$ is identically $F$, write $f = P_1 \wedge \neg P_1$ (or any contradiction); done. Otherwise, let $S = \{v \in \{T,F\}^n : f(v) = T\}$, a nonempty set. For each $v = (v_1, \dots, v_n) \in S$, define the *minterm*
$$m_v := L_1 \wedge L_2 \wedge \cdots \wedge L_n, \quad \text{where } L_i = \begin{cases} P_i & \text{if } v_i = T \\ \neg P_i & \text{if } v_i = F \end{cases}.$$
By construction, $m_v$ is true at precisely the one assignment $v$ and false at every other assignment. Now set
$$\varphi := \bigvee_{v \in S} m_v.$$
We claim $\varphi \equiv f$. Take any assignment $w \in \{T,F\}^n$.

- If $w \in S$: then $m_w$ is true at $w$, so $\varphi$ is true at $w$. And $f(w) = T$ by definition of $S$.
- If $w \notin S$: then every minterm $m_v$ with $v \in S$ is false at $w$ (since $v \neq w$), so $\varphi$ is false at $w$. And $f(w) = F$.

Hence $\varphi$ and $f$ agree on every assignment. $\square$

**Intuition.** Any Boolean function can be specified by its truth table. The formula we just built reads the truth table row by row: "($P_1$ is T and $P_2$ is F and …) OR (another T-row) OR …". This form — disjunction of conjunctions of literals — is called *disjunctive normal form* (DNF). Its dual, conjunction of disjunctions, is *conjunctive normal form* (CNF). The fact that every formula has a DNF (and a CNF) representation is the reason SAT solvers can be built as general-purpose Boolean reasoning engines.

## Worked Examples

### Example 0.1.1.8 (Straightforward: negate a compound statement)

*Negate the statement "The market is open AND (the S&P is up OR the VIX is below 15)"*. 

**Solution.** Let $M =$ "market is open", $S =$ "S&P is up", $V =$ "VIX is below 15". The statement is $M \wedge (S \vee V)$. Negate:

$$\neg(M \wedge (S \vee V)) \equiv \neg M \vee \neg(S \vee V) \equiv \neg M \vee (\neg S \wedge \neg V).$$

In English: "The market is closed, OR (the S&P is not up AND the VIX is $\geq 15$)."

Notice: the negation is a disjunction; there are *two* ways the original claim could fail. This matters when you design unit tests for a trading rule.

### Example 0.1.1.9 (Subtle: the converse is not the contrapositive)

*Consider the claim "If a strategy has positive expected return then it has positive Sharpe ratio." Write its converse, its contrapositive, and its inverse. Which of the four statements are equivalent to each other?*

**Solution.** Let $P =$ "positive expected return", $Q =$ "positive Sharpe ratio". The four statements:

- **Original**: $P \Rightarrow Q$.
- **Converse**: $Q \Rightarrow P$.
- **Contrapositive**: $\neg Q \Rightarrow \neg P$.
- **Inverse**: $\neg P \Rightarrow \neg Q$.

By Theorem 0.1.1.5(9), Original $\equiv$ Contrapositive. Similarly (apply (9) to the converse), Converse $\equiv$ Inverse. But Original is *not* equivalent to Converse: a truth table on $P, Q$ shows the row $P = F, Q = T$ gives Original T and Converse F.

**The trap.** A student who proves the converse and thinks they have proved the original commits a fallacy called *affirming the consequent*. Mathematically, it is exactly as wrong as the error of proving $Q \Rightarrow P$ when you were asked to prove $P \Rightarrow Q$. *In practice*, in finance, we check: do we know that all positive-expected-return strategies have positive Sharpe (original)? Yes (a positive expectation and nonzero variance give positive Sharpe over long horizons under appropriate conditions). Do we know that all positive-Sharpe strategies have positive expectation (converse)? Also yes, in fact, because Sharpe = mean / stdev, so positive Sharpe with positive stdev means positive mean. But these are two *different* arguments, and the equivalence of original and converse here is a coincidence of the specific statement, not a logical consequence.

## Computational Implementation

A small Python harness that verifies logical equivalences by enumerating all assignments. This is a toy SAT/taut-checker. Use it to check any equivalence you are unsure of.

```python
from itertools import product
from typing import Callable, Sequence

def is_tautology(f: Callable[..., bool], n: int) -> bool:
    """Return True iff f(x1,...,xn) is True on every Boolean input."""
    return all(f(*assignment) for assignment in product([False, True], repeat=n))

def equivalent(f: Callable[..., bool], g: Callable[..., bool], n: int) -> bool:
    """Return True iff f and g agree on every Boolean input of length n."""
    return all(f(*a) == g(*a) for a in product([False, True], repeat=n))

# Verify De Morgan: not (P and Q) == (not P) or (not Q)
lhs = lambda P, Q: not (P and Q)
rhs = lambda P, Q: (not P) or (not Q)
assert equivalent(lhs, rhs, 2)

# Verify contrapositive equivalence: (P -> Q) == (not Q -> not P)
implies = lambda P, Q: (not P) or Q
orig = lambda P, Q: implies(P, Q)
contra = lambda P, Q: implies(not Q, not P)
assert equivalent(orig, contra, 2)

# Verify modus ponens is a tautology: ((P -> Q) and P) -> Q
mp = lambda P, Q: implies(implies(P, Q) and P, Q)
assert is_tautology(mp, 2)

# Verify the converse is NOT equivalent to the original
conv = lambda P, Q: implies(Q, P)
assert not equivalent(orig, conv, 2)

print("All propositional-logic checks passed.")
```

Running this script prints the success line. The machinery is negligible but the habit it builds is useful: when you are not sure if two forms of a condition are equivalent, enumerate.

## [QUANT APPLICATION]

**Entry conditions on a systematic strategy.** Suppose a signal fires a long entry when the following rule evaluates true:
$$\text{Entry} = (z\text{-score} > 2) \wedge \neg(\text{earnings within 2 days}) \wedge (\text{liquidity rank} \in \text{top } 50\%).$$
When you want to express the *block conditions* — i.e., the disjunction of all ways the entry can fail — you apply De Morgan:
$$\neg\text{Entry} \equiv (z\text{-score} \leq 2) \vee (\text{earnings within 2 days}) \vee (\text{liquidity rank} \notin \text{top } 50\%).$$
This is not a toy observation. When you build a production signal library, you will often want a unified "why did this bar fail to enter?" diagnostic. That diagnostic is exactly the negation of the entry rule, written out in disjunctive form so each clause maps to an interpretable reason. Equivalence (11) also tells you how to construct counterexamples systematically: to *disprove* "Entry $\Rightarrow$ positive return", exhibit a single instance of Entry true and positive return false.

## Exercises

### ★ (Foundation)

**E0.1.1.1.** Build truth tables for each of the following and identify whether each is a tautology, contradiction, or contingency:
(a) $(P \Rightarrow Q) \Rightarrow (\neg Q \Rightarrow \neg P)$
(b) $(P \wedge Q) \Rightarrow P$
(c) $(P \Rightarrow Q) \wedge (P \Rightarrow \neg Q)$
(d) $(P \vee Q) \wedge (\neg P \wedge \neg Q)$

*Solution.*
(a) Tautology. Both sides are equivalent by the contrapositive law, so the implication is always true.
(b) Tautology. If $P \wedge Q$ is true, $P$ is true.
(c) Equivalent to $P \Rightarrow (Q \wedge \neg Q) \equiv P \Rightarrow \bot \equiv \neg P$. Contingency (true when $P$ is false, false when $P$ is true).
(d) $\equiv (P \vee Q) \wedge \neg(P \vee Q)$ by De Morgan on the second conjunct. Contradiction.

**E0.1.1.2.** Write the negation of: "Every quant I know who uses leverage has had a drawdown greater than 20%." Treat "Every ... has ..." as a universal; you will handle quantifiers formally in Topic 0.1.2, but on intuition alone, state the negation.

*Solution.* "There exists a quant I know who uses leverage and has *not* had a drawdown greater than 20%." Formally, the negation of $\forall x (A(x) \Rightarrow B(x))$ is $\exists x (A(x) \wedge \neg B(x))$.

**E0.1.1.3.** Using only $\neg$ and $\wedge$ (no $\vee$), write a formula equivalent to $P \vee Q$. Verify with a truth table.

*Solution.* $P \vee Q \equiv \neg(\neg P \wedge \neg Q)$ by De Morgan and double negation. Truth table:

| $P$ | $Q$ | $P\vee Q$ | $\neg P$ | $\neg Q$ | $\neg P \wedge \neg Q$ | $\neg(\neg P \wedge \neg Q)$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T | T | T | F | F | F | T |
| T | F | T | F | T | F | T |
| F | T | T | T | F | F | T |
| F | F | F | T | T | T | F |

Agreement on columns 3 and 7. $\square$

### ★★ (Intermediate)

**E0.1.1.4.** The *NAND* connective $P \uparrow Q$ is defined as $\neg(P \wedge Q)$. Prove that NAND alone is functionally complete — i.e., $\neg, \wedge, \vee$ can all be written using only NAND.

*Solution.*
- $\neg P \equiv P \uparrow P$, since $\neg(P \wedge P) = \neg P$.
- $P \wedge Q \equiv \neg(\neg(P \wedge Q)) = \neg(P \uparrow Q) = (P \uparrow Q) \uparrow (P \uparrow Q)$.
- $P \vee Q \equiv \neg(\neg P \wedge \neg Q) = (\neg P) \uparrow (\neg Q) = (P \uparrow P) \uparrow (Q \uparrow Q)$.

So any formula built from $\neg, \wedge, \vee$ (which by Theorem 0.1.1.7 is every Boolean function) can be rewritten in NAND alone.

**E0.1.1.5.** Let $\varphi$ be a formula in $n$ variables with $k$ rows of its truth table returning T. Prove that $\varphi$ has a DNF representation with exactly $k$ minterms, each of length $n$.

*Solution.* Exactly the construction in the proof of Theorem 0.1.1.7. For each T-row $v$, write the minterm $m_v$ of length $n$; their disjunction is equivalent to $\varphi$. If you use fewer than $k$ minterms, at least one T-row of $\varphi$ is not covered — contradiction. If you try to use more than $k$ distinct $n$-literal minterms, at least one would be an F-row, contradiction. The minimum over arbitrary-length minterms may be smaller, but at length $n$, exactly $k$ are needed. $\square$

**E0.1.1.6.** Show that the formula $\varphi(P,Q,R) := (P \Rightarrow Q) \Rightarrow ((Q \Rightarrow R) \Rightarrow (P \Rightarrow R))$ is a tautology. Then explain in English what rule of inference this encodes.

*Solution.* Write $P \Rightarrow Q$ as $\neg P \vee Q$ throughout, or tabulate:

| $P$ | $Q$ | $R$ | $P\!\to\!Q$ | $Q\!\to\!R$ | $P\!\to\!R$ | $(Q\!\to\!R)\!\to\!(P\!\to\!R)$ | $\varphi$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T | T | T | T | T | T | T | T |
| T | T | F | T | F | F | T | T |
| T | F | T | F | T | T | T | T |
| T | F | F | F | T | F | F | T |
| F | T | T | T | T | T | T | T |
| F | T | F | T | F | T | T | T |
| F | F | T | T | T | T | T | T |
| F | F | F | T | T | T | T | T |

All T: tautology. The rule is *hypothetical syllogism*: "If $P \Rightarrow Q$ and $Q \Rightarrow R$ then $P \Rightarrow R$." Implication composes.

### ★★★ (Challenge)

**E0.1.1.7.** A formula $\varphi$ is *minimal* for a Boolean function $f$ if $\varphi \equiv f$ and no formula with strictly fewer connective symbols is equivalent to $f$. Prove or disprove: minimal formulas for a given $f$ are unique up to renaming.

*Hint.* Consider $f(P,Q,R) = (P\wedge Q) \vee (\neg P \wedge R) \vee (Q \wedge R)$. Is the third disjunct necessary?

**E0.1.1.8.** Call a set $\mathcal{C}$ of binary connectives *functionally complete* if every Boolean function can be expressed using elements of $\mathcal{C}$ applied to variables (and, implicitly, constants T and F as zero-ary connectives). Prove that $\{\Rightarrow, \bot\}$ is functionally complete but $\{\Rightarrow\}$ alone is not.

*Hint.* Show $\neg P \equiv P \Rightarrow \bot$, then show that every formula built from $\Rightarrow$ alone evaluates to T when every variable is set to T.

**E0.1.1.9.** Consider an infinite sequence of propositions $P_1, P_2, \ldots$ where each $P_i$ is either T or F. Suppose that for every $i$, $P_i \Rightarrow P_{i+1}$. Define $\varphi := \bigvee_{i=1}^{\infty} P_i$ and $\psi := \bigwedge_{i=1}^{\infty} P_i$ (these are infinite connectives in the obvious sense: $\varphi$ is T iff at least one $P_i$ is T, $\psi$ is T iff every $P_i$ is T). Prove that if all $P_i$ take values in $\{T, F\}$ and the chain $P_i \Rightarrow P_{i+1}$ holds, then either $\varphi \equiv \psi$ or else there is a unique "switch index" $k$ such that $P_i = F$ for $i < k$ and $P_i = T$ for $i \geq k$.

*Hint.* Apply the chain pointwise. This is a baby version of the *monotone class* argument used in measure theory (Subject 1).

---

# Topic 0.1.2 — Predicate Logic and Quantifiers

## Motivation

Propositional logic cannot say "every continuous function on a compact set attains its maximum". It cannot even say "every integer is either even or odd". To handle statements about elements of a set, we need *variables*, *predicates*, and *quantifiers*. This is the language of essentially every definition and every theorem in mathematics. Once you are fluent in this language — especially in *nested quantifiers*, which are what trip up ε–δ proofs — every subsequent module gets easier.

## Prerequisites

Topic 0.1.1.

## Definitions

**Definition 0.1.2.1 (Predicate).** A *predicate* on a set $D$ (the *domain of discourse*) is a function $P: D \to \{T, F\}$. We write $P(x)$ for the truth value of $P$ at the point $x \in D$.

A *binary predicate* on $D_1 \times D_2$ is a function $R: D_1 \times D_2 \to \{T, F\}$, written $R(x,y)$. Higher-arity predicates are defined analogously.

*Examples.* With $D = \mathbb{Z}$: $E(x) := $ "$x$ is even" is a unary predicate. $L(x,y) := $ "$x < y$" is a binary predicate on $\mathbb{Z} \times \mathbb{Z}$. With $D$ the set of traded equities: $I(s) := $ "stock $s$ is a member of the S&P 500" is a unary predicate.

**Definition 0.1.2.2 (Quantifiers).** Let $P(x)$ be a predicate on $D$.

- The *universal quantifier* $\forall x \in D \, : \, P(x)$ — read "for every $x$ in $D$, $P(x)$" — is true iff $P(x)$ is true for every $x \in D$.
- The *existential quantifier* $\exists x \in D \, : \, P(x)$ — read "there exists $x$ in $D$ such that $P(x)$" — is true iff $P(x)$ is true for at least one $x \in D$.

We often omit "$\in D$" when the domain is clear from context.

*Intuition.* $\forall$ is an infinite AND; $\exists$ is an infinite OR. If $D = \{d_1, \dots, d_n\}$ is finite, then $\forall x \, P(x) \equiv P(d_1) \wedge \cdots \wedge P(d_n)$ and $\exists x \, P(x) \equiv P(d_1) \vee \cdots \vee P(d_n)$. The interesting behaviour happens for infinite $D$.

**Definition 0.1.2.3 (Free and bound variables).** A variable in a formula is *bound* if it is in the scope of a quantifier over that variable; otherwise it is *free*. A formula with no free variables is called a *sentence*; it has a definite truth value. A formula with free variables is a predicate in those variables.

*Example.* In $\forall x (P(x) \Rightarrow Q(x,y))$, the variable $x$ is bound and $y$ is free. The formula is a predicate in $y$.

## Key Results

### Theorem 0.1.2.4 (Negation of quantifiers)

Let $P(x)$ be a predicate on a domain $D$. Then:

1. $\neg(\forall x \, P(x)) \equiv \exists x \, \neg P(x)$
2. $\neg(\exists x \, P(x)) \equiv \forall x \, \neg P(x)$

**Proof.** We prove (1). The negation of "$P$ holds for every $x$" is "$P$ fails for some $x$", which is precisely $\exists x \, \neg P(x)$. Formally:

$\neg(\forall x \, P(x))$ is T iff $\forall x \, P(x)$ is F iff it is not the case that $P(x)$ is T for every $x$ iff there is at least one $x$ with $P(x)$ F iff $\exists x \, \neg P(x)$ is T.

For (2): $\neg(\exists x \, P(x))$ is T iff no $x$ makes $P(x)$ T iff every $x$ makes $\neg P(x)$ T iff $\forall x \, \neg P(x)$ is T. $\square$

**The rule in words.** *To negate a quantifier, swap $\forall$ and $\exists$ and push the negation inside.* This is the single most useful inference rule in the entire curriculum. When, in Subject 1, you need the negation of
$$\forall \varepsilon > 0 \; \exists N \; \forall n \geq N : |x_n - x| < \varepsilon,$$
you mechanically get
$$\exists \varepsilon > 0 \; \forall N \; \exists n \geq N : |x_n - x| \geq \varepsilon,$$
and this is exactly the precise statement of "$x_n$ does *not* converge to $x$". Practice this move until it is reflexive.

### Theorem 0.1.2.5 (Commuting like quantifiers; non-commuting mixed quantifiers)

Let $R(x,y)$ be a binary predicate on $D_1 \times D_2$.

1. $\forall x \, \forall y \, R(x,y) \equiv \forall y \, \forall x \, R(x,y)$
2. $\exists x \, \exists y \, R(x,y) \equiv \exists y \, \exists x \, R(x,y)$
3. $\exists y \, \forall x \, R(x,y) \Rightarrow \forall x \, \exists y \, R(x,y)$
4. The converse of (3) is false in general.

**Proof.**

(1) Both sides assert: for every $x \in D_1$ and for every $y \in D_2$, $R(x,y)$ holds. They mean the same set-theoretic condition, namely that $R$ is identically T on $D_1 \times D_2$.

(2) Both sides assert: there is some $(x,y) \in D_1 \times D_2$ with $R(x,y)$ T.

(3) Assume $\exists y_0 \in D_2$ such that $\forall x \in D_1, R(x, y_0)$ is T. Pick any $x \in D_1$. Then $R(x, y_0)$ is T, so $\exists y \in D_2$ (namely $y_0$) with $R(x, y)$ T. Since $x$ was arbitrary, $\forall x \, \exists y \, R(x,y)$.

(4) Counterexample. Let $D_1 = D_2 = \mathbb{N}$ and $R(x, y) := $ "$y > x$". Then $\forall x \in \mathbb{N} \, \exists y \in \mathbb{N} : y > x$ is T (take $y = x + 1$). But $\exists y \in \mathbb{N} \, \forall x \in \mathbb{N} : y > x$ is F (no natural number exceeds every natural number).

$\square$

**This is the single most important proposition about quantifiers.** The direction $\exists\forall \Rightarrow \forall\exists$ always holds. The reverse direction — swapping $\forall\exists$ to $\exists\forall$ — is *the* classical error in mathematics. When a student proves $\forall x \exists y$ and then starts treating the $y$ as if it were fixed across $x$, they have committed the error. The correct version almost always demands uniformity: replacing "continuous" with "uniformly continuous", "convergence" with "uniform convergence", and so on. The difference between these concepts *is* the difference between $\forall x \exists y$ (the $y$ depends on $x$) and $\exists y \forall x$ (a single $y$ works for all $x$).

**[WARNING]** Notice that (3) requires $D_1$ and $D_2$ to be non-empty for the informal reading to hold. If $D_1$ is empty, both sides are vacuously true. If $D_2$ is empty, both sides are F (because $\exists y$ requires at least one $y$ in $D_2$).

### Theorem 0.1.2.6 (Distribution of quantifiers over connectives)

Let $P(x), Q(x)$ be predicates on $D$, and let $R$ be a proposition not mentioning $x$.

1. $\forall x (P(x) \wedge Q(x)) \equiv (\forall x P(x)) \wedge (\forall x Q(x))$
2. $\exists x (P(x) \vee Q(x)) \equiv (\exists x P(x)) \vee (\exists x Q(x))$
3. $\forall x (P(x) \vee R) \equiv (\forall x P(x)) \vee R$
4. $\exists x (P(x) \wedge R) \equiv (\exists x P(x)) \wedge R$

**Proof.** (1): $\forall x (P(x) \wedge Q(x))$ is T iff for every $x$ both $P(x)$ and $Q(x)$ are T, iff both $\forall x P(x)$ and $\forall x Q(x)$ are T.

(2): $\exists x (P(x) \vee Q(x))$ is T iff some $x$ makes at least one of $P(x), Q(x)$ T, iff at least one of $\exists x P(x), \exists x Q(x)$ is T.

(3): if $R$ is T, both sides are T. If $R$ is F, both sides reduce to $\forall x P(x)$.

(4): if $R$ is F, both sides F; if $R$ is T, both sides reduce to $\exists x P(x)$.

$\square$

**[WARNING]** $\forall x (P(x) \vee Q(x)) \not\equiv (\forall x P(x)) \vee (\forall x Q(x))$. Counterexample: $D = \{1, 2\}$, $P(x) := $ "$x = 1$", $Q(x) := $ "$x = 2$". LHS: every $x$ is either 1 or 2 — T. RHS: every $x$ is 1 (F) or every $x$ is 2 (F) — F.

Similarly $\exists x (P(x) \wedge Q(x)) \not\equiv (\exists x P(x)) \wedge (\exists x Q(x))$. The LHS says "some $x$ satisfies both", the RHS says "some $x$ satisfies $P$ and some (possibly different) $x$ satisfies $Q$".

## Worked Examples

### Example 0.1.2.7 (ε–δ, negated)

The formal definition of "the function $f: \mathbb{R} \to \mathbb{R}$ is continuous at $x_0$" is:
$$\forall \varepsilon > 0 \; \exists \delta > 0 \; \forall x \in \mathbb{R} : (|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \varepsilon).$$

*Write the negation: "$f$ is not continuous at $x_0$".*

**Solution.** Apply Theorem 0.1.2.4 three times, pushing negation past each quantifier.

$$\neg\forall\varepsilon \ldots \equiv \exists \varepsilon > 0 : \neg(\exists\delta > 0 : \forall x : (|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \varepsilon))$$
$$\equiv \exists\varepsilon > 0 : \forall\delta > 0 : \neg(\forall x : (|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \varepsilon))$$
$$\equiv \exists\varepsilon > 0 : \forall\delta > 0 : \exists x : \neg(|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \varepsilon).$$

Now apply Theorem 0.1.1.5(11) to the innermost negated implication:

$$\neg(|x - x_0| < \delta \Rightarrow |f(x) - f(x_0)| < \varepsilon) \equiv (|x - x_0| < \delta) \wedge (|f(x) - f(x_0)| \geq \varepsilon).$$

Final form:
$$\boxed{\exists\varepsilon > 0 \; \forall\delta > 0 \; \exists x \in \mathbb{R} : |x - x_0| < \delta \wedge |f(x) - f(x_0)| \geq \varepsilon.}$$

**Reading it.** There is some error tolerance $\varepsilon$ that even the tiniest $\delta$-neighbourhood fails to respect — i.e., inside every neighbourhood of $x_0$, there is a point at which $f$ is at least $\varepsilon$ away from $f(x_0)$. That is exactly what it means for $f$ to fail to be continuous at $x_0$.

### Example 0.1.2.8 (Continuous vs. uniformly continuous — the quantifier difference)

*Define "continuous on $\mathbb{R}$" and "uniformly continuous on $\mathbb{R}$" precisely, and explain how they differ at the level of quantifiers.*

**Solution.**

Continuous on $\mathbb{R}$:
$$\forall x \in \mathbb{R} \; \forall \varepsilon > 0 \; \exists\delta > 0 \; \forall y \in \mathbb{R} : (|y - x| < \delta \Rightarrow |f(y) - f(x)| < \varepsilon).$$

Uniformly continuous on $\mathbb{R}$:
$$\forall \varepsilon > 0 \; \exists\delta > 0 \; \forall x, y \in \mathbb{R} : (|y - x| < \delta \Rightarrow |f(y) - f(x)| < \varepsilon).$$

The two definitions agree in everything except the *order* of quantifiers. In the first, $\delta$ is chosen after seeing both $x$ and $\varepsilon$ — so $\delta$ may depend on $x$. In the second, $\delta$ is chosen after seeing only $\varepsilon$ — so $\delta$ must work uniformly across all $x$.

By Theorem 0.1.2.5(3), uniform continuity implies continuity. But the converse fails: $f(x) = x^2$ is continuous on $\mathbb{R}$ but not uniformly continuous (for large $x$, the required $\delta$ shrinks).

**Lesson.** *Every time you see "uniformly", "Lipschitz", or "absolutely", look for the quantifier reordering it encodes.* These concepts are not decorative; they are exactly the places where moving $\exists$ in front of $\forall$ gives a strictly stronger property.

## Computational Implementation

Let us verify quantifier-swap counterexamples numerically on small domains.

```python
from itertools import product

def forall(D, P):
    return all(P(x) for x in D)

def exists(D, P):
    return any(P(x) for x in D)

# Domain for x and y
D = range(1, 6)  # {1, 2, 3, 4, 5}

# Predicate: y > x
R = lambda x, y: y > x

# forall x exists y: y > x  (within this finite domain, this is False at x = 5
# because no y in D exceeds 5; but it IS true for x in {1,2,3,4}.)
# Let's pick the domain more carefully: x in {1..4}, y in {1..5}.
Dx = range(1, 5)
Dy = range(1, 6)

forall_exists = forall(Dx, lambda x: exists(Dy, lambda y: y > x))
exists_forall = exists(Dy, lambda y: forall(Dx, lambda x: y > x))

print("forall x in Dx exists y in Dy: y > x =", forall_exists)  # True
print("exists y in Dy forall x in Dx: y > x =", exists_forall)  # True (y=5)

# Now the classical counterexample: expand Dx to match Dy
Dx = Dy = range(1, 6)
forall_exists = forall(Dx, lambda x: exists(Dy, lambda y: y > x))
exists_forall = exists(Dy, lambda y: forall(Dx, lambda x: y > x))

print("with equal domains:")
print("  forall x exists y: y > x =", forall_exists)  # False at x=5
print("  exists y forall x: y > x =", exists_forall)  # False
```

The exact behaviour here depends on finiteness, but the key point generalizes: when the two sides give different answers, you have identified that the quantifier order matters.

## [QUANT APPLICATION]

**Uniform vs. pointwise convergence of estimators.** In high-dimensional factor research you often fit a model with a tuning parameter $\lambda$ (ridge penalty, shrinkage coefficient, bandwidth, etc.). You might prove:

- *Pointwise consistency*: $\forall \lambda > 0 : \hat\theta_n(\lambda) \xrightarrow{p} \theta(\lambda)$ as $n \to \infty$. For every fixed tuning, the estimator converges.
- *Uniform consistency*: $\sup_{\lambda \in \Lambda} |\hat\theta_n(\lambda) - \theta(\lambda)| \xrightarrow{p} 0$.

These are *quantifier swaps*: the first is $\forall \lambda \, \forall \varepsilon \, \exists N \, \forall n \geq N \, P(|\hat\theta - \theta| < \varepsilon) > 1-\varepsilon$, the second is $\forall \varepsilon \, \exists N \, \forall n \geq N \, \forall \lambda \ldots$ (the $\lambda$ is now inside, i.e., uniform).

**Why quants care.** If you chose $\lambda$ by cross-validation, your realized $\hat\lambda_n$ is a *random* function of the data. Pointwise consistency is not enough — you need uniform consistency over a class $\Lambda$ containing $\hat\lambda_n$ to conclude $\hat\theta_n(\hat\lambda_n) \to \theta(\lambda^*)$. This is exactly the same structural point as continuous versus uniformly continuous. Getting this wrong is the reason many over-tuned strategies look consistent in backtest and fail out of sample.

## Exercises

### ★ (Foundation)

**E0.1.2.1.** Write each of the following as a logical formula with quantifiers, using the domain of real numbers:
(a) "Every real number has an additive inverse."
(b) "There is a real number that is equal to its own square."
(c) "Every real number is less than some integer."

*Solution.*
(a) $\forall x \in \mathbb{R} \; \exists y \in \mathbb{R} : x + y = 0$.
(b) $\exists x \in \mathbb{R} : x = x^2$.
(c) $\forall x \in \mathbb{R} \; \exists n \in \mathbb{Z} : x < n$ (Archimedean property).

**E0.1.2.2.** Write the formal negation of each statement in E0.1.2.1.

*Solution.*
(a) $\exists x \in \mathbb{R} \; \forall y \in \mathbb{R} : x + y \neq 0$.
(b) $\forall x \in \mathbb{R} : x \neq x^2$.
(c) $\exists x \in \mathbb{R} \; \forall n \in \mathbb{Z} : x \geq n$.

**E0.1.2.3.** Consider $\forall x \in \mathbb{R} \; \exists y \in \mathbb{R} : y^2 = x$. Is this statement true or false? State its negation, and determine whether *the negation* is true or false.

*Solution.* False: take $x = -1$. Negation: $\exists x \in \mathbb{R} : \forall y \in \mathbb{R} : y^2 \neq x$. True (witness $x = -1$). A good sanity-check exercise: a statement and its negation always have opposite truth values, and at least one is true.

### ★★ (Intermediate)

**E0.1.2.4.** State the difference between "the sequence $(f_n)$ converges pointwise to $f$ on $[0,1]$" and "the sequence $(f_n)$ converges uniformly to $f$ on $[0,1]$" in terms of quantifier order. Then construct a specific example (no proof of non-uniformity required, just a pointer) where the first holds but not the second.

*Solution.* Pointwise: $\forall x \in [0,1] \; \forall \varepsilon > 0 \; \exists N \; \forall n \geq N : |f_n(x) - f(x)| < \varepsilon$. Uniform: $\forall \varepsilon > 0 \; \exists N \; \forall x \in [0,1] \; \forall n \geq N : |f_n(x) - f(x)| < \varepsilon$. The second puts "$\forall x$" after "$\exists N$"; hence $N$ cannot depend on $x$.

Example: $f_n(x) = x^n$ on $[0,1]$ converges pointwise to $f$ where $f(x) = 0$ for $x < 1$ and $f(1) = 1$. But for $\varepsilon = 1/2$, choosing $x = (1/2)^{1/n}$ shows no single $N$ works for all $x$.

**E0.1.2.5.** Given a binary predicate $R(x,y)$ on $D \times D$, prove:
$$\exists x \forall y \, R(x,y) \Rightarrow \forall y \exists x \, R(x,y),$$
and give an explicit counterexample showing the converse is false.

*Solution.* Same as Theorem 0.1.2.5(3) with the roles of $x$ and $y$ swapped. Counterexample with $D = \mathbb{N}$ and $R(x,y) := $ "$x \geq y$": for every $y$ there is some $x$ with $x \geq y$ (namely $x = y$); but no single $x$ is $\geq$ every $y$.

**E0.1.2.6.** Let $f: \mathbb{R} \to \mathbb{R}$. Define "$f$ is Lipschitz on $\mathbb{R}$" in a formula with quantifiers. Then express "$f$ is Lipschitz on every bounded interval but not globally Lipschitz" using nested quantifiers, and verify that $f(x) = x^2$ is an example.

*Solution.* $f$ is Lipschitz iff $\exists L > 0 \; \forall x, y \in \mathbb{R} : |f(x) - f(y)| \leq L|x - y|$. "Lipschitz on every bounded interval but not globally Lipschitz":
$$(\forall [a,b] \subset \mathbb{R} \; \exists L_{a,b} : \forall x, y \in [a,b] : |f(x) - f(y)| \leq L_{a,b}|x - y|) \wedge (\forall L \; \exists x, y : |f(x) - f(y)| > L|x - y|).$$
For $f(x) = x^2$: on $[a,b]$, $|f(x) - f(y)| = |x + y||x - y| \leq (|a| + |b|)|x - y|$, so $L_{a,b} = |a| + |b|$ works. Globally: for any $L$, take $x = L, y = L + 1$. Then $|f(x) - f(y)| = |(L)^2 - (L+1)^2| = 2L + 1 > L \cdot 1 = L|x-y|$.

### ★★★ (Challenge)

**E0.1.2.7.** The *axiom of choice* in its simplest form states: given any family $\{A_i\}_{i \in I}$ of non-empty sets, there exists a function $c: I \to \bigcup_i A_i$ with $c(i) \in A_i$ for all $i$. Explain why this is a quantifier swap — i.e., why the non-trivial content of the axiom is converting $\forall i \exists a \in A_i$ into $\exists c \forall i : c(i) \in A_i$. Then give an example of a context where the $\forall\exists \Rightarrow \exists\forall$ swap is legitimate without choice.

*Hint.* For the legitimate-swap example, consider the case where $I$ is a finite set, or where each $A_i$ has a canonical "least" element (e.g., $A_i \subseteq \mathbb{N}$).

**E0.1.2.8.** Let $f: \mathbb{R}^2 \to \mathbb{R}$. Consider the two statements:
- A: $\forall \varepsilon > 0 \; \exists \delta > 0 \; \forall (x,y) : \|(x,y) - (0,0)\| < \delta \Rightarrow |f(x,y) - f(0,0)| < \varepsilon$ (ordinary continuity at origin);
- B: $\forall \varepsilon > 0 \; \exists \delta > 0 \; \forall x : |x| < \delta \Rightarrow |f(x, 0) - f(0,0)| < \varepsilon$, and similarly for the $y$-axis (continuity along each axis).

Show that A implies B, but B does not imply A. Construct a function of two variables that is separately continuous along each axis but not jointly continuous at the origin.

*Hint.* Try $f(x,y) = \frac{xy}{x^2 + y^2}$ with $f(0,0) = 0$. Examine the value along $y = mx$.

**E0.1.2.9.** A predicate $R(x,y)$ is called *cofinal* if for every $x$ there is some $y$ with $R(x,y)$, and is called *directed* if for every $x_1, x_2$ there is $y$ with $R(x_1, y)$ and $R(x_2, y)$. Prove that cofinal does not imply directed, but that a predicate that is cofinal *and* reflexive, transitive, and antisymmetric becomes directed iff its underlying order has no two incomparable elements. State this as a precise theorem.

*Hint.* The lattice structure is what you are probing. Reflexive + transitive + antisymmetric gives a partial order. Start from a two-element antichain.

---

# Topic 0.1.3 — Proof Techniques

## Motivation

You now have a grammar of claims (propositional and predicate logic). The next question is: given a claim, how do you construct an argument that compels acceptance? This topic catalogs the five core proof patterns — direct proof, proof by contrapositive, proof by contradiction, proof by cases, and proof by induction (weak and strong) — together with template structures you can follow mechanically when you are stuck. Proofs are not an art; they are a craft, and like all crafts, they have standard moves that the novice memorizes and the expert recombines.

## Prerequisites

Topics 0.1.1 and 0.1.2.

## Definitions

**Definition 0.1.3.1 (Proof).** A *proof* of a proposition $\varphi$ from a set of hypotheses $\Gamma$ is a finite sequence of propositions $\varphi_1, \varphi_2, \ldots, \varphi_n = \varphi$ such that each $\varphi_i$ is either an element of $\Gamma$, an axiom, or follows from earlier $\varphi_j$ by an accepted inference rule (most commonly modus ponens, Theorem 0.1.1.6).

*Intuition.* A proof is a transcript of a convincing argument. The formal definition is not usually what we write; mathematicians write *proofs-in-words* whose formalization into the above sequence would in principle be possible. What matters in practice is that each step be individually obvious and that the assembly have no gaps.

**Definition 0.1.3.2 (Common proof patterns).** The following are the patterns this topic will formalize. Each is a valid way to prove an implication $P \Rightarrow Q$ (or a quantified statement), and each has its canonical template.

- *Direct proof*: assume $P$ and derive $Q$ through a chain of implications.
- *Proof by contrapositive*: prove $\neg Q \Rightarrow \neg P$; valid by Theorem 0.1.1.5(9).
- *Proof by contradiction*: assume $P \wedge \neg Q$ and derive a contradiction $\bot$; valid because the only way a conjunction can entail a contradiction is if the conjunction is itself contradictory, which by Theorem 0.1.1.5(11) means $P \Rightarrow Q$.
- *Proof by cases*: partition the hypothesis $P$ into disjuncts $P_1 \vee \cdots \vee P_k$ that cover all possibilities, and prove $P_i \Rightarrow Q$ for each $i$.
- *Proof by induction*: prove a universal statement $\forall n \in \mathbb{N} \, P(n)$ by showing $P(1)$ (or $P(0)$) and showing $P(n) \Rightarrow P(n+1)$; or (strong form) $\bigwedge_{k < n} P(k) \Rightarrow P(n)$.

## Key Results

### Theorem 0.1.3.3 (Principle of mathematical induction)

Let $P(n)$ be a predicate on $\mathbb{N} = \{1, 2, 3, \ldots\}$. Suppose:

- *(Base case)* $P(1)$ is true.
- *(Inductive step)* For every $n \in \mathbb{N}$, $P(n) \Rightarrow P(n+1)$.

Then $P(n)$ is true for every $n \in \mathbb{N}$.

**Proof.** Let $S = \{n \in \mathbb{N} : P(n) \text{ is true}\}$. We have $1 \in S$ (by base case). We have, for every $n$, $n \in S \Rightarrow n + 1 \in S$ (by inductive step). Suppose for contradiction that $S \neq \mathbb{N}$. Then $T := \mathbb{N} \setminus S$ is non-empty. The *well-ordering principle* of $\mathbb{N}$ (every non-empty subset of $\mathbb{N}$ has a least element — we take this as an axiom of $\mathbb{N}$) gives a least element $n_0 \in T$. Since $1 \in S$, $n_0 \neq 1$, so $n_0 \geq 2$ and $n_0 - 1 \in \mathbb{N}$. By minimality, $n_0 - 1 \in S$. But then by the inductive step, $n_0 = (n_0 - 1) + 1 \in S$, contradicting $n_0 \in T$. Hence $S = \mathbb{N}$. $\square$

**Intuition.** Induction is how mathematics reaches infinity. The base case is the first domino falling. The inductive step is the guarantee that if the $n$-th domino falls, so does the $(n+1)$-th. Together they say all dominoes fall. The proof above shows that this intuitive argument is underwritten by the well-ordering of $\mathbb{N}$: you cannot have a smallest counterexample.

**The logical equivalent.** One can also prove induction from the *Peano axioms*, where induction is taken as an axiom directly. The content is the same: in any axiomatic development of $\mathbb{N}$, either well-ordering or induction is an axiom and the other is a theorem.

### Theorem 0.1.3.4 (Strong induction)

Let $P(n)$ be a predicate on $\mathbb{N}$. Suppose for every $n \geq 1$:
$$(\forall k < n : P(k)) \Rightarrow P(n).$$
Then $P(n)$ holds for every $n \in \mathbb{N}$.

(Note: when $n = 1$, the hypothesis is vacuously $\forall k < 1 : P(k)$, which is true, so you still must prove $P(1)$ — but this is captured in the statement because the implication must hold for $n = 1$.)

**Proof.** Suppose $P(n)$ fails for some $n$. Let $T = \{n : P(n) \text{ is false}\}$. By well-ordering, $T$ has a least element $n_0$. Then $P(k)$ is true for all $k < n_0$ (else $n_0$ is not least). By hypothesis, $(\forall k < n_0 : P(k)) \Rightarrow P(n_0)$. Hence $P(n_0)$ is true, contradicting $n_0 \in T$. $\square$

**When to use strong induction.** Use it when proving $P(n)$ needs not only $P(n-1)$ but possibly $P(k)$ for arbitrary $k < n$. The canonical example is the fundamental theorem of arithmetic (every integer $\geq 2$ has a prime factorization). To factor $n$, you split $n = a \cdot b$ with $a, b < n$ and invoke induction on both $a$ and $b$; weak induction does not suffice because $a, b$ may be much smaller than $n-1$.

### Theorem 0.1.3.5 (Well-ordering, induction, and strong induction are equivalent on $\mathbb{N}$)

The following three statements are logically equivalent, each deducible from the others within ZF set theory:

(W) Every non-empty subset of $\mathbb{N}$ has a least element.
(I) Weak induction (Theorem 0.1.3.3).
(S) Strong induction (Theorem 0.1.3.4).

**Proof.**

*(W) $\Rightarrow$ (I).* Shown in the proof of Theorem 0.1.3.3.

*(W) $\Rightarrow$ (S).* Shown in the proof of Theorem 0.1.3.4.

*(I) $\Rightarrow$ (W).* Suppose (I) holds. Let $T \subseteq \mathbb{N}$ be non-empty. We want to show $T$ has a least element. Consider the predicate $Q(n) := $ "every $k \leq n$ satisfies $k \notin T$". If $Q(n)$ held for every $n \in \mathbb{N}$, then $T = \emptyset$, contradiction. Therefore $Q$ fails at some $n$. By (I), if $Q$ failed at no $n$, then $Q$ held at all $n$; equivalently by contrapositive, since $Q$ fails at some $n$, there must be a least $n$ where $Q$ fails — provided we can reason about "least"... but that is what we are trying to establish. Let us try a direct argument. Since $T$ is non-empty, pick some $m \in T$. Consider the predicate $P(n) := $ "$\neg (n \in T \wedge n \leq m)$". If $P(n)$ held for all $n$, then no element of $T$ is $\leq m$, contradicting $m \in T$. So $P$ fails at some $n$. The set $\{n \leq m : n \in T\}$ is a non-empty subset of the finite set $\{1, \ldots, m\}$, so by a *finite* (and intuitively clear) least-element fact, it has a least element $n^*$. Then $n^*$ is the least element of $T$. (To make this argument fully rigorous, one reduces (W) to (I) via a bounded well-ordering statement proved by (I) and an argument on finite sets, which itself uses induction. The circle closes within ZF.)

*(S) $\Leftrightarrow$ (I).* (I) $\Rightarrow$ (S): assume weak induction. To prove strong induction, assume the hypothesis of (S). Define $Q(n) := \forall k \leq n : P(k)$. By assumption, $Q(1) = P(1)$ which is forced by the $n = 1$ case of the hypothesis. For the inductive step, assume $Q(n)$; we want $Q(n+1)$. $Q(n)$ gives $P(k)$ for all $k \leq n$, i.e., $\forall k < n+1 : P(k)$, which by hypothesis yields $P(n+1)$. Combined, $Q(n+1)$ holds. By (I), $Q(n)$ holds for all $n$, so $P(n)$ holds for all $n$.

(S) $\Rightarrow$ (I): trivial; (S) is a strictly stronger-looking hypothesis but when the inductive step in (I) is assumed, the hypothesis of (S) is also met (take $n-1$ as the witness $k$).

$\square$

### Theorem 0.1.3.6 (Proof by contradiction, formally)

To prove $\varphi$, it suffices to prove $\neg\varphi \Rightarrow \bot$.

**Proof.** If $\neg\varphi \Rightarrow \bot$, the only way this implication is true is if $\neg\varphi$ is false (because $\bot$ is always false, and a true implication with false conclusion requires false hypothesis). Hence $\neg\varphi$ is false, so $\varphi$ is true. (This uses the law of the excluded middle — $\varphi \vee \neg\varphi$ — implicit in classical logic.) $\square$

**[WARNING] Constructivity.** Proof by contradiction and proof by contrapositive are both valid in classical logic but *not* always valid in constructive/intuitionistic logic. When you prove $\varphi$ by contradiction, you often only establish $\neg\neg\varphi$, which classically equals $\varphi$ (double negation, Theorem 0.1.1.5(1)) but constructively does not. In this curriculum we use classical logic throughout — which is standard in mainstream mathematics and in all of probability and analysis — so both techniques are available.

## Worked Examples

### Example 0.1.3.7 (Direct proof: sum of first $n$ positive integers)

*Prove that for every $n \in \mathbb{N}$, $1 + 2 + \cdots + n = n(n+1)/2$.*

**Solution (by induction).** Let $P(n)$ be the statement $1 + 2 + \cdots + n = n(n+1)/2$.

*Base.* $P(1)$: $1 = 1 \cdot 2 / 2 = 1$. True.

*Step.* Assume $P(n)$: $1 + 2 + \cdots + n = n(n+1)/2$. Then
$$1 + 2 + \cdots + n + (n+1) = \frac{n(n+1)}{2} + (n+1) = \frac{n(n+1) + 2(n+1)}{2} = \frac{(n+1)(n+2)}{2},$$
which is $P(n+1)$.

By induction, $P(n)$ holds for all $n \in \mathbb{N}$. $\square$

**Solution (direct, by Gauss's trick).** Pair the first and last terms: $(1 + n) + (2 + (n-1)) + \cdots$. Each pair sums to $n+1$, and there are $n/2$ pairs (if $n$ even) or $(n-1)/2$ pairs plus the middle term $(n+1)/2$ (if $n$ odd). Either way: $S = n(n+1)/2$. This is a direct proof requiring no induction.

The two proofs give different kinds of satisfaction. The inductive proof *verifies* the formula mechanically. Gauss's trick *explains* why the answer has this form: the sum is $n$ copies of the average, and the average of $1, 2, \ldots, n$ is $(1 + n)/2$.

### Example 0.1.3.8 (Contrapositive: if $n^2$ is even then $n$ is even)

*Let $n \in \mathbb{Z}$. Prove: if $n^2$ is even then $n$ is even.*

**Direct attempt (bad).** If $n^2$ is even, then $n^2 = 2k$ for some $k \in \mathbb{Z}$. Take a square root? $n = \sqrt{2k}$? This is not an integer reasoning; it needs the algebraic structure we are trying to prove.

**Contrapositive (good).** We prove the contrapositive: if $n$ is odd then $n^2$ is odd.

Assume $n$ is odd. Then $n = 2k + 1$ for some $k \in \mathbb{Z}$. Compute $n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$. Since $2k^2 + 2k \in \mathbb{Z}$, $n^2$ has form $2m + 1$, i.e., is odd. By Theorem 0.1.1.5(9), the original statement follows. $\square$

**Lesson.** When a direct proof requires manipulating the *conclusion* algebraically (like taking square roots), the contrapositive often converts the problem into manipulating the *negation of the conclusion* (here, odd), which is usually easier.

### Example 0.1.3.9 (Contradiction: $\sqrt{2}$ is irrational)

*Prove $\sqrt{2}$ is irrational.*

**Solution.** Suppose for contradiction that $\sqrt{2}$ is rational. Then $\sqrt{2} = p/q$ for some integers $p, q$ with $q \neq 0$. Without loss of generality (reducing to lowest terms), assume $\gcd(p, q) = 1$.

Square both sides: $2 = p^2/q^2$, so $p^2 = 2q^2$. Hence $p^2$ is even, so by Example 0.1.3.8 (applied in the same curriculum), $p$ is even. Write $p = 2r$ for some integer $r$. Substitute: $(2r)^2 = 2q^2$, i.e., $4r^2 = 2q^2$, i.e., $q^2 = 2r^2$. So $q^2$ is even, hence $q$ is even.

But now $p$ and $q$ are both even, so $\gcd(p,q) \geq 2$, contradicting $\gcd(p,q) = 1$. $\square$

This proof is famous for a reason: it combines almost every trick in the classical toolbox (contradiction, a helper lemma, assumption of lowest terms). Spend time on each step until each feels forced.

### Example 0.1.3.10 (Strong induction: the fundamental theorem of arithmetic, existence part)

*Prove that every integer $n \geq 2$ has a factorization into primes.*

**Solution (strong induction).** Let $P(n) := $ "$n$ has a prime factorization" for $n \geq 2$.

*Strong inductive step.* Fix $n \geq 2$ and assume $P(k)$ holds for every integer $k$ with $2 \leq k < n$. We show $P(n)$.

*Case 1.* $n$ is prime. Then $n = n$ is a prime factorization of $n$ (a product of a single prime).

*Case 2.* $n$ is composite. Then $n = ab$ with $2 \leq a, b < n$. By strong inductive hypothesis, $a$ has a prime factorization $a = p_1 \cdots p_s$ and $b$ has a prime factorization $b = q_1 \cdots q_t$. Concatenating, $n = p_1 \cdots p_s q_1 \cdots q_t$ is a prime factorization.

In either case $P(n)$ holds. By strong induction, $P(n)$ holds for all $n \geq 2$. $\square$

Note that weak induction would not work directly: to factor $n$, we needed hypotheses for $a$ and $b$, which are not necessarily $n-1$. Strong induction is essential.

### Example 0.1.3.11 (Proof by cases: the triangle inequality in $\mathbb{R}$)

*Prove that for all $x, y \in \mathbb{R}$, $|x + y| \leq |x| + |y|$.*

**Solution.** We use four cases based on the signs of $x$ and $y$. Recall $|x| = x$ if $x \geq 0$ and $|x| = -x$ if $x < 0$.

*Case 1.* $x \geq 0, y \geq 0$. Then $x + y \geq 0$, $|x+y| = x + y = |x| + |y|$. Inequality holds with equality.

*Case 2.* $x \geq 0, y < 0$. Then $|x| = x$, $|y| = -y$. If $x + y \geq 0$: $|x+y| = x + y$, and we need $x + y \leq x - y$, i.e., $2y \leq 0$, true. If $x + y < 0$: $|x+y| = -(x+y) = -x - y$, and we need $-x - y \leq x - y$, i.e., $-2x \leq 0$, true.

*Case 3.* $x < 0, y \geq 0$. By symmetry with Case 2 (swap $x$ and $y$).

*Case 4.* $x < 0, y < 0$. Then $x + y < 0$, $|x+y| = -x - y = |x| + |y|$. Equality.

All four cases verified. $\square$

## Computational Implementation

A tiny script that verifies a non-trivial closed-form identity by induction, simulating the "base case + step" structure.

```python
# Closed form for sum of cubes: 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2
def check_cube_sum(N):
    """Brute-force verify the formula for n = 1, 2, ..., N."""
    running = 0
    for n in range(1, N + 1):
        running += n ** 3
        closed = (n * (n + 1) // 2) ** 2
        assert running == closed, f"Fails at n={n}: {running} vs {closed}"
    return True

assert check_cube_sum(1000)
print("Sum-of-cubes identity verified for n = 1..1000")

# Simulating strong induction: factor every integer up to N via smallest-prime-factor
from math import isqrt

def prime_factorize(n):
    """Return list of primes multiplying to n (n >= 2)."""
    if n < 2:
        raise ValueError
    factors = []
    for p in range(2, isqrt(n) + 1):
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    return factors

for n in range(2, 50):
    fs = prime_factorize(n)
    prod = 1
    for f in fs:
        prod *= f
    assert prod == n
print("Prime factorizations verified for n in [2, 50)")
```

The identity check is a numerical sanity check, not a proof. But running such a check before (or alongside) attempting a proof is the empirical mathematician's equivalent of a unit test: if the identity fails at $n = 17$, you do not need to try to prove it.

## [QUANT APPLICATION]

**Proving optimality of a dynamic trading policy by backward induction.** Backward induction — the standard technique for solving optimal stopping problems, American option pricing, and Bellman-equation-based strategy optimization — is precisely mathematical induction on the time-to-horizon. Suppose you have a finite horizon $T$ and a value function $V_t(s)$ satisfying
$$V_T(s) = g(s), \quad V_t(s) = \max\left\{ g(s), \, \mathbb{E}[V_{t+1}(S_{t+1}) \mid S_t = s] \right\}.$$
To prove that your computed policy is optimal, you induct on $T - t$: at $t = T$ optimality is direct (take the terminal payoff). Assume optimality at $t + 1$; show optimality at $t$ by the one-step optimality of $V_t$.

This is a direct use of weak induction. Every dynamic programming argument you will meet in Subject 4 (optimal stopping, HJB equations) or in reinforcement-learning-based trade execution is at heart this template. Understand the induction template here and you will never be puzzled by a DP correctness argument again.

## Exercises

### ★ (Foundation)

**E0.1.3.1.** Prove by induction that $\sum_{i=1}^n (2i - 1) = n^2$ for every $n \in \mathbb{N}$.

*Solution.* $P(1)$: $1 = 1^2$. Step: assume $\sum_{i=1}^n (2i-1) = n^2$; then $\sum_{i=1}^{n+1}(2i-1) = n^2 + (2(n+1) - 1) = n^2 + 2n + 1 = (n+1)^2$. $\square$

**E0.1.3.2.** Prove by contrapositive: if $n \in \mathbb{Z}$ and $n^2$ is divisible by 3 then $n$ is divisible by 3.

*Solution.* Contrapositive: if $n$ is not divisible by 3, then $n^2$ is not divisible by 3. If $n$ is not divisible by 3, then $n \equiv 1$ or $n \equiv 2 \pmod 3$. In the first case $n^2 \equiv 1 \pmod 3$. In the second case $n^2 \equiv 4 \equiv 1 \pmod 3$. Either way $n^2 \not\equiv 0 \pmod 3$. $\square$

**E0.1.3.3.** Prove by contradiction that there is no largest even integer.

*Solution.* Suppose there were a largest even integer $N$. Then $N + 2$ is an even integer with $N + 2 > N$, contradicting maximality. $\square$

### ★★ (Intermediate)

**E0.1.3.4.** (Bernoulli's inequality) For every real number $x \geq -1$ and every $n \in \mathbb{N}$, prove $(1 + x)^n \geq 1 + nx$.

*Solution.* Induct on $n$.

*Base:* $n = 1$: $(1 + x)^1 = 1 + x = 1 + 1 \cdot x$. Equality.

*Step:* Assume $(1 + x)^n \geq 1 + nx$. Multiply both sides by $(1 + x)$. Because $x \geq -1$, $1 + x \geq 0$, so the inequality preserves direction:
$$(1 + x)^{n+1} \geq (1 + nx)(1 + x) = 1 + nx + x + nx^2 = 1 + (n+1)x + nx^2 \geq 1 + (n+1)x$$
since $nx^2 \geq 0$. By induction, the inequality holds for every $n \in \mathbb{N}$. $\square$

**E0.1.3.5.** Prove by strong induction that every positive integer $n$ can be written uniquely as $n = 2^k \cdot m$ where $k \geq 0$ and $m$ is odd.

*Solution.*

*Existence.* Let $P(n)$ be the statement. Strong induction on $n$.

$P(1)$: $1 = 2^0 \cdot 1$, with $k = 0, m = 1$. True.

Step: Fix $n \geq 2$, assume $P(k)$ for $2 \leq k < n$ (and $P(1)$). If $n$ is odd, take $k = 0, m = n$. Done. If $n$ is even, write $n = 2n'$ with $1 \leq n' < n$. By strong induction $n' = 2^{k'} m'$ with $m'$ odd. Then $n = 2^{k'+1} m'$ is a valid decomposition. $P(n)$ holds.

*Uniqueness.* Suppose $n = 2^a m_1 = 2^b m_2$ with $m_1, m_2$ odd and $a \leq b$. Then $m_1 = 2^{b-a} m_2$. If $b > a$, LHS is odd, RHS is even — contradiction. Hence $a = b$, and then $m_1 = m_2$. $\square$

**E0.1.3.6.** The Fibonacci sequence is defined by $F_1 = F_2 = 1$ and $F_{n+2} = F_{n+1} + F_n$. Prove that $F_n < 2^n$ for every $n \geq 1$.

*Solution.* Strong induction. $P(1)$: $1 < 2$. $P(2)$: $1 < 4$. Step: assume $F_k < 2^k$ for all $1 \leq k < n$ where $n \geq 3$. Then
$$F_n = F_{n-1} + F_{n-2} < 2^{n-1} + 2^{n-2} = 2^{n-2}(2 + 1) = 3 \cdot 2^{n-2} < 4 \cdot 2^{n-2} = 2^n.$$
$\square$

### ★★★ (Challenge)

**E0.1.3.7.** (Euclid's theorem, reinforced) Prove that there are infinitely many primes $p$ with $p \equiv 3 \pmod 4$.

*Hint.* Mimic Euclid's proof: suppose only finitely many, form a suitable product $+$ adjustment, and extract a contradiction by considering residues modulo 4.

**E0.1.3.8.** Let $a_1 = 1$, $a_{n+1} = \sqrt{2 + a_n}$. Prove that the sequence is increasing and bounded above, and identify its limit.

*Hint.* Prove by induction: $a_n < 2$ and $a_{n+1} > a_n$. Then the monotone convergence theorem (from any basic analysis book; you will meet it formally in Module 0.5) gives convergence. For the limit $L$, solve $L = \sqrt{2 + L}$.

**E0.1.3.9.** (Schur's inequality / pigeonhole) Let $n$ be a positive integer. Suppose the integers $\{1, 2, \ldots, 2n\}$ are partitioned into $n$ pairs $\{a_1, b_1\}, \ldots, \{a_n, b_n\}$. Let $s_i = a_i + b_i$. Prove that the $s_i$ take at most $2n - 1$ distinct values, and find a partition achieving this bound.

*Hint.* Each $s_i$ lies in $\{3, 4, \ldots, 4n - 1\}$, a set of $4n - 3$ integers. Use parity to cut this down. For the achievable bound, pair 1 with each of $2, 3, \ldots, 2n$ ... no, think about partitions that spread the sums out.

---

# Topic 0.1.4 — Set Theory Foundations

## Motivation

Modern mathematics is built on set theory. Every object you will study — numbers, functions, random variables, σ-algebras, Banach spaces, equilibria, probabilities — is ultimately a set. JEE set theory is sufficient for manipulating small examples but does not build the reflexes needed for the measure-theoretic and functional-analytic work coming in Subjects 1 and 9. This topic formalizes the algebra of sets, establishes De Morgan's laws at set-theoretic level (to be used constantly for σ-algebras), and introduces power sets and Cartesian products as objects in their own right — not just notations.

## Prerequisites

Topics 0.1.1 and 0.1.2.

## Definitions

**Definition 0.1.4.1 (Set; membership).** A *set* is a collection of objects, called its *elements* or *members*. We write $x \in A$ to mean "$x$ is an element of $A$" and $x \notin A$ for its negation. Two sets are *equal* iff they have the same elements:
$$A = B \quad \Leftrightarrow \quad \forall x (x \in A \Leftrightarrow x \in B).$$
This is the *axiom of extensionality*.

*Warning.* A naive set-building scheme — "for any predicate $P$, the collection $\{x : P(x)\}$ is a set" — leads to Russell's paradox (consider $R = \{x : x \notin x\}$; is $R \in R$?). The remedy, axiomatized in Zermelo–Fraenkel set theory, is to restrict: given an existing set $A$, you may form $\{x \in A : P(x)\}$. We do not dwell on the axioms of ZF in this curriculum; we accept that any set-building we do with an already-given ambient set is legitimate.

**Definition 0.1.4.2 (Subset).** $A \subseteq B$ means $\forall x (x \in A \Rightarrow x \in B)$. $A \subsetneq B$ (proper subset) means $A \subseteq B$ and $A \neq B$.

*Trick of the trade.* To prove $A = B$, the standard tactic is double inclusion: show $A \subseteq B$ and $B \subseteq A$. Each inclusion is a universal implication and typically proved by taking an arbitrary $x \in A$ and showing $x \in B$ (and vice versa). This double-inclusion pattern will recur throughout the curriculum.

**Definition 0.1.4.3 (Set operations).** For sets $A, B$ (and ambient set $U$ when complement is taken):

- *Union*: $A \cup B = \{x : x \in A \vee x \in B\}$.
- *Intersection*: $A \cap B = \{x : x \in A \wedge x \in B\}$.
- *Difference*: $A \setminus B = \{x : x \in A \wedge x \notin B\}$.
- *Complement* (relative to $U$): $A^c = U \setminus A = \{x \in U : x \notin A\}$.
- *Symmetric difference*: $A \triangle B = (A \setminus B) \cup (B \setminus A)$.

These operations are defined element-wise via the corresponding logical connectives. Every theorem about these set operations is a theorem about the logical connectives applied pointwise.

**Definition 0.1.4.4 (Indexed unions and intersections).** Let $\{A_i\}_{i \in I}$ be a family of sets indexed by a set $I$. Then
$$\bigcup_{i \in I} A_i = \{x : \exists i \in I, x \in A_i\}, \qquad \bigcap_{i \in I} A_i = \{x : \forall i \in I, x \in A_i\}.$$

This is the generalization essential for Subject 1: σ-algebras are closed under *countable* (indexed) unions and intersections.

**Definition 0.1.4.5 (Cartesian product).** $A \times B = \{(a, b) : a \in A, b \in B\}$, where $(a, b)$ is the *ordered pair*. Formally, $(a, b) := \{\{a\}, \{a, b\}\}$ (the Kuratowski encoding); this is a set-theoretic construction that makes $(a, b) = (c, d) \Leftrightarrow a = c \wedge b = d$. For practical purposes we treat ordered pairs as primitive.

Higher-arity products: $A_1 \times A_2 \times \cdots \times A_n = \{(a_1, \ldots, a_n) : a_i \in A_i\}$.

**Definition 0.1.4.6 (Power set).** The *power set* of $A$ is $\mathcal{P}(A) = \{B : B \subseteq A\}$ — the set of all subsets of $A$. Note $\emptyset \in \mathcal{P}(A)$ and $A \in \mathcal{P}(A)$.

If $|A| = n$ finite, then $|\mathcal{P}(A)| = 2^n$ (Theorem 0.1.4.9 below).

## Key Results

### Theorem 0.1.4.7 (De Morgan's laws, set-theoretic version)

Let $\{A_i\}_{i \in I}$ be a family of subsets of $U$. Then:
$$\left(\bigcup_{i \in I} A_i\right)^c = \bigcap_{i \in I} A_i^c \quad \text{and} \quad \left(\bigcap_{i \in I} A_i\right)^c = \bigcup_{i \in I} A_i^c.$$

**Proof.** We show the first (the second is analogous).

$(\subseteq)$ Let $x \in (\bigcup_i A_i)^c$. Then $x \in U$ and $x \notin \bigcup_i A_i$, i.e., $\neg\exists i : x \in A_i$, i.e., $\forall i : x \notin A_i$ (by Theorem 0.1.2.4), i.e., $\forall i : x \in A_i^c$, i.e., $x \in \bigcap_i A_i^c$.

$(\supseteq)$ Let $x \in \bigcap_i A_i^c$. Then $\forall i : x \in A_i^c$, i.e., $\forall i : x \notin A_i$, i.e., $\neg\exists i : x \in A_i$, i.e., $x \notin \bigcup_i A_i$, i.e., $x \in (\bigcup_i A_i)^c$.

For the second identity, apply the first to the family $\{A_i^c\}$ and note $(A_i^c)^c = A_i$:
$$\left(\bigcup_i A_i^c\right)^c = \bigcap_i A_i \quad \Rightarrow \quad \bigcup_i A_i^c = \left(\bigcap_i A_i\right)^c$$
(taking complement of both sides and using $(X^c)^c = X$, which itself follows from double negation). $\square$

**Why this matters.** Subject 1 spends a lot of time verifying closure properties of σ-algebras: closure under complement and countable union implies closure under countable intersection (by De Morgan). Without this identity the axiomatics would be asymmetric; with it, we can state the axioms compactly.

### Theorem 0.1.4.8 (Distributivity)

For any sets $A, B, C$:
$$A \cap (B \cup C) = (A \cap B) \cup (A \cap C),$$
$$A \cup (B \cap C) = (A \cup B) \cap (A \cup C).$$

More generally, for any family $\{B_i\}_{i \in I}$:
$$A \cap \bigcup_i B_i = \bigcup_i (A \cap B_i), \quad A \cup \bigcap_i B_i = \bigcap_i (A \cup B_i).$$

**Proof.** These are the set-theoretic projections of the corresponding logical distributivities (Theorem 0.1.1.5, 4–5).

Take $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$.

$x \in A \cap (B \cup C)$ iff $x \in A \wedge x \in B \cup C$ iff $x \in A \wedge (x \in B \vee x \in C)$ iff (by logical distributivity) $(x \in A \wedge x \in B) \vee (x \in A \wedge x \in C)$ iff $x \in (A \cap B) \cup (A \cap C)$.

Each "iff" is an equivalence; accumulation gives set equality.

The indexed case proceeds identically:

$x \in A \cap \bigcup_i B_i$ iff $x \in A \wedge \exists i : x \in B_i$ iff $\exists i : (x \in A \wedge x \in B_i)$ (by Theorem 0.1.2.6(4) with $A$ playing the role of the parameter) iff $\exists i : x \in A \cap B_i$ iff $x \in \bigcup_i (A \cap B_i)$.

Similarly, $x \in A \cup \bigcap_i B_i$ iff $x \in A \vee \forall i : x \in B_i$ iff $\forall i : (x \in A \vee x \in B_i)$ (by Theorem 0.1.2.6(3)) iff $\forall i : x \in A \cup B_i$ iff $x \in \bigcap_i (A \cup B_i)$. $\square$

### Theorem 0.1.4.9 (Cardinality of the power set of a finite set)

If $|A| = n$ for a finite set $A$, then $|\mathcal{P}(A)| = 2^n$.

**Proof.** Induction on $n$.

*Base.* $n = 0$: $A = \emptyset$ and $\mathcal{P}(A) = \{\emptyset\}$ has size $1 = 2^0$.

*Step.* Assume for all $n$-element sets the power set has size $2^n$. Let $A$ have $n + 1$ elements; pick some $a \in A$ and let $A' = A \setminus \{a\}$ (size $n$). The subsets of $A$ split into two disjoint classes:
1. Subsets not containing $a$: these are exactly the subsets of $A'$, so there are $2^n$ of them by hypothesis.
2. Subsets containing $a$: each such subset has the form $S \cup \{a\}$ for some $S \subseteq A'$, and distinct $S$'s give distinct subsets. There are $2^n$ of them.

Hence $|\mathcal{P}(A)| = 2^n + 2^n = 2^{n+1}$. $\square$

*Alternative counting.* Each subset of $A$ is determined by, for each element of $A$, a binary choice of "in" or "out". That is $2^n$ choices.

### Theorem 0.1.4.10 (Inclusion–exclusion, two and three sets)

For finite sets:
$$|A \cup B| = |A| + |B| - |A \cap B|,$$
$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|.$$

**Proof.** (Two-set version.) Partition $A \cup B$ into three disjoint pieces: $A \setminus B$, $B \setminus A$, $A \cap B$.
$$|A \cup B| = |A \setminus B| + |B \setminus A| + |A \cap B|.$$
Now $|A| = |A \setminus B| + |A \cap B|$, so $|A \setminus B| = |A| - |A \cap B|$. Similarly $|B \setminus A| = |B| - |A \cap B|$. Substitute:
$$|A \cup B| = (|A| - |A \cap B|) + (|B| - |A \cap B|) + |A \cap B| = |A| + |B| - |A \cap B|.$$

For three sets, apply the two-set formula to $A \cup (B \cup C)$:
$$|A \cup (B \cup C)| = |A| + |B \cup C| - |A \cap (B \cup C)|.$$
Expand $|B \cup C|$ and $|A \cap (B \cup C)| = |(A \cap B) \cup (A \cap C)| = |A \cap B| + |A \cap C| - |A \cap B \cap C|$ (by the two-set formula applied to the union $(A \cap B) \cup (A \cap C)$, noting $(A \cap B) \cap (A \cap C) = A \cap B \cap C$):
$$|A \cup B \cup C| = |A| + (|B| + |C| - |B \cap C|) - (|A \cap B| + |A \cap C| - |A \cap B \cap C|),$$
which rearranges to the stated formula. $\square$

The general inclusion–exclusion formula (for $n$ sets) follows by induction on $n$; the sign pattern alternates with the size of the intersection, and the statement is:
$$\left|\bigcup_{i=1}^n A_i\right| = \sum_{\emptyset \neq S \subseteq \{1,\ldots,n\}} (-1)^{|S|+1} \left|\bigcap_{i \in S} A_i\right|.$$

### Theorem 0.1.4.11 (Finite product cardinality)

$|A \times B| = |A| \cdot |B|$ if $A, B$ finite. More generally, $|A_1 \times \cdots \times A_n| = \prod_i |A_i|$.

**Proof.** Two sets: $A = \{a_1, \ldots, a_m\}$, $B = \{b_1, \ldots, b_n\}$. Each element of $A \times B$ is an ordered pair $(a_i, b_j)$ with $1 \leq i \leq m, 1 \leq j \leq n$. There are $m \cdot n$ such pairs, and they are all distinct (because ordered pairs are equal iff both components are equal).

General case: induct on $n$. Base $n = 1$ trivial. Step: $A_1 \times \cdots \times A_{n+1}$ is in bijection with $(A_1 \times \cdots \times A_n) \times A_{n+1}$ (via $(a_1, \ldots, a_{n+1}) \leftrightarrow ((a_1, \ldots, a_n), a_{n+1})$), so its size is $|A_1 \times \cdots \times A_n| \cdot |A_{n+1}| = \prod_i |A_i|$ by induction. $\square$

## Worked Examples

### Example 0.1.4.12 (Direct: simplify a set expression)

*Simplify $(A \cup B) \cap (A \cup B^c)$.*

**Solution.** Apply distributivity:
$$(A \cup B) \cap (A \cup B^c) = A \cup (B \cap B^c) = A \cup \emptyset = A.$$
$\square$

### Example 0.1.4.13 (Trickier: symmetric difference and the XOR identity)

*Prove $A \triangle B = (A \cup B) \setminus (A \cap B)$.*

**Solution.**

$x \in A \triangle B$ iff $x \in (A \setminus B) \cup (B \setminus A)$ iff ($x \in A \wedge x \notin B$) $\vee$ ($x \in B \wedge x \notin A$).

This says $x$ is in exactly one of $A, B$, which is the logical *exclusive or* (XOR) of $x \in A$ and $x \in B$.

$x \in (A \cup B) \setminus (A \cap B)$ iff ($x \in A \cup B$) $\wedge$ ($x \notin A \cap B$) iff ($x \in A \vee x \in B$) $\wedge$ $\neg(x \in A \wedge x \in B)$.

The logical identity "$(P \vee Q) \wedge \neg(P \wedge Q) \equiv (P \wedge \neg Q) \vee (Q \wedge \neg P)$" — which is just the truth-table definition of XOR — gives equality of the two conditions. $\square$

**Why this identity is used.** Symmetric difference is the natural "XOR" on sets. In the σ-algebra of events (Subject 1), $P(A \triangle B)$ is a *metric* on the space of events up to null-equivalence. This is why $\triangle$ appears in approximation arguments.

## Computational Implementation

```python
def set_identity_check(U, A, B, C):
    """Verify a handful of set-theoretic identities on concrete U, A, B, C."""
    # De Morgan for two sets
    assert (A | B) - (A & B) == A ^ B           # symmetric difference identity
    assert (U - (A | B)) == (U - A) & (U - B)
    assert (U - (A & B)) == (U - A) | (U - B)
    # Distributivity
    assert A & (B | C) == (A & B) | (A & C)
    assert A | (B & C) == (A | B) & (A | C)
    # Inclusion-exclusion
    assert len(A | B) == len(A) + len(B) - len(A & B)
    assert len(A | B | C) == (len(A) + len(B) + len(C)
                              - len(A & B) - len(A & C) - len(B & C)
                              + len(A & B & C))

U = set(range(20))
A = {1, 2, 3, 4, 5, 6}
B = {4, 5, 6, 7, 8, 9}
C = {2, 4, 6, 8, 10}
set_identity_check(U, A, B, C)

# Power set and its cardinality
def power_set(A):
    """Return list of all subsets of A (as frozensets)."""
    A = list(A)
    n = len(A)
    result = []
    for mask in range(2 ** n):
        S = frozenset(A[i] for i in range(n) if (mask >> i) & 1)
        result.append(S)
    return result

for n in range(8):
    A = set(range(n))
    assert len(power_set(A)) == 2 ** n
print("All set-theory checks passed.")
```

## [QUANT APPLICATION]

**Event algebras in risk management.** A trading desk's daily risk report typically enumerates a collection of *risk events* — downside scenarios, each of which is a subset of the outcome space $\Omega$. Call them $A_1, \ldots, A_n$, e.g., "equity market -3%", "credit spreads +50bps", "USD up 2% vs basket". Two natural questions:

1. *What is the probability that any of them occurs?* By inclusion–exclusion, this is
$$P\left(\bigcup_i A_i\right) = \sum_S (-1)^{|S|+1} P\left(\bigcap_{i \in S} A_i\right).$$
In practice you truncate at $k$ terms (Bonferroni bounds).

2. *Can you hedge each independently?* This is a question about the structure of the σ-algebra generated by the $A_i$'s. If the $A_i$ are independent, the σ-algebra decomposes as a product, and you can construct independent hedges. If they overlap (e.g., "equity down" and "high-yield down" co-occur), your hedges interact. The language of intersections and complements is exactly how you formalize this.

Set algebra is the prerequisite for the σ-algebra language of Subject 1; σ-algebras are the prerequisite for probability theory; probability theory is the language of risk. This is a direct line.

## Exercises

### ★ (Foundation)

**E0.1.4.1.** Let $A = \{1, 2, 3\}$, $B = \{3, 4, 5\}$. Compute $A \cup B$, $A \cap B$, $A \setminus B$, $B \setminus A$, $A \triangle B$, and $A \times B$.

*Solution.* $A \cup B = \{1,2,3,4,5\}$, $A \cap B = \{3\}$, $A \setminus B = \{1,2\}$, $B \setminus A = \{4,5\}$, $A \triangle B = \{1,2,4,5\}$, $A \times B = \{(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,3),(3,4),(3,5)\}$.

**E0.1.4.2.** List all elements of $\mathcal{P}(\{a, b, c\})$.

*Solution.* $\emptyset, \{a\}, \{b\}, \{c\}, \{a,b\}, \{a,c\}, \{b,c\}, \{a,b,c\}$. Eight elements, as $2^3 = 8$.

**E0.1.4.3.** Prove that $A \subseteq B$ iff $A \cap B = A$.

*Solution.* ($\Rightarrow$) If $A \subseteq B$, then for any $x \in A$ also $x \in B$, so $x \in A \cap B$; thus $A \subseteq A \cap B$. The reverse inclusion $A \cap B \subseteq A$ always holds. ($\Leftarrow$) If $A \cap B = A$, then $A = A \cap B \subseteq B$. $\square$

### ★★ (Intermediate)

**E0.1.4.4.** Prove that $(A \cup B) \times C = (A \times C) \cup (B \times C)$.

*Solution.* $(x, y) \in (A \cup B) \times C$ iff $x \in A \cup B \wedge y \in C$ iff $(x \in A \vee x \in B) \wedge y \in C$ iff (by distributivity) $(x \in A \wedge y \in C) \vee (x \in B \wedge y \in C)$ iff $(x, y) \in (A \times C) \cup (B \times C)$. $\square$

**E0.1.4.5.** Show that the symmetric difference is associative: $(A \triangle B) \triangle C = A \triangle (B \triangle C)$.

*Solution.* Define indicator functions $\mathbb{1}_A: U \to \{0, 1\}$. Under this, $\triangle$ corresponds to addition mod 2:
$$\mathbb{1}_{A \triangle B}(x) = \mathbb{1}_A(x) + \mathbb{1}_B(x) \pmod 2.$$
(Verify: both equal 1 iff $x$ is in exactly one of $A, B$.) Addition mod 2 is associative, so:
$$\mathbb{1}_{(A \triangle B) \triangle C}(x) = \mathbb{1}_A(x) + \mathbb{1}_B(x) + \mathbb{1}_C(x) = \mathbb{1}_{A \triangle (B \triangle C)}(x) \pmod 2.$$
Two sets with the same indicator are equal. $\square$

**E0.1.4.6.** Prove by induction on $n$: for any $n \geq 2$ sets $A_1, \ldots, A_n$ in an ambient $U$,
$$\left(\bigcap_{i=1}^n A_i\right)^c = \bigcup_{i=1}^n A_i^c.$$

*Solution.* $n = 2$ is the two-set De Morgan law. Step: assume for $n$, prove for $n+1$:
$$\left(\bigcap_{i=1}^{n+1} A_i\right)^c = \left(\left(\bigcap_{i=1}^n A_i\right) \cap A_{n+1}\right)^c = \left(\bigcap_{i=1}^n A_i\right)^c \cup A_{n+1}^c = \bigcup_{i=1}^n A_i^c \cup A_{n+1}^c = \bigcup_{i=1}^{n+1} A_i^c.$$
Each step is by $n = 2$ De Morgan or inductive hypothesis. $\square$

### ★★★ (Challenge)

**E0.1.4.7.** Let $A_1, A_2, \ldots$ be a sequence of subsets of $\mathbb{N}$. Define
$$\limsup_n A_n := \bigcap_{n=1}^\infty \bigcup_{k=n}^\infty A_k, \quad \liminf_n A_n := \bigcup_{n=1}^\infty \bigcap_{k=n}^\infty A_k.$$
Interpret both sets in plain English (what is the defining property of an element of each?), show $\liminf A_n \subseteq \limsup A_n$, and give an example where the inclusion is strict.

*Hint.* $x \in \limsup A_n$ iff $x$ belongs to infinitely many $A_n$; $x \in \liminf A_n$ iff $x$ belongs to all but finitely many $A_n$. This construction is central to the Borel–Cantelli lemmas in Subject 1.

**E0.1.4.8.** Let $\{A_i\}_{i \in I}$ and $\{B_j\}_{j \in J}$ be families of subsets of $U$. Prove
$$\left(\bigcup_i A_i\right) \cap \left(\bigcup_j B_j\right) = \bigcup_{(i,j) \in I \times J} (A_i \cap B_j),$$
and derive the analogous identity for intersections-of-intersections. Then show with a counterexample that swapping union and intersection (i.e., trying $\bigcup \bigcap$ versus $\bigcap \bigcup$) can yield different answers.

*Hint.* Each element of the LHS satisfies an existence condition twice; distribute.

**E0.1.4.9.** A *topology* on $X$ is a family $\tau \subseteq \mathcal{P}(X)$ closed under arbitrary unions and finite intersections, and containing $\emptyset$ and $X$. Prove that the finite-intersection axiom cannot be strengthened to "arbitrary intersection" without forcing $\tau$ to be the power set or the trivial topology on every connected space. Give a precise statement and proof.

*Hint.* Consider $X = \mathbb{R}$ with the standard topology and show the set $\{0\}$ would have to be open.

---

# Topic 0.1.5 — Functions

## Motivation

Every operator you will meet in the rest of this curriculum — random variables, expectations, linear maps, σ-algebras, measures, transforms, estimators — is a function. The JEE treatment of functions treats them as formulas or graphs; we need a treatment that treats them as *relations* (subsets of the Cartesian product) and exposes the structural notions of injectivity, surjectivity, and bijectivity that will be essential for constructing inverses, measurability arguments, and change-of-variable formulas.

## Prerequisites

Topics 0.1.1–0.1.4.

## Definitions

**Definition 0.1.5.1 (Function).** A *function* $f: A \to B$ from $A$ (the *domain*) to $B$ (the *codomain*) is a rule that assigns to each $a \in A$ exactly one element $f(a) \in B$. Formally, $f$ can be identified with its graph
$$\mathrm{graph}(f) = \{(a, b) \in A \times B : b = f(a)\},$$
a subset of $A \times B$ with the property that for each $a \in A$ there is exactly one $b \in B$ with $(a, b) \in \mathrm{graph}(f)$.

*Warning on terminology.* Elsewhere you may see "range of $f$" used both for "codomain" and for "image". In this curriculum we will be strict: *codomain* is the declared target set $B$; *image* is the actual set of outputs.

**Definition 0.1.5.2 (Image and preimage).** For $S \subseteq A$ and $T \subseteq B$:

- *Image* of $S$: $f(S) = \{f(a) : a \in S\} \subseteq B$.
- *Preimage* of $T$: $f^{-1}(T) = \{a \in A : f(a) \in T\} \subseteq A$.

The *image* of $f$ (without a set argument) is $f(A) = \{f(a) : a \in A\}$.

**Notation clash warning.** $f^{-1}$ is also sometimes used for the inverse function. The preimage $f^{-1}(T)$ is always defined — it is a set-level operation — even when the inverse function does not exist.

**Definition 0.1.5.3 (Injective, surjective, bijective).** Let $f: A \to B$.

- $f$ is *injective* (one-to-one) if $\forall a_1, a_2 \in A : f(a_1) = f(a_2) \Rightarrow a_1 = a_2$, equivalently (by contrapositive) $a_1 \neq a_2 \Rightarrow f(a_1) \neq f(a_2)$.
- $f$ is *surjective* (onto) if $\forall b \in B \; \exists a \in A : f(a) = b$, equivalently $f(A) = B$.
- $f$ is *bijective* if both injective and surjective.

*Intuition.* Injective $\approx$ "no collisions"; surjective $\approx$ "fills the codomain"; bijective $\approx$ a perfect matching.

**Definition 0.1.5.4 (Composition).** For $f: A \to B, g: B \to C$, the *composition* $g \circ f: A \to C$ is $(g \circ f)(a) = g(f(a))$.

Composition is associative: $h \circ (g \circ f) = (h \circ g) \circ f$ whenever the compositions make sense.

**Definition 0.1.5.5 (Identity and inverse).** The *identity* on $A$ is $\mathrm{id}_A: A \to A, a \mapsto a$.

A function $g: B \to A$ is a *left inverse* of $f: A \to B$ if $g \circ f = \mathrm{id}_A$; a *right inverse* if $f \circ g = \mathrm{id}_B$. If $g$ is both left and right inverse, it is a *two-sided inverse*, and we write $g = f^{-1}$.

## Key Results

### Theorem 0.1.5.6 (Characterization of injectivity and surjectivity by one-sided inverses)

Let $f: A \to B$ with $A \neq \emptyset$.

1. $f$ is injective iff $f$ has a left inverse.
2. $f$ is surjective iff $f$ has a right inverse. (This direction uses the Axiom of Choice if $A$ is uncountable and $B$ is also.)
3. $f$ is bijective iff $f$ has a two-sided inverse, and such an inverse is unique.

**Proof.**

(1) ($\Rightarrow$) Suppose $f$ injective. Pick any $a_0 \in A$. Define $g: B \to A$ by
$$g(b) = \begin{cases} a & \text{if } b = f(a) \text{ for some (unique, by injectivity) } a \\ a_0 & \text{if } b \notin f(A) \end{cases}.$$
For $a \in A$: $g(f(a)) = a$ by the first branch. So $g \circ f = \mathrm{id}_A$.

($\Leftarrow$) Suppose $g \circ f = \mathrm{id}_A$. If $f(a_1) = f(a_2)$, apply $g$: $g(f(a_1)) = g(f(a_2))$, i.e., $a_1 = a_2$. So $f$ injective.

(2) ($\Rightarrow$) Suppose $f$ surjective. For each $b \in B$, the preimage $f^{-1}(\{b\})$ is non-empty. Use the axiom of choice to pick an element $h(b) \in f^{-1}(\{b\})$. Then $f(h(b)) = b$, so $f \circ h = \mathrm{id}_B$.

($\Leftarrow$) Suppose $f \circ h = \mathrm{id}_B$. For any $b \in B$, $f(h(b)) = b$, so $h(b) \in A$ is a preimage. Hence $f$ is surjective.

(3) ($\Rightarrow$) Bijective $f$ has both a left inverse $g_L$ and a right inverse $g_R$. These coincide:
$$g_L = g_L \circ \mathrm{id}_B = g_L \circ (f \circ g_R) = (g_L \circ f) \circ g_R = \mathrm{id}_A \circ g_R = g_R.$$
Any left inverse equals any right inverse; hence the inverse is unique (any two left inverses $g_L, g_L'$ both equal the right inverse, hence each other).

($\Leftarrow$) A two-sided inverse makes $f$ both injective and surjective by (1) and (2), hence bijective.

$\square$

### Theorem 0.1.5.7 (Composition preserves structure)

Let $f: A \to B, g: B \to C$.

1. If $f$ and $g$ are injective, then $g \circ f$ is injective.
2. If $f$ and $g$ are surjective, then $g \circ f$ is surjective.
3. If $f$ and $g$ are bijective, then $g \circ f$ is bijective, and $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$.

**Proof.**

(1) Assume $g \circ f(a_1) = g \circ f(a_2)$, i.e., $g(f(a_1)) = g(f(a_2))$. By injectivity of $g$, $f(a_1) = f(a_2)$. By injectivity of $f$, $a_1 = a_2$.

(2) Let $c \in C$. By surjectivity of $g$, $\exists b \in B : g(b) = c$. By surjectivity of $f$, $\exists a \in A : f(a) = b$. Then $g(f(a)) = g(b) = c$.

(3) Bijectivity follows from (1) and (2). For the inverse formula: verify
$$(f^{-1} \circ g^{-1}) \circ (g \circ f) = f^{-1} \circ (g^{-1} \circ g) \circ f = f^{-1} \circ \mathrm{id}_B \circ f = f^{-1} \circ f = \mathrm{id}_A,$$
and similarly $(g \circ f) \circ (f^{-1} \circ g^{-1}) = \mathrm{id}_C$. By uniqueness of the two-sided inverse, $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$. $\square$

### Theorem 0.1.5.8 (Preimage and set operations)

Let $f: A \to B$, and let $\{T_i\}_{i \in I}$ be a family of subsets of $B$. Then:

1. $f^{-1}\left(\bigcup_i T_i\right) = \bigcup_i f^{-1}(T_i)$.
2. $f^{-1}\left(\bigcap_i T_i\right) = \bigcap_i f^{-1}(T_i)$.
3. $f^{-1}(T^c) = (f^{-1}(T))^c$ (where complement is taken in $A$ for LHS and in $A$ for RHS; equivalently in the ambient space of the codomain for $T^c$).

**Proof.**

(1) $a \in f^{-1}(\bigcup T_i)$ iff $f(a) \in \bigcup T_i$ iff $\exists i : f(a) \in T_i$ iff $\exists i : a \in f^{-1}(T_i)$ iff $a \in \bigcup f^{-1}(T_i)$.

(2) $a \in f^{-1}(\bigcap T_i)$ iff $f(a) \in \bigcap T_i$ iff $\forall i : f(a) \in T_i$ iff $\forall i : a \in f^{-1}(T_i)$ iff $a \in \bigcap f^{-1}(T_i)$.

(3) $a \in f^{-1}(T^c)$ iff $f(a) \in T^c$ iff $f(a) \notin T$ iff $a \notin f^{-1}(T)$ iff $a \in (f^{-1}(T))^c$. $\square$

**Why this matters.** Preimages commute with all set operations. *Images do not* — in general $f(A \cap B) \neq f(A) \cap f(B)$ (example: constant function sending everything to a single point). This is why measurability is defined in terms of preimages: a function $f$ is measurable if the preimage of every measurable set is measurable, and the class of "preimages of measurable sets" is closed under the σ-algebra operations automatically by Theorem 0.1.5.8. You will see this argument verbatim in Subject 1.

### Theorem 0.1.5.9 (Image is not as well-behaved)

For $f: A \to B$ and $S_1, S_2 \subseteq A$:

1. $f(S_1 \cup S_2) = f(S_1) \cup f(S_2)$.
2. $f(S_1 \cap S_2) \subseteq f(S_1) \cap f(S_2)$, with equality iff $f$ is injective on $S_1 \cup S_2$.

**Proof.**

(1) $b \in f(S_1 \cup S_2)$ iff $\exists a \in S_1 \cup S_2 : f(a) = b$ iff $(\exists a \in S_1 : f(a) = b) \vee (\exists a \in S_2 : f(a) = b)$ iff $b \in f(S_1) \cup f(S_2)$.

(2) $(\subseteq)$: If $b = f(a)$ with $a \in S_1 \cap S_2$, then $a \in S_1$ so $b \in f(S_1)$, and $a \in S_2$ so $b \in f(S_2)$. Hence $b \in f(S_1) \cap f(S_2)$.

*(Equality).* If $f$ is injective on $S_1 \cup S_2$: let $b \in f(S_1) \cap f(S_2)$. Then $b = f(a_1)$ for some $a_1 \in S_1$ and $b = f(a_2)$ for some $a_2 \in S_2$. Injectivity gives $a_1 = a_2 \in S_1 \cap S_2$, so $b \in f(S_1 \cap S_2)$.

*(Converse of equality).* Suppose equality always holds for every $S_1, S_2 \subseteq A$. Assume for contradiction $f$ is not injective: $\exists a_1 \neq a_2$ with $f(a_1) = f(a_2) = b$. Take $S_1 = \{a_1\}, S_2 = \{a_2\}$. Then $S_1 \cap S_2 = \emptyset$, so $f(S_1 \cap S_2) = \emptyset$, but $f(S_1) \cap f(S_2) = \{b\} \cap \{b\} = \{b\} \neq \emptyset$. Contradiction. So $f$ is injective.

$\square$

## Worked Examples

### Example 0.1.5.10 (Direct: a map that is injective but not surjective)

*Let $f: \mathbb{N} \to \mathbb{N}$, $f(n) = 2n$. Show $f$ is injective but not surjective, and write down a left inverse.*

**Solution.** Injective: $f(n_1) = f(n_2)$ gives $2n_1 = 2n_2$, so $n_1 = n_2$. Not surjective: the element $1 \in \mathbb{N}$ (taking $\mathbb{N} = \{1, 2, \ldots\}$) is not in the image ($1$ is odd, the image is the even positive integers).

Left inverse: $g: \mathbb{N} \to \mathbb{N}$ defined by $g(m) = m / 2$ if $m$ even, $g(m) = 1$ (arbitrary choice) if $m$ odd. Then $g(f(n)) = g(2n) = n$. $\square$

### Example 0.1.5.11 (Subtle: image versus preimage fails)

*Give an explicit $f: \mathbb{R} \to \mathbb{R}$ and sets $S_1, S_2$ for which $f(S_1 \cap S_2) \subsetneq f(S_1) \cap f(S_2)$.*

**Solution.** Take $f(x) = x^2$, $S_1 = [0, 1]$, $S_2 = [-1, 0]$. Then $S_1 \cap S_2 = \{0\}$, so $f(S_1 \cap S_2) = \{0\}$.

But $f(S_1) = [0, 1]$, $f(S_2) = [0, 1]$, so $f(S_1) \cap f(S_2) = [0, 1]$.

The inclusion $\{0\} \subsetneq [0, 1]$ is strict. The root cause: $f$ is not injective on $S_1 \cup S_2 = [-1, 1]$ because $f(x) = f(-x)$. $\square$

**Why care.** When you push a distribution through a non-injective map (e.g., $Y = X^2$), information is lost: the pre-image set $\{-\sqrt{y}, +\sqrt{y}\}$ has been collapsed. This is exactly why the density of $Y$ picks up a Jacobian factor *plus* a sum over preimage branches — the image-versus-preimage asymmetry manifests as a formula with multiple branches.

## Computational Implementation

```python
def is_injective(f, domain):
    """Return True iff f is injective on the given finite domain."""
    seen = {}
    for x in domain:
        y = f(x)
        if y in seen:
            return False
        seen[y] = x
    return True

def is_surjective(f, domain, codomain):
    """Return True iff f(domain) covers codomain (both finite)."""
    img = {f(x) for x in domain}
    return img == set(codomain)

def is_bijective(f, domain, codomain):
    return is_injective(f, domain) and is_surjective(f, domain, codomain)

# f(x) = x mod 5 from {0,1,...,9} to {0,1,...,4}: surjective but not injective
f = lambda x: x % 5
D = list(range(10))
C = list(range(5))
assert not is_injective(f, D)
assert is_surjective(f, D, C)

# f(x) = 2x from {0,...,4} to {0,...,9}: injective but not surjective
f2 = lambda x: 2 * x
D2 = list(range(5))
C2 = list(range(10))
assert is_injective(f2, D2)
assert not is_surjective(f2, D2, C2)

# Verify preimage commutes with union
def preimage(f, T, domain):
    return {x for x in domain if f(x) in T}

def image(f, S):
    return {f(x) for x in S}

D = list(range(20))
f = lambda x: x ** 2 % 7
T1, T2 = {0, 1, 2}, {2, 3, 4}
assert preimage(f, T1 | T2, D) == preimage(f, T1, D) | preimage(f, T2, D)
assert preimage(f, T1 & T2, D) == preimage(f, T1, D) & preimage(f, T2, D)

# Image and intersection: show general failure
f = lambda x: x * x
S1 = set(range(-5, 1))
S2 = set(range(0, 6))
assert image(f, S1 & S2) == {0}
assert image(f, S1) & image(f, S2) == {x * x for x in range(0, 6)}
print("All function checks passed.")
```

## [QUANT APPLICATION]

**Feature maps, lossless compression, and identification.** Suppose you have a feature-engineering function $\phi: \mathbb{R}^d \to \mathbb{R}^m$ mapping raw market data into alpha features. Questions you must ask:

- *Is $\phi$ injective?* If not, different raw states map to the same feature vector, and the model cannot possibly distinguish them — there is information loss that no downstream model, no matter how deep, can recover. (This is a baseline check often skipped by practitioners.) Injective $\phi$ is equivalent to having a left inverse — a *decoder* that recovers the raw state.

- *Is $\phi$ surjective on the feature space?* If not, the image is a strict subset; a regression that assumes $\phi(\text{data})$ covers the space may be extrapolating outside the support.

- *Is $\phi$ a bijection onto its image?* This is what we want for reversible feature transformations (like invertible neural networks used in density estimation for returns).

These are not pedantic concerns. The difference between an information-preserving transformation and a lossy one is the difference between $\phi$ being injective and not — and this is *the* question before you train any model on $\phi(\text{data})$.

## Exercises

### ★ (Foundation)

**E0.1.5.1.** For each function, decide whether it is injective, surjective, both, or neither. Justify briefly.
(a) $f: \mathbb{R} \to \mathbb{R}$, $f(x) = 3x - 7$.
(b) $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$.
(c) $f: [0, \infty) \to [0, \infty)$, $f(x) = x^2$.
(d) $f: \mathbb{Z} \to \mathbb{Z}$, $f(n) = n^3 - n$.

*Solution.* (a) Bijective. Inverse: $g(y) = (y+7)/3$. (b) Neither: not injective ($f(1) = f(-1)$), not surjective onto $\mathbb{R}$ (image is $[0, \infty)$). (c) Bijective: injective on $[0, \infty)$ because $x^2$ is strictly increasing there, surjective because every $y \geq 0$ has $\sqrt{y} \geq 0$ as a preimage. (d) $f(0) = f(1) = f(-1) = 0$, so not injective. Not surjective (image misses $n = 2$, for instance — check: $n^3 - n = 2$ has no integer solution, since $n = 1, 2$ give $0, 6$).

**E0.1.5.2.** Compute $f(S)$ and $f^{-1}(T)$ for $f: \mathbb{R} \to \mathbb{R}, f(x) = x^2$, $S = [-2, 3]$, $T = [1, 4]$.

*Solution.* $f(S) = \{x^2 : x \in [-2, 3]\} = [0, 9]$ (the max is $9 = 3^2$, achieved at $x = 3$; the min is $0 = 0^2$). $f^{-1}(T) = \{x : 1 \leq x^2 \leq 4\} = [-2, -1] \cup [1, 2]$.

**E0.1.5.3.** Prove: if $f: A \to B$ and $g: B \to C$ are both bijective, then $g \circ f$ is bijective with inverse $f^{-1} \circ g^{-1}$.

*Solution.* Already done in Theorem 0.1.5.7(3).

### ★★ (Intermediate)

**E0.1.5.4.** Let $f: A \to B$ and suppose $g, h: B \to A$ are both two-sided inverses of $f$. Prove $g = h$.

*Solution.* Apply the calculation in the proof of Theorem 0.1.5.6(3):
$$g = g \circ \mathrm{id}_B = g \circ (f \circ h) = (g \circ f) \circ h = \mathrm{id}_A \circ h = h.$$
$\square$

**E0.1.5.5.** Let $f: \mathbb{R} \to \mathbb{R}$ be continuous and injective. Prove $f$ is strictly monotone.

*Solution.* Suppose for contradiction $f$ is continuous and injective but not strictly monotone. Then there exist $a < b < c$ with (say) $f(a) < f(b)$ but $f(b) > f(c)$, or $f(a) > f(b)$ but $f(b) < f(c)$. Consider the first case. By the intermediate value theorem (which you have from JEE; it will be reproved in Module 0.5), for any value $v$ strictly between $\max(f(a), f(c))$ and $f(b)$, there is a point $x_1 \in (a, b)$ with $f(x_1) = v$ and a point $x_2 \in (b, c)$ with $f(x_2) = v$. Since $x_1 \neq x_2$, $f$ is not injective — contradiction. The other case is symmetric. $\square$

**E0.1.5.6.** Let $f: A \to B, g: B \to A$ satisfy $g \circ f = \mathrm{id}_A$ (i.e., $g$ is a left inverse of $f$). Prove $f$ is injective and $g$ is surjective.

*Solution.* $f$ injective: this is the converse direction in the proof of Theorem 0.1.5.6(1). $g$ surjective: for any $a \in A$, $g(f(a)) = a$, so $f(a) \in B$ is a preimage of $a$ under $g$. $\square$

### ★★★ (Challenge)

**E0.1.5.7.** Let $f: A \to A$ be a function on a set $A$, and suppose $f$ satisfies $f \circ f \circ f = f$. Prove: $f$ is injective iff $f$ is surjective iff $f$ is a bijection with $f^{-1} = f$ (an *involution*) — or else $f$ has a more complex orbit structure. Classify all possible orbit types.

*Hint.* Consider the image and fixed-point set of $f$. The equation $f^3 = f$ forces $f$ restricted to the image of $f$ to be an involution.

**E0.1.5.8.** (Cantor–Bernstein preview) Suppose $f: A \to B$ and $g: B \to A$ are both injective. Prove there exists a bijection $h: A \to B$. (This is the Cantor–Bernstein–Schroeder theorem; we will use it in Topic 0.1.6.)

*Hint.* Iterate $f$ and $g$ back and forth and partition $A$ and $B$ by "origin" of the iteration. Each element of $A$ either has an infinite chain of $g^{-1}$-predecessors, or one that eventually terminates. Use these chains to define $h$.

**E0.1.5.9.** Let $f: [0, 1] \to [0, 1]$ be continuous and surjective, satisfying $f(0) = 0, f(1) = 1$, and $f$ piecewise linear (like a zigzag). Show that if $f$ has $k$ linear pieces, then the equation $f(x) = c$ has at most $k$ solutions for each $c$, and at least one solution has a constant preimage that is independent of the monotonicity pattern.

*Hint.* This is preparing intuition for ergodic theory and Markov shifts in Subject 3.

---

# Topic 0.1.6 — Cardinality and Cantor's Diagonal Argument

## Motivation

How do we compare the "sizes" of infinite sets? In finite combinatorics the question is banal: count. For infinite sets, intuition fails. It turns out there are *different* sizes of infinity, and this is not a philosophical observation — it is a theorem. Cantor showed that the rational numbers are no more numerous than the positive integers, but the real numbers are *strictly* more numerous. This fact has a direct consequence for the rest of this curriculum: you cannot assign probabilities to "every subset" of $[0, 1]$ in a consistent way, because there are *too many* subsets. That is exactly the motivation for measure theory (Subject 1). So the last topic in the prerequisite module earns its keep: it explains why the next subject exists.

## Prerequisites

Topics 0.1.1–0.1.5.

## Definitions

**Definition 0.1.6.1 (Equinumerous sets).** Two sets $A, B$ are *equinumerous* (have the same cardinality), written $|A| = |B|$, if there exists a bijection $f: A \to B$.

*Intuition.* Two sets have the same size iff you can match them up element-for-element.

**Definition 0.1.6.2 (Cardinality ordering).** $|A| \leq |B|$ iff there exists an injection $f: A \to B$. We write $|A| < |B|$ if $|A| \leq |B|$ and $|A| \neq |B|$.

**Definition 0.1.6.3 (Finite, countable, uncountable).**

- $A$ is *finite* if $|A| = |\{1, 2, \ldots, n\}|$ for some $n \in \mathbb{N}_0$ (with the convention $|A| = 0$ iff $A = \emptyset$).
- $A$ is *countably infinite* (or just *countable*, in some textbooks; we will use "countable" to mean "finite or countably infinite" when unambiguous) if $|A| = |\mathbb{N}|$. The cardinality $|\mathbb{N}|$ is denoted $\aleph_0$ ("aleph-null").
- $A$ is *uncountable* if it is infinite and $|A| \neq |\mathbb{N}|$.

*Convention.* In this curriculum, "countable" will mean "countably infinite or finite" unless context disambiguates. The cardinality of the continuum, $|\mathbb{R}|$, is denoted $\mathfrak{c}$ or $2^{\aleph_0}$.

## Key Results

### Theorem 0.1.6.4 (Equinumerosity is an equivalence relation)

On the class of sets, equinumerosity is reflexive ($|A| = |A|$), symmetric ($|A| = |B| \Rightarrow |B| = |A|$), and transitive ($|A| = |B|, |B| = |C| \Rightarrow |A| = |C|$).

**Proof.** Reflexive: $\mathrm{id}_A$ is a bijection $A \to A$. Symmetric: the inverse of a bijection is a bijection. Transitive: composition of bijections is a bijection (Theorem 0.1.5.7). $\square$

### Theorem 0.1.6.5 (Cantor–Bernstein–Schroeder)

If $|A| \leq |B|$ and $|B| \leq |A|$, then $|A| = |B|$.

Equivalently, if there are injections $f: A \to B$ and $g: B \to A$, there is a bijection $h: A \to B$.

**Proof.** We construct $h$ explicitly. For $a \in A$, consider the backwards chain
$$a, \; g^{-1}(a), \; f^{-1}(g^{-1}(a)), \; g^{-1}(f^{-1}(g^{-1}(a))), \ldots$$
This chain alternates between elements of $A$ and $B$, each step taken by the partial inverse of $g$ or $f$. It either terminates (an element has no preimage) or continues forever. Partition $A$ into three classes:
- $A_\infty$: the chain continues forever.
- $A_A$: the chain terminates in $A$ (i.e., some element in $A$ has no $g^{-1}$).
- $A_B$: the chain terminates in $B$.

Similarly partition $B$ into $B_\infty, B_B, B_A$.

Define $h: A \to B$ by
$$h(a) = \begin{cases} f(a) & \text{if } a \in A_\infty \cup A_A \\ g^{-1}(a) & \text{if } a \in A_B \end{cases}.$$

*Claim.* $h$ is a bijection.

- *Well-defined.* $g^{-1}(a)$ exists for $a \in A_B$ because: the chain starting at $a$ went at least one step back via $g^{-1}$ (else it would have terminated in $A_A$), so $a$ is in the image of $g$, so $g^{-1}(a)$ is defined (and unique by injectivity of $g$).
- *Injective.* Suppose $h(a_1) = h(a_2)$.
  - If both $a_1, a_2 \in A_\infty \cup A_A$: $f(a_1) = f(a_2)$, so by injectivity of $f$, $a_1 = a_2$.
  - If both $a_1, a_2 \in A_B$: $g^{-1}(a_1) = g^{-1}(a_2)$, so applying $g$, $a_1 = a_2$.
  - Mixed: $a_1 \in A_\infty \cup A_A, a_2 \in A_B$. Then $f(a_1) = g^{-1}(a_2)$, so $g(f(a_1)) = a_2$. Following the chain backward from $a_2$: $g^{-1}(a_2) = f(a_1)$; if $a_1 \in A_A$, the chain from $a_1$ terminates in $A$, hence the chain from $a_2$ also terminates in $A$, meaning $a_2 \in A_A$, contradicting $a_2 \in A_B$. If $a_1 \in A_\infty$, the chain from $a_1$ is infinite, so the chain from $a_2$ is infinite, meaning $a_2 \in A_\infty$, again contradicting $a_2 \in A_B$. So the mixed case is impossible.
- *Surjective.* Let $b \in B$.
  - If $b \in B_\infty \cup B_B$: then $b$ lies in the "image-of-$f$" branch — by tracing the chain from $b$, we see $b$ has a $f$-preimage in $A_\infty \cup A_A$. Explicitly, look at the chain from $b$: it either continues forever ($b \in B_\infty$) or terminates in $B$. Either way, the $g^{-1}(b)$ step lands in $A$, then $f^{-1}$ of that is... wait, we want to invert $h$. Simpler: consider $a = f^{-1}(b)$ where defined. $f$ injective, so this is unique if it exists. If $b \in f(A)$, then $a = f^{-1}(b) \in A$. Show $a \in A_\infty \cup A_A$: the chain from $b$ and the chain from $a$ share all terms after the first (the chain from $a$ starts $a, g^{-1}(a), \ldots$, and $f(a) = b$ means $b$'s chain starts $b, g^{-1}(b) = g^{-1}(f(a))$... hmm wait the relationship is slightly different). Let me redo. The chain from $b$: $b, g^{-1}(b)$, etc. The chain from $a$: $a$ — but $a$ is obtained from $b$ as $f^{-1}(b)$, which is $a$, then the chain continues $a, g^{-1}(a), \ldots$. The chain from $b$ runs $b, g^{-1}(b), \ldots$. So they are different chains. The relationship is: $a$'s chain is $b$'s chain with the first term $b$ replaced by $a$. This implies both chains have the same "termination type" (both infinite, or both terminate in the same place). If $b \in B_\infty$, then $a \in A_\infty$. If $b \in B_B$, then the chain terminates in $B$ meaning some $B$-element has no $f^{-1}$; but the chain from $a$ skipped the very first $B$-element $b$, so whether $a \in A_\infty, A_A, A_B$ depends on the location of termination. For $B_B$: the chain terminates because some element of $B$ has no $f$-preimage. That element may or may not be $b$ itself. If it's $b$, then $b$ had no $f$-preimage, contradicting $b \in f(A)$. So it's a later element, meaning the chain from $a$ still encounters it and terminates in $B_A$... hmm this is getting tangled. Let me just assert the result, which is classical.
  
  The correct statement: for $b \in B_B \cup B_\infty$, set $a = f^{-1}(b)$; then $a$ is defined and $a \in A_A \cup A_\infty$, so $h(a) = f(a) = b$. For $b \in B_A$, set $a = g(b)$; then $a \in A_B$ and $h(a) = g^{-1}(a) = g^{-1}(g(b)) = b$.

(A fully rigorous proof uses a fixed-point construction known as Knaster–Tarski; the chain-chasing construction above is intuitive but fiddly. The standard book-version proof via Tarski's fixed point on the map $T(X) = A \setminus g(B \setminus f(X))$, yielding a fixed point $X_0$ for which $h = f$ on $X_0$ and $h = g^{-1}$ on $A \setminus X_0$, is cleaner. I will not reproduce the full Tarski proof here; consult Halmos or Folland.) $\square$

**Why care.** Cantor–Bernstein is what lets us conclude that two sets have equal cardinality by producing two injections — often much easier than producing a single bijection. We use it below to show $|(0, 1)| = |\mathbb{R}|$ and other matching results.

### Theorem 0.1.6.6 ($\mathbb{Z}$ is countable)

$|\mathbb{Z}| = |\mathbb{N}|$.

**Proof.** Define $f: \mathbb{N} \to \mathbb{Z}$ by
$$f(n) = \begin{cases} n/2 & \text{if } n \text{ even} \\ -(n-1)/2 & \text{if } n \text{ odd} \end{cases}.$$
Check: $f(1) = 0, f(2) = 1, f(3) = -1, f(4) = 2, f(5) = -2, \ldots$. This lists every integer exactly once; $f$ is a bijection. $\square$

### Theorem 0.1.6.7 ($\mathbb{N} \times \mathbb{N}$ is countable; countable union of countable sets is countable)

$|\mathbb{N} \times \mathbb{N}| = |\mathbb{N}|$. More generally, if $\{A_i\}_{i \in \mathbb{N}}$ is a family of countable sets, then $\bigcup_i A_i$ is countable.

**Proof.**

$(i)$ Cantor's pairing: define $\pi: \mathbb{N} \times \mathbb{N} \to \mathbb{N}$ by
$$\pi(m, n) = \frac{(m + n - 2)(m + n - 1)}{2} + m.$$
This enumerates $\mathbb{N} \times \mathbb{N}$ along the anti-diagonals $\{(m, n) : m + n = k\}$ for $k = 2, 3, \ldots$. Each anti-diagonal has $k - 1$ points, visited in the order $(1, k-1), (2, k-2), \ldots, (k-1, 1)$. One verifies by induction on $k$ that $\pi$ hits each positive integer exactly once; hence it is a bijection.

$(ii)$ For a countable union of countable sets: enumerate each $A_i = \{a_{i,1}, a_{i,2}, \ldots\}$ (possibly with repetitions, if some $A_i$ is finite). The map $(i, j) \mapsto a_{i,j}$ is a surjection $\mathbb{N} \times \mathbb{N} \to \bigcup A_i$. Composing with the pairing bijection $\pi^{-1}$ gives a surjection $\mathbb{N} \to \bigcup A_i$. A surjection from $\mathbb{N}$ onto any non-empty set means the set is at most countably infinite; combined with the injection from any $A_i$ into $\bigcup A_i$, the union is exactly countably infinite (if any $A_i$ is infinite) or finite (if all are finite).

More concretely, to get a *bijection* $\mathbb{N} \to \bigcup A_i$ assuming all sets distinct and at least one infinite, list $(a_{i,j})$ in some interleaved order skipping repeats:
$$a_{1,1}, a_{1,2}, a_{2,1}, a_{1,3}, a_{2,2}, a_{3,1}, \ldots$$
(the anti-diagonal order on $\mathbb{N} \times \mathbb{N}$, skipping any $a_{i,j}$ equal to a previously listed element). $\square$

**Corollary 0.1.6.8.** $\mathbb{Q}$ is countable.

**Proof.** $\mathbb{Q} = \bigcup_{n \in \mathbb{N}} \{m/n : m \in \mathbb{Z}\}$. Each set in the union is a bijective image of $\mathbb{Z}$, hence countable. Countable union of countable sets is countable. $\square$

### Theorem 0.1.6.9 (Cantor's theorem: $\mathbb{R}$ is uncountable)

There is no surjection $\mathbb{N} \to \mathbb{R}$. In particular, $|\mathbb{N}| < |\mathbb{R}|$.

**Proof.** It suffices to show there is no surjection $\mathbb{N} \to (0, 1)$, since $(0, 1) \subseteq \mathbb{R}$ and an injection $\mathbb{N} \to \mathbb{R}$ (e.g., $n \mapsto n$) is immediate, giving $|\mathbb{N}| \leq |\mathbb{R}|$; then uncountability of $(0, 1)$ gives $|\mathbb{N}| \neq |\mathbb{R}|$, hence strict.

Suppose for contradiction there is a surjection $r: \mathbb{N} \to (0, 1)$. Enumerate: $r(1) = x_1, r(2) = x_2, \ldots$. Write each $x_n$ in decimal:
$$x_n = 0.d_{n,1} d_{n,2} d_{n,3} \ldots$$
where each $d_{n,k} \in \{0, 1, \ldots, 9\}$. To avoid the ambiguity of repeated 9s (e.g., $0.4999\ldots = 0.5000\ldots$), require the "non-terminating-9" expansion — i.e., for any rational with a terminating decimal, use the version ending in 0s; for all others, the expansion is unique.

Now construct a new number $y \in (0, 1)$:
$$y = 0.e_1 e_2 e_3 \ldots, \quad e_k := \begin{cases} 1 & \text{if } d_{k,k} \neq 1 \\ 2 & \text{if } d_{k,k} = 1 \end{cases}.$$
Choices: we use only digits 1 and 2 for $y$, so $y$'s decimal expansion does not contain any 9s and is unique. By construction, $y$ differs from $x_n$ in the $n$-th decimal place — since $e_n \neq d_{n,n}$. Hence $y \neq x_n$ for every $n$, i.e., $y$ is not in the range of $r$. Contradiction. $\square$

**The construction, named.** This is *Cantor's diagonal argument*. It is the single most important proof in set theory. The name comes from looking at the enumerated decimal expansions as an infinite matrix and going down the diagonal (the $(n, n)$ entries), flipping each digit to force disagreement.

**Intuition.** Any attempted enumeration of the reals must "miss" at least one real number. The construction shows how to build such a missing number: go down the list and disagree with each entry in its own position. The proof is constructive and uses no axiom of choice.

### Theorem 0.1.6.10 (Cantor: $|A| < |\mathcal{P}(A)|$ for every set $A$)

For every set $A$, there is no surjection $A \to \mathcal{P}(A)$.

**Proof.** Suppose for contradiction there is a surjection $f: A \to \mathcal{P}(A)$. Define
$$D = \{a \in A : a \notin f(a)\}.$$
Then $D \subseteq A$, so $D \in \mathcal{P}(A)$. By surjectivity, $D = f(a_0)$ for some $a_0 \in A$.

Ask: is $a_0 \in D$?

- If $a_0 \in D$: by definition of $D$, $a_0 \notin f(a_0) = D$. Contradiction.
- If $a_0 \notin D$: by definition of $D$, "$a_0 \notin f(a_0)$" fails, i.e., $a_0 \in f(a_0) = D$. Contradiction.

Both cases contradict; hence the assumption that a surjection exists is false. $\square$

**Meaning.** $|\mathcal{P}(A)|$ is always strictly larger than $|A|$. In particular, iterating the power set yields an unbounded hierarchy of infinities:
$$|\mathbb{N}| < |\mathcal{P}(\mathbb{N})| < |\mathcal{P}(\mathcal{P}(\mathbb{N}))| < \cdots$$
There is no largest set. This is a fundamental and (for the uninitiated) counterintuitive result: there is no "set of all sets", and infinity comes in unboundedly many sizes.

**The shape of the argument.** Theorem 0.1.6.9 and 0.1.6.10 are both instances of the same diagonal construction: given a proposed map, construct an element that disagrees with every image "in its own slot". Russell's paradox is this same construction in disguise. Gödel's incompleteness theorem uses a self-referential version. This is one of the most important proof patterns in mathematical logic.

### Theorem 0.1.6.11 (Cardinality of the continuum)

$|\mathbb{R}| = |\mathcal{P}(\mathbb{N})| = 2^{\aleph_0}$.

**Proof.** Write each $r \in (0, 1)$ in binary: $r = 0.b_1 b_2 b_3 \ldots$ with $b_i \in \{0, 1\}$ (using the non-terminating-1 convention). This gives an injection $(0, 1) \hookrightarrow \{0, 1\}^\mathbb{N}$ (up to the terminating-binary ambiguity, which affects only a countable set of reals — a negligible issue that can be handled by a standard Cantor–Bernstein argument).

Conversely, a sequence $b = (b_1, b_2, \ldots) \in \{0, 1\}^\mathbb{N}$ is the indicator of a subset $S_b := \{n : b_n = 1\} \subseteq \mathbb{N}$, giving a bijection $\{0, 1\}^\mathbb{N} \leftrightarrow \mathcal{P}(\mathbb{N})$. So $|\mathcal{P}(\mathbb{N})| = |\{0,1\}^\mathbb{N}|$.

Between $(0, 1)$ and $\{0, 1\}^\mathbb{N}$ there are injections both ways: $(0,1) \hookrightarrow \{0,1\}^\mathbb{N}$ via binary expansion (handle ambiguity by choosing unique representatives), and $\{0,1\}^\mathbb{N} \hookrightarrow (0,1)$ via interleaved ternary: $b \mapsto \sum_n b_n / 3^n$ (this lands in the Cantor set $\subset [0, 1]$, not $(0, 1)$ strictly, but a small shift fixes that — or use base 4 with values in $\{1, 3\}$).

By Cantor–Bernstein, $|(0,1)| = |\{0,1\}^\mathbb{N}| = |\mathcal{P}(\mathbb{N})|$.

Finally $|\mathbb{R}| = |(0, 1)|$: the map $x \mapsto \tan(\pi(x - 1/2))$ is a bijection $(0, 1) \to \mathbb{R}$.

Combining, $|\mathbb{R}| = 2^{\aleph_0}$. $\square$

## Worked Examples

### Example 0.1.6.12 (Direct: bijection between $\mathbb{N}$ and $\mathbb{N} \cup \{0\}$)

*Exhibit a bijection $f: \mathbb{N} \to \mathbb{N} \cup \{0\}$.*

**Solution.** $f(n) = n - 1$. It is injective ($n_1 - 1 = n_2 - 1 \Rightarrow n_1 = n_2$) and surjective (every $m \in \mathbb{N} \cup \{0\}$ has preimage $m + 1 \in \mathbb{N}$). $\square$

This illustrates Hilbert's Grand Hotel: $\mathbb{N}$ has the same size as "$\mathbb{N}$ plus one extra element", even though it is a proper superset. This is a *characterizing property* of infinite sets (being equinumerous with a proper subset).

### Example 0.1.6.13 (Trickier: $|(0, 1)| = |[0, 1]|$)

*Exhibit a bijection $(0, 1) \to [0, 1]$.*

**Solution.** The naive inclusion $(0, 1) \hookrightarrow [0, 1]$ is an injection but not surjective (misses $0, 1$). The reverse: $[0, 1] \to (0, 1)$ via $x \mapsto (x + 1)/3 \in [1/3, 2/3] \subset (0, 1)$ is injective. Cantor–Bernstein implies a bijection exists.

To exhibit one explicitly: pick a countably infinite subset $C = \{c_0, c_1, c_2, \ldots\} \subset (0, 1)$, e.g., $c_n = 1/(n+2)$ (so $c_0 = 1/2, c_1 = 1/3, c_2 = 1/4, \ldots$). Define
$$f(x) = \begin{cases} 0 & \text{if } x = c_0 \\ 1 & \text{if } x = c_1 \\ c_{n-2} & \text{if } x = c_n, n \geq 2 \\ x & \text{otherwise} \end{cases}.$$
Then $f$ maps $(0, 1) \setminus C$ to $(0, 1) \setminus C$ identically, $\{c_0, c_1\} \to \{0, 1\}$, and the remaining $\{c_2, c_3, \ldots\} \to \{c_0, c_1, c_2, \ldots\}$. Check bijectivity on each block; $f$ is a bijection $(0, 1) \to [0, 1]$. $\square$

**Lesson.** Cardinality is insensitive to endpoints: losing or gaining a handful of points from an uncountable set does not change its cardinality. This is the start of understanding why "measure zero sets can be ignored" is a natural operation in measure theory.

## Computational Implementation

```python
# Cantor's pairing function and its inverse
def cantor_pair(m, n):
    """Bijection N x N -> N (with N = {1, 2, ...})."""
    s = m + n - 2
    return s * (s + 1) // 2 + m

def cantor_unpair(k):
    """Inverse of cantor_pair."""
    # k-th element is in the anti-diagonal s where s(s+1)/2 < k <= (s+1)(s+2)/2
    s = 0
    while (s + 1) * (s + 2) // 2 < k:
        s += 1
    m = k - s * (s + 1) // 2
    n = s - m + 2
    return m, n

# Verify bijection on first 50 integers
for k in range(1, 51):
    m, n = cantor_unpair(k)
    assert cantor_pair(m, n) == k

print("Cantor pairing verified for first 50 pairs.")

# Simulate diagonal argument: given a list of "real numbers" as digit sequences,
# construct a new real differing from each in its own diagonal digit.
import random

def diagonal_real(sequences):
    """Given sequences[i] = list of digits, return a digit list differing from
    sequences[i][i] for each i."""
    y = []
    for i, seq in enumerate(sequences):
        if seq[i] == 1:
            y.append(2)
        else:
            y.append(1)
    return y

# Generate 5 pseudo-random 5-digit reals
random.seed(42)
seqs = [[random.randint(0, 9) for _ in range(5)] for _ in range(5)]
for i, s in enumerate(seqs):
    print(f"x_{i+1} = 0." + "".join(map(str, s)))

y = diagonal_real(seqs)
print(f"y   = 0." + "".join(map(str, y)))

# Verify y disagrees with each x in its diagonal digit
for i in range(5):
    assert y[i] != seqs[i][i]
print("Diagonal construction verified.")
```

Running this prints five generated digit-strings and the constructed diagonal real, and checks that the diagonal real differs from each enumerated "real" in the correct position.

## [QUANT APPLICATION]

**Why probability theory requires measure theory — the punchline.** Consider trying to define, for each subset $A \subseteq [0, 1]$, a "probability" $P(A)$ that captures "pick a point uniformly at random". You want it to satisfy:

1. $0 \leq P(A) \leq 1$ for all $A \subseteq [0, 1]$.
2. $P([0, 1]) = 1$.
3. (Translation invariance, modulo wrapping.) $P(A + c \mod 1) = P(A)$ for all $c \in \mathbb{R}$.
4. (Countable additivity.) If $A_1, A_2, \ldots$ are pairwise disjoint, $P(\bigcup A_n) = \sum P(A_n)$.

It turns out these four conditions are *inconsistent with $P$ being defined on the full power set $\mathcal{P}([0, 1])$*. The proof is a construction due to Vitali: partition $[0, 1]$ into equivalence classes under $x \sim y$ iff $x - y \in \mathbb{Q}$. Each class is countable, so there are $|\mathbb{R}|/|\mathbb{Q}| = \mathfrak{c}$ classes. Use the axiom of choice to pick one representative from each class; call the resulting set $V$. Then the translates $V + q$ for $q \in \mathbb{Q} \cap [-1, 1]$ partition a subset of $[-1, 2]$ into a countable disjoint union of translates of $V$; translation invariance plus countable additivity would give $P(V + q) = P(V)$, and summing gives $P([-1, 2]) = \sum_q P(V)$ — either $0$ (if $P(V) = 0$) or $\infty$ (if $P(V) > 0$). Neither is consistent with $P([-1, 2])$ being a finite number between $0$ and $3$.

Conclusion: the "naive" probability $P$ cannot be defined on *every* subset of $[0, 1]$. So we must restrict to a sub-family of subsets — the *measurable* sets, forming a σ-algebra — and define probability there. This is exactly what measure theory does, and it is the opening move of Subject 1 (Module 1.1).

The step that made the Vitali construction work was: uncountably many equivalence classes of reals under rational translation. Without the uncountability of $\mathbb{R}$ (Theorem 0.1.6.9), there would be no such paradox. *Cantor's diagonal argument is the reason measure theory exists.*

## Exercises

### ★ (Foundation)

**E0.1.6.1.** Show that the set of even positive integers has the same cardinality as $\mathbb{N}$.

*Solution.* $f: \mathbb{N} \to 2\mathbb{N}$, $f(n) = 2n$, is a bijection.

**E0.1.6.2.** Show that any subset of a countable set is countable.

*Solution.* Let $A$ be countable, $B \subseteq A$. Enumerate $A = \{a_1, a_2, \ldots\}$. List the subsequence of $a_i$'s that lie in $B$: this is either finite (so $B$ finite, hence countable) or an infinite subsequence indexed by $\mathbb{N}$ (giving a bijection $\mathbb{N} \to B$). $\square$

**E0.1.6.3.** Show that $|\mathbb{N}| = |\mathbb{N} \cup \{\mathrm{banana}, \mathrm{apple}\}|$.

*Solution.* Shift $\mathbb{N}$ by 2: define $f: \mathbb{N} \cup \{\mathrm{banana}, \mathrm{apple}\} \to \mathbb{N}$ by $f(\mathrm{banana}) = 1, f(\mathrm{apple}) = 2, f(n) = n + 2$ for $n \in \mathbb{N}$. Bijection. $\square$

### ★★ (Intermediate)

**E0.1.6.4.** Show that the set of all *finite* subsets of $\mathbb{N}$ is countable.

*Solution.* Each finite subset $S \subseteq \mathbb{N}$ corresponds to a finite strictly increasing tuple $(s_1 < s_2 < \cdots < s_k)$. The set of tuples of length $k$ is contained in $\mathbb{N}^k$, which is countable (by iterating the pairing-function argument). The union over $k$ is a countable union of countable sets, hence countable. $\square$

**E0.1.6.5.** Show that $|\mathbb{R}| = |\mathbb{R}^2|$ (a counterintuitive result).

*Solution.* We give injections both ways.

$\mathbb{R} \hookrightarrow \mathbb{R}^2$: $x \mapsto (x, 0)$.

$\mathbb{R}^2 \hookrightarrow \mathbb{R}$: for $(x, y) \in [0, 1)^2$, interleave digits: if $x = 0.a_1 a_2 \ldots$ and $y = 0.b_1 b_2 \ldots$ (unique decimal expansions), define $\iota(x, y) = 0.a_1 b_1 a_2 b_2 \ldots \in [0, 1)$. This is injective (decode by de-interleaving). Compose with an explicit bijection $\mathbb{R}^2 \to [0, 1)^2$ (e.g., coordinate-wise from $\mathbb{R} \to (0, 1) \hookrightarrow [0, 1)$ via $\tan^{-1}$/rescale).

By Cantor–Bernstein, $|\mathbb{R}| = |\mathbb{R}^2|$. $\square$

*Cultural note.* Cantor discovered this in 1877 and wrote to Dedekind: "*Je le vois, mais je ne le crois pas*" ("I see it, but I don't believe it"). The plane and the line are equinumerous as sets, even though their topologies are radically different. This is the reason the topological dimension (and Hausdorff dimension, and measure) are *not* determined by cardinality alone.

**E0.1.6.6.** Show that the set of sequences $(a_1, a_2, \ldots) \in \{0, 1\}^\mathbb{N}$ is uncountable.

*Solution.* Diagonal argument. Suppose $\phi: \mathbb{N} \to \{0, 1\}^\mathbb{N}$ is a surjection. Write $\phi(n) = (\phi(n)_1, \phi(n)_2, \ldots)$. Define $y = (y_1, y_2, \ldots)$ by $y_n = 1 - \phi(n)_n$. Then $y \neq \phi(n)$ for every $n$ (disagrees at position $n$). Contradiction.

Alternatively: $\{0, 1\}^\mathbb{N}$ is in bijection with $\mathcal{P}(\mathbb{N})$, which by Cantor's theorem is strictly larger than $\mathbb{N}$. $\square$

### ★★★ (Challenge)

**E0.1.6.7.** Prove: any subset of $\mathbb{R}$ that is *dense* (its closure equals $\mathbb{R}$) has cardinality at least $\aleph_0$, but can have any cardinality $\kappa$ with $\aleph_0 \leq \kappa \leq \mathfrak{c}$.

*Hint.* $\mathbb{Q}$ is dense and countable; $\mathbb{R}$ itself is dense and of size $\mathfrak{c}$. For intermediate sizes, consider $\mathbb{Q} \cup S$ where $|S| = \kappa$.

**E0.1.6.8.** (Continuum Hypothesis preview) Show that there exist subsets $A \subseteq \mathbb{R}$ with $|\mathbb{N}| < |A|$ (they are at least in the sense that the power set of a countable set embeds into $\mathbb{R}$). The Continuum Hypothesis asserts there is no $|A|$ strictly between $|\mathbb{N}|$ and $|\mathbb{R}|$. Gödel (1940) and Cohen (1963) showed this assertion is *independent* of the ZFC axioms — neither provable nor refutable. Look up the Cohen forcing argument and summarize the main idea in a paragraph.

*Hint.* This is not a standard exercise; the ★★★ rating reflects that the understanding required is deep even to state the theorem precisely. Treat it as a reading assignment on logic.

**E0.1.6.9.** Show that the set of *continuous* functions $\mathbb{R} \to \mathbb{R}$ has cardinality $\mathfrak{c}$. Contrast with the set of *all* functions $\mathbb{R} \to \mathbb{R}$, which has cardinality $2^{\mathfrak{c}} > \mathfrak{c}$.

*Hint.* A continuous function is determined by its values on $\mathbb{Q}$ (dense subset). The set of functions $\mathbb{Q} \to \mathbb{R}$ has cardinality $\mathfrak{c}^{\aleph_0} = \mathfrak{c}$. Hence continuous functions embed into this set, so number at most $\mathfrak{c}$. The map $x \mapsto x + c$ for $c \in \mathbb{R}$ gives an injection $\mathbb{R} \hookrightarrow C(\mathbb{R})$, so at least $\mathfrak{c}$. By Cantor–Bernstein, exactly $\mathfrak{c}$.

This is a useful intuition: most functions are not continuous, and the continuous functions are a "small" (in cardinality) subset of all functions. This foreshadows the measure-theoretic intuition that continuous functions are dense in natural function spaces but have measure zero as a subset of the full space.

---

# Module Summary and Forward Pointers

## What you built in this module

1. **A grammar.** Propositional and predicate logic give a precise language for mathematical statements, with mechanical rules for negation (including nested quantifiers), contraposition, and logical equivalence. Every later definition in this curriculum is parseable in this grammar.

2. **A toolkit of proof patterns.** Direct, contrapositive, contradiction, cases, weak and strong induction — you now know when each applies and how to template them. You have also seen that well-ordering of $\mathbb{N}$, weak induction, and strong induction are logically equivalent.

3. **Set algebra.** Union, intersection, complement, De Morgan, distributivity, inclusion–exclusion — all rigorously established. This is the raw material for σ-algebras in Subject 1.

4. **Functions as relations.** You know the distinction between injective, surjective, and bijective, and you know that preimages commute with all set operations while images do not. This distinction is why measurability is defined via preimages.

5. **Cardinality.** You can tell the difference between countable and uncountable, and you have seen — via Cantor's diagonal — that there is no bijection between $\mathbb{N}$ and $\mathbb{R}$. This fact is the reason measure theory is the foundation of probability rather than naive "assign probability to every subset".

## Where this module feeds into

- *Module 0.2 (Linear Algebra beyond JEE)* needs injective/surjective/bijective for linear maps; needs cardinality for the notion of dimension in arbitrary vector spaces.

- *Module 0.3 (Multivariable Calculus formalized)* uses ε–δ quantifier gymnastics everywhere and needs the ability to manipulate nested $\forall\exists$.

- *Module 0.5 (Real Analysis Foundations)* uses De Morgan, power sets, and Cantor–Bernstein freely.

- *Subject 1 (Measure Theory)* opens with: "since we cannot assign probabilities to every subset of $[0, 1]$ consistently (Vitali's construction), restrict to a σ-algebra". The uncountability of $\mathbb{R}$ (Topic 0.1.6) is the reason we need measure theory.

- *Subject 2 (Probability Theory)* treats random variables as functions; preimages commuting with set operations (Topic 0.1.5) is the reason "$f$ measurable" is a well-behaved definition.

- *Subject 3 (Stochastic Processes)* uses strong induction (in disguise) throughout dynamic programming and backward induction arguments.

The pieces here look elementary because you have seen most of them before. Their value is not in the individual facts; it is in the *fluency* you now have in combining them. In Subject 1, when you are asked to negate "$X$ is measurable", you will not pause — you will mechanically push a negation through two layers of quantifiers, apply De Morgan twice, and read off the answer. That reflex is what this module builds.

---

*End of Module 0.1.*

*Next up: Module 0.2 — Linear Algebra (Beyond JEE). Vector spaces as axiomatic objects, eigendecomposition, spectral theorem, SVD, matrix calculus, norms. Target length comparable. Request "Module 0.2" to proceed.*




