# Scientist Brief: Shachi-Based Synthetic Bureaucracy and Its Policy Interface

## 1. What Shachi contributes

Shachi is used here as the agent-and-environment execution framework. It does not supply a ready-made model of Japanese officials. This template defines a new employee agent, a new organizational environment, and a fixed experimental interface on top of Shachi.

Each employee is an LLM agent with four bounded components:

1. **Config** - stable synthetic profile attributes such as department, career stage, family constraints, and skill fields;
2. **Observation** - only current department conditions, assigned work, valid realized events, and objective fact memory;
3. **Memory** - bounded factual history plus selected narrative context; policy labels, expected effects, provider cost, and other employees' private data are excluded;
4. **Structured action** - one schema-constrained monthly response rather than unrestricted conversation.

Employee agents do not chat directly. They interact indirectly through the shared task queue, staffing state, transfers, realized management outcomes, and department conditions. This matters for interpretation: the experiment studies organizational feedback mediated by an explicit Python environment, not spontaneous conversational coordination among agents.

## 2. What the employee LLM decides

For each active employee-month, the LLM chooses bounded semantic actions:

- relative effort rather than absolute output;
- allocation of the non-protected remainder across core delivery, coordination, learning, and process improvement;
- a work response such as normal delivery, overtime, support request, low-priority deferral, health protection, health leave, caregiving leave, or overtime refusal;
- a voice action such as explanation, staffing relief, process reform, or operational-risk escalation;
- a career action such as staying, requesting transfer, requesting a specialist track, or exploring external exit;
- a structured transfer preference when applicable;
- a contemporaneous self-report on fatigue, turnover intent, trust, fairness, and exploratory interest measures.

The LLM does **not** choose absolute work completed, task arrivals, staffing quantities, exit probabilities, approval quantities, policy effects, or evaluation scores.

## 3. What Python owns

The Python environment is the causal state-transition system. It owns:

- a fixed authorized-slot population with dynamic person identities and identity epochs;
- exogenous deadline-bearing public obligations independent of realized employee capacity;
- monthly time conservation and conversion from relative effort to absolute work capacity;
- deadline-aware task cohorts with criticality, public-harm weight, required field, quality threshold, errors, and rework;
- completed work units, overdue work units, weighted service-harm points, and terminal liability as separate quantities;
- modeled work strain, tacit/formal/routine knowledge, skill accumulation, and process modernization;
- one competing-risk exit draw per employee-month;
- vacancy creation, field-aware batch hiring, onboarding, and replacement identities;
- simultaneous capacity-constrained transfers with donor floors, recipient limits, expertise fit, preference fit, and involuntary-transfer accounting;
- identity-safe event delivery keyed by slot, person ID, identity epoch, and department;
- response caches, run manifests, immutable fingerprints, and scientific metrics.

A policy field cannot directly change welfare because of its label or hypothesis. Effects require the relevant request, eligibility draw, finite allocation, implementation, or realized exposure path defined in `policy_space.json`.

## 4. Monthly organizational sequence

The service month closes on the frozen start-of-month workforce:

1. apply prior-month support, triage, onboarding, transfer, and implementation effects;
2. add exogenous deadline-bearing task cohorts;
3. construct privacy-bounded observations and valid memory;
4. obtain structured employee actions concurrently;
5. conserve total time and calculate employee output, quality, learning, and requests;
6. allocate output to task cohorts and record completion, rework, overdue work, service harm, and terminal liability;
7. resolve competing-risk exits;
8. on scheduled months, plan simultaneous transfers and batch hiring;
9. build privacy-minimized management dockets;
10. apply the public deterministic priority rule and finite management envelopes;
11. revalidate every person-specific request against current identity and department;
12. create next-month realized outcomes only for valid commitments;
13. append state, task, staffing, management, transfer, and audit ledgers.

Management is deliberately deterministic in the primary comparison. Requests are compressed into a configured finite docket per department: the baseline capacity is six cases, and the scientist-editable `management_case_capacity` field can raise or lower that capacity within the declared policy-space range. The allocator still has finite support, triage, reform, explanation, and specialist envelopes. An optional LLM manager is outside candidate selection and must be treated as a separate ablation.

## 5. Sealed survey channel

Self-report is a sealed contemporaneous evaluation channel. Fatigue, turnover intent, trust, fairness, free-text reasons, and next-intent text are not written to future prompts, do not alter management priority, do not drive exits or transfers, and do not enter causal transitions. Report the mechanical welfare anchor and sealed survey composite separately as well as their preregistered combination.

## 6. Policy design interface

The only intervention interface is the strict JSON mapping described in `policy_space.json`. The file lists every permitted field, baseline, valid range, synthetic implementation-cost rule, realization path, and dormancy condition.

A research program contains exactly four intervention objects. Each object must include:

- `Run`: one of 1, 2, 3, or 4;
- `Mechanism`: the state-transition mechanism being tested;
- `ExpectedDirection`: a falsifiable directional prediction;
- `AdverseEffectPrediction`: a concrete service, fairness, composition, or timing failure mode;
- `Policy`: an exact JSON object containing a unique label, an explicit hypothesis, and numeric values only for changed fields.

Omitted policy fields remain at the reference value. Runs 1-3 may change at most eight fields; run 4 may change at most twelve. Each policy must remain within 35 synthetic implementation points. These points are a versioned design-complexity regularizer, not yen, headcount, fiscal expenditure, API cost, or empirical implementation cost.

## 7. Fixed comparison and evidence flow

The executable design is fixed by software:

- `run_0`: one human-authored synthetic stressed reference;
- `run_1`-`run_4`: four full-fidelity interventions;
- 120 authorized civil-servant slots with dynamic identities;
- 48 monthly steps;
- months 1-12 warm-up;
- policy switch at month 13;
- one fixed development scenario and seed;
- matched keyed shocks and exact compatible warm-up cache reuse;
- identical employee schema, prompts, model settings, transition equations, and metrics across arms.

The AI Scientist designs and validates exact policy mappings. The protected orchestrator materializes those JSON mappings directly and runs immutable Python. It does not edit executable experiment or plotting code.

After development, selection is frozen. A candidate is eligible only if it passes every administrative-service and fairness guardrail and strictly improves the preregistered staff-welfare endpoint over `run_0`. If none does, the negative result is final and no holdout is run.

current public release fast-track freezes selection after the five-arm development comparison and does not run additional paid holdout cells. This preserves the AI Scientist improvement task while reducing ABM calls under weekend constraints.

## 8. Evidence and manuscript rules

The only final manuscript is generated after immutable selection and, when applicable, completed disabled holdout. Machine-generated verified tables carry numerical outcome values. Manuscript prose must describe direction, mechanism, trade-offs, and uncertainty without inventing percentages or confidence intervals.

The simulation arm is the treatment unit. Employee rows and person-months are dependent observations inside one organization trajectory and cannot be treated as independent statistical replications.

Explain whether apparent improvement comes from sustained capacity, delayed work, external staffing, composition change, exits, hiring, transfer spillovers, or knowledge accumulation. Always separate development selection from post-selection robustness.

## 9. Claim boundary

Permitted claims describe trajectories and mechanisms inside the declared synthetic organization. Results may identify hypotheses, failure modes, and measurements for later empirical work.

Do not claim that the policy will work in a real ministry, that synthetic employees represent actual officials, that synthetic implementation points are fiscal quantities, that the holdout establishes real-world external validity, or that an LLM reviewer represents calibrated Japanese public-administration expertise.

## 10. Local real-world motivation source

The template provides `source_cards/wakate2022_colorful_public_service.md`, an English summary of the 2022 Cabinet Secretariat / National Personnel Authority young-team proposal. The card is supplied because live literature APIs may not reliably retrieve Japanese government PDFs. Use it for motivation and mechanism vocabulary around career preference, staffing-to-workload fit, job sharing, learning communities, management quality, explanation, process modernization, and parliamentary-demand timing.

The source is not an adopted policy, not a causal evaluation, not a coefficient-calibration source, and not evidence that any simulated intervention works in a real ministry.
