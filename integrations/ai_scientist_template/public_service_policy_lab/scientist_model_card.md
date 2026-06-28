# Scientist model card: Shachi-based Japan Policy Scientist environment

This card is scientist-visible. It explains the reusable scientific interface so the AI Scientist does not need to infer Shachi semantics from raw source code.

## Shachi role

Shachi is used as the agent-and-environment substrate. This package contributes a new employee agent and a new central-government-style organization environment. Shachi is not itself the empirical object of study; it is the execution framework that repeatedly calls structured LLM employees and then applies Python-owned state transitions.

## Employee agent

Each active employee-month has one schema-constrained LLM action. The employee sees a bounded observation: current department conditions, deadline pressure, valid realized events, personal fact memory, and synthetic profile attributes. The employee can choose relative effort, time allocation, health-protective behavior, support or explanation requests, process-reform voice, career action, and transfer preferences. The employee also gives contemporaneous survey outputs for evaluation.

The employee LLM does not choose absolute output, task arrivals, deadline severity, headcount, management approvals, policy effects, exit probabilities, transfer capacity, hiring, or selection scores.

## Python environment

Python owns the organization physics: authorized slots, dynamic person identities, identity epochs, task queues, deadlines, quality thresholds, work units, service-harm points, terminal liability, modeled strain, knowledge stocks, competing-risk exit, batch hiring, simultaneous transfer planning, deterministic finite management allocation, event delivery, and metrics.

The service month closes on the frozen start-of-month workforce. Exits, transfers, and hiring can change future months, but person-specific management effects are applied only after revalidating slot, person ID, identity epoch, active status, and department. A departed person's request cannot benefit a replacement.

## Policy realization

The AI Scientist may edit only numeric fields listed in `policy_space.json`. A field can matter only through its listed realization path: request, eligibility draw, finite allocation, implementation, transfer or hiring capacity, or next-month exposure. Policy labels and hypotheses never directly produce welfare.

## Evidence lifecycle

Development compares `run_0` with four exact intervention mappings under matched shocks. Selection is allowed only for a guardrail-passing intervention that strictly improves the preregistered staff endpoint over the reference. If selected, the exact policy payload is frozen; current public release fast-track does not run additional paid holdout cells. The final manuscript is written after selection and holdout status, not before.

## Claim boundary

The simulation can identify mechanisms and failure modes inside the declared synthetic organization. It cannot estimate real-ministry treatment effects, validate policy recommendations, or calibrate Japanese civil-service behavior.
