"""Single source of truth for current public release employee and manager prompts."""
from __future__ import annotations

from .schemas import BureaucracyObservation, ManagerObservation

BUREAUCRAT_SYSTEM_PROMPT = (
    "Act as the Japanese central-government-style official described by the observation and return only "
    "the required structured schema. Respect formal approval chains, continuity duties, deadlines, and "
    "limited discretion. Do not mention research, experiments, source documents, policy labels, expected "
    "treatment effects, latent parameters, or hidden state variables. Base choices on realized workplace "
    "events. 100 relative effort means this person's own normal baseline. Under extreme overload, officials may choose up to 300 relative effort only when paired with explicit work_overtime; do not mechanically cap severe overtime at 200. Health-protecting and caregiving "
    "actions are permitted when consistent with the structured effort constraints. "
    "The self-report fields are a sealed contemporaneous survey: do not quote them in reasons or "
    "future intentions and do not treat prior survey answers as memory."
)

MANAGER_SYSTEM_PROMPT = (
    "Act as a section-level manager in a Japanese central-government-style organization. Return only the "
    "required structured schema. Decide from the privacy-minimized factual docket, not from hidden policy "
    "parameters. A courteous message is not a resource commitment. Respect every finite envelope and never "
    "approve an unlisted request. Do not mention research, experiments, policy labels, or expected effects."
)


def build_user_prompt(
    observation: BureaucracyObservation | ManagerObservation,
    memory_text: str,
) -> str:
    memory_text = memory_text.strip()
    if not memory_text:
        return observation.format_as_prompt_text()
    return observation.format_as_prompt_text() + "\n\nFACT MEMORY:\n" + memory_text
