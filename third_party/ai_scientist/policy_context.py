"""Optional context and contract hooks for protected domain templates."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any



def _load_local_source_contexts(base_dir: Path) -> str:
    context_dir = base_dir / "source_contexts"
    if not context_dir.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(context_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(f"## {path.name}\n\n{text}")
    if not chunks:
        return ""
    return "\n\n<local_source_contexts>\n" + "\n\n".join(chunks) + "\n</local_source_contexts>\n"


def _load_legacy_source_briefs(base_dir: Path) -> str:
    brief_dir = base_dir / "source_briefs"
    if not brief_dir.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(brief_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append(f"## {path.name}\n\n{text}")
    if not chunks:
        return ""
    return "\n\n<local_source_briefs>\n" + "\n\n".join(chunks) + "\n</local_source_briefs>\n"

def _load_source_cards(base: Path) -> str:
    cards_dir = base / "source_cards"
    if not cards_dir.is_dir():
        return ""
    chunks: list[str] = []
    for card in sorted(cards_dir.glob("*.md")):
        body = card.read_text(encoding="utf-8").strip()
        if body:
            tag = "source_card_" + card.stem.replace("-", "_")
            chunks.append(f"<{tag}>\n{body}\n</{tag}>")
    return "\n\n" + "\n\n".join(chunks) + "\n" if chunks else ""


def load_problem_context(base_dir: str | Path) -> str:
    """Load the scientist-visible model brief and curated source cards.

    The AI Scientist is not expected to infer this environment by reading every
    implementation file, nor to rediscover Japanese public-service source PDFs
    through live literature search.  This bounded bundle gives it the model
    interface, permitted policy space, and curated Japanese context while keeping
    executable contracts under Python control.
    """
    base = Path(base_dir)
    sections: list[str] = []
    brief = base / "problem_context.md"
    if brief.exists():
        body = brief.read_text(encoding="utf-8").strip()
        if body:
            sections.append(f"<scientist_brief>\n{body}\n</scientist_brief>")
    model_card = base / "scientist_model_card.md"
    if model_card.exists():
        body = model_card.read_text(encoding="utf-8").strip()
        if body:
            sections.append(f"<scientist_model_card>\n{body}\n</scientist_model_card>")
    cards = _load_source_cards(base).strip()
    if cards:
        sections.append(cards)
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections) + "\n"

def enrich_task_description(base_dir: str | Path, description: str) -> str:
    return description + load_problem_context(base_dir)


def get_aider_model_name(model: str) -> str:
    # Aider/LiteLLM distinguishes Google AI Studio from Vertex with this prefix.
    if model.startswith("gemini-"):
        return f"gemini/{model}"
    return model


def _load_template_contract(folder_name: str | Path):
    candidate = Path(folder_name).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while not (candidate / "template_contract.py").exists() and candidate != candidate.parent:
        candidate = candidate.parent
    contract_path = candidate / "template_contract.py"
    if not contract_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "_ai_scientist_template_contract", contract_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load template contract: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scientist_visible_results(folder_name: str | Path, payload: Any) -> Any:
    module = _load_template_contract(folder_name)
    if module is None:
        return payload
    sanitizer = getattr(module, "scientist_visible_results", None)
    return payload if sanitizer is None else sanitizer(payload)


def validate_template_run(
    folder_name: str | Path, run_num: int
) -> dict[str, float | None] | None:
    module = _load_template_contract(folder_name)
    if module is None:
        return None
    validator = getattr(module, "validate_run", None)
    if validator is None:
        return None
    return validator(Path(folder_name) / f"run_{run_num}", run_num=run_num)


def render_template_idea_prompt(
    base_dir: str | Path,
    *,
    task_description: str,
    previous_ideas: str,
    num_reflections: int,
) -> str | None:
    module = _load_template_contract(base_dir)
    renderer = None if module is None else getattr(module, "render_idea_prompt", None)
    if renderer is None:
        return None
    return renderer(
        task_description=task_description,
        previous_ideas=previous_ideas,
        num_reflections=num_reflections,
    )


def render_template_idea_reflection_prompt(
    base_dir: str | Path, *, current_round: int, num_reflections: int
) -> str | None:
    module = _load_template_contract(base_dir)
    renderer = None if module is None else getattr(
        module, "render_idea_reflection_prompt", None
    )
    if renderer is None:
        return None
    return renderer(current_round=current_round, num_reflections=num_reflections)


def render_template_novelty_context(
    base_dir: str | Path, *, task_description: str
) -> str | None:
    module = _load_template_contract(base_dir)
    renderer = None if module is None else getattr(module, "render_novelty_context", None)
    if renderer is None:
        return None
    return renderer(task_description=task_description)


def validate_template_idea(base_dir: str | Path, idea: Any) -> Any:
    module = _load_template_contract(base_dir)
    validator = None if module is None else getattr(module, "validate_idea", None)
    return idea if validator is None else validator(idea)


def is_protected_policy_template(folder_name: str | Path) -> bool:
    module = _load_template_contract(folder_name)
    return module is not None and hasattr(module, "validate_idea")
