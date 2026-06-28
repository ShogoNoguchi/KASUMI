"""Bounded identity-aware fact/narrative memory for current public release agents."""
from __future__ import annotations

from collections import deque

from shachi import BaseMemory


class BureaucratMemory(BaseMemory):
    """Separate objective fact memory from self-generated narrative memory."""

    def __init__(
        self,
        monthly_window: int = 6,
        quarterly_window: int = 4,
        fact_window: int = 18,
    ):
        self.facts = deque(maxlen=fact_window)
        self.monthly = deque(maxlen=monthly_window)
        self.quarterly = deque(maxlen=quarterly_window)

    def add_record(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            role = message.get("role", "monthly")
            content = message.get("content", "")
            if role == "fact":
                self.facts.append(content)
            elif role == "quarterly":
                self.quarterly.append(content)
            else:
                self.monthly.append(content)

    def retrieve(self, query: str | None = None) -> str:
        del query
        facts = "\n".join(f"- {item}" for item in self.facts)
        monthly = "\n".join(f"- {item}" for item in self.monthly)
        quarterly = "\n".join(f"- {item}" for item in self.quarterly)
        return (
            "OBJECTIVE FACT MEMORY (realized events only):\n"
            + (facts or "- none")
            + "\nSELF-NARRATIVE MONTHLY MEMORY:\n"
            + (monthly or "- none")
            + "\nSELF-NARRATIVE QUARTERLY MEMORY:\n"
            + (quarterly or "- none")
        )

    def clear(self) -> None:
        self.facts.clear()
        self.monthly.clear()
        self.quarterly.clear()


class ManagerFactMemory(BaseMemory):
    """Managers retain only compact factual docket/outcome summaries."""

    def __init__(self, window: int = 12):
        self.facts = deque(maxlen=window)

    def add_record(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            self.facts.append(message.get("content", ""))

    def retrieve(self, query: str | None = None) -> str:
        del query
        return "\n".join(f"- {item}" for item in self.facts) or "- none"

    def clear(self) -> None:
        self.facts.clear()
