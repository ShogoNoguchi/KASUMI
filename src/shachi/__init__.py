"""Tiny compatibility interfaces for the packaged public-service task extension.

The full agent framework is an optional external dependency for live LLM runs.
These classes allow static analysis, imports, and artifact replay to work without
vendoring a larger framework.
"""
from __future__ import annotations

from typing import Generic, TypeVar
import pydantic

T = TypeVar("T")
M = TypeVar("M")

class Message(pydantic.BaseModel):
    pass

class Observation(pydantic.BaseModel, Generic[M]):
    pass

class BaseMemory:
    pass

class Environment(Generic[T]):
    pass

class Agent(Generic[T]):
    pass
