"""The contract every specialised agent implements.

Subclasses supply only `_execute`; the base class handles timing, error capture
and the uniform result shape. An agent that raises is recorded as failed and the
workflow continues, because one broken stage should degrade the analysis rather
than destroy the run.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from state import AgentOutput, WorkflowState


class Agent(ABC):
    """Base class for all specialised agents.

    Attributes:
        name: Stable identifier used as the key in `WorkflowState.outputs` and in
            the orchestrator's plan.
        description: One line describing the agent's responsibility. The
            orchestrator includes these when planning, so it is a prompt input,
            not just documentation.
        requires: Agents whose findings this one needs. The orchestrator uses
            this to order the plan and to avoid scheduling an agent whose
            dependencies were skipped.
    """

    name: str = "agent"
    description: str = ""
    requires: tuple[str, ...] = ()

    def run(self, state: WorkflowState) -> AgentOutput:
        state.log("agent_started", agent=self.name)
        t0 = time.perf_counter()

        missing = [dep for dep in self.requires if not state.has_run(dep)]
        if missing:
            return AgentOutput(
                agent=self.name,
                status="failed",
                error=f"Missing required upstream results: {', '.join(missing)}.",
                duration_s=time.perf_counter() - t0,
            )

        try:
            output = self._execute(state)
        except Exception as exc:
            return AgentOutput(
                agent=self.name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - t0,
            )

        output.agent = self.name
        output.duration_s = time.perf_counter() - t0
        return output

    @abstractmethod
    def _execute(self, state: WorkflowState) -> AgentOutput:
        """Do the agent's work. Called only when dependencies are satisfied."""

    # -- helpers available to every subclass -------------------------------- #

    @staticmethod
    def _degraded(findings: dict[str, Any], narrative: str,
                  evidence: dict[str, Any], warnings: list[str]) -> AgentOutput:
        """Build a result for a stage that produced usable but incomplete output."""
        return AgentOutput(
            agent="", status="degraded", findings=findings,
            narrative=narrative, evidence=evidence, warnings=warnings,
        )
