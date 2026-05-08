"""Backend session-state and snapshot helpers.

Despite the package name, this module is no longer about a UI — the React
frontend now lives at ``frontend/`` and the Python side just exposes session
state and snapshot building for the FastAPI server in :mod:`bracket.api`.

Kept under ``bracket.ui`` for v0.1 import-path compatibility; will move to
``bracket.session`` / ``bracket.snapshot`` in v0.2.
"""
from bracket.ui.session import OrchestrationSession, SessionState

__all__ = ["OrchestrationSession", "SessionState"]
