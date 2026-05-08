"""FastAPI surface for the React frontend.

Phase 0 of the Gradio→React migration: every Gradio event handler in
``bracket/ui/app.py`` has a REST/WebSocket counterpart here, sharing the same
``OrchestrationSession`` singleton. The Gradio app is intentionally left
untouched so both UIs can run side-by-side throughout Phase 0/1.

Public entry points:
  - :func:`bracket.api.server.create_app` - FastAPI app factory.
  - :func:`bracket.api.cli.main`           - argparse CLI exposed as
                                             ``bracket-serve``.
"""

from __future__ import annotations

from bracket.api.server import create_app, get_session, serve

__all__ = ["create_app", "get_session", "serve"]
