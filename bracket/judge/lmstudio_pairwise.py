"""LMStudio-backed PairwiseJudge for finals-stage Bradley-Terry ranking.

Mirrors :mod:`bracket.judge.lmstudio` (single-image axis judge) but
constrains the model to a two-choice (plus tie) verdict per shared
prompt. The strict JSON schema is the same trick that made the
single-image judge ~30% more reliable: llama.cpp's grammar enforcement
keeps the model from rambling and short-circuits the parser.

Cost: one chat call per pair. A round-robin tournament on ``top_k=5``
finalists with 4 prompts = 40 calls. The orchestrator gates this behind
``enable_pairwise_finals`` because that's 7-10x the judge load of the
single-image pass.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from bracket.judge.base import PairwiseOutcome

logger = logging.getLogger("bracket.judge.lmstudio_pairwise")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


_DEFAULT_PROMPT_TEMPLATE = (
    "You are comparing two images, A and B, generated from the same text "
    "prompt by two different fine-tuned models.\n\n"
    'TEXT PROMPT GIVEN TO THE MODEL:\n"{prompt}"\n\n'
    "Decide which image is a better realisation of the prompt overall, "
    "considering prompt adherence, visual quality, and absence of AI "
    "artifacts.\n\n"
    "Reply with EXACTLY one JSON object on a single line, no other text:\n"
    '{{"winner": "a" | "b" | "tie", "reason": "<one short sentence>"}}'
)


@dataclass(frozen=True)
class LMStudioPairwiseConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen3-vl-8b-thinking-abliterated"
    timeout_s: float = 120.0
    max_tokens: int = 1024
    temperature: float = 0.0
    prompt_template: str = _DEFAULT_PROMPT_TEMPLATE
    use_json_schema: bool = True
    enable_thinking: Optional[bool] = None


class LMStudioPairwiseJudge:
    """A :class:`bracket.judge.base.PairwiseJudge` backed by LMStudio.

    Why a separate class instead of overloading :class:`LMStudioJudge`:
    the request shape, schema, and response parsing are different, and
    a smaller surface is easier to mock in tests.
    """

    _SCHEMA: dict = {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["a", "b", "tie"]},
            "reason": {"type": "string"},
        },
        "required": ["winner"],
        "additionalProperties": False,
    }

    def __init__(self, config: Optional[LMStudioPairwiseConfig] = None) -> None:
        self.config = config or LMStudioPairwiseConfig()

    def judge_pair(
        self, image_a: Path, image_b: Path, prompt: str,
    ) -> PairwiseOutcome:
        try:
            url_a = self._encode_image(Path(image_a))
            url_b = self._encode_image(Path(image_b))
            raw = self._post(self._build_payload(url_a, url_b, prompt))
            text = self._extract_choice_text(raw)
            verdict = self.parse_verdict(text)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "pairwise judge: pair (%s, %s) failed — %s; recording as tie.",
                image_a.name, image_b.name, e,
            )
            return 0
        if verdict == "a":
            return -1
        if verdict == "b":
            return 1
        return 0

    def eject(self) -> None:
        """Delegate to :class:`LMStudioJudge.eject` so we share the same
        ``lms unload <model>`` path. Best-effort; never raises."""
        try:
            from bracket.judge.lmstudio import LMStudioJudge, LMStudioJudgeConfig
            stub = LMStudioJudge(LMStudioJudgeConfig(
                base_url=self.config.base_url, model=self.config.model,
            ))
            stub.eject()
        except Exception:  # noqa: BLE001
            logger.exception("pairwise judge: eject failed; ignoring")

    # ------------------------------------------------------------------

    def _encode_image(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix or "png"
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/{mime};base64,{b64}"

    def _build_payload(
        self, image_a_url: str, image_b_url: str, prompt: str,
    ) -> dict:
        instruction = self.config.prompt_template.format(prompt=prompt)
        payload: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "text", "text": "Image A:"},
                        {"type": "image_url", "image_url": {"url": image_a_url}},
                        {"type": "text", "text": "Image B:"},
                        {"type": "image_url", "image_url": {"url": image_b_url}},
                    ],
                },
            ],
        }
        if self.config.use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "pairwise_verdict",
                    "strict": True,
                    "schema": self._SCHEMA,
                },
            }
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(self.config.enable_thinking),
            }
        return payload

    def _post(self, payload: dict) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")

    @staticmethod
    def _extract_choice_text(raw_body: str) -> str:
        obj = json.loads(raw_body)
        choices = obj.get("choices") or []
        if not choices:
            raise ValueError(f"empty choices in pairwise response: {raw_body[:200]!r}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            raise ValueError(f"no message content: {raw_body[:200]!r}")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text") or "")
            return ""
        return str(content)

    @staticmethod
    def parse_verdict(text: str) -> Literal["a", "b", "tie"]:
        """Extract the winner field from the model's JSON response."""
        cleaned = _THINK_RE.sub("", text).strip()
        for candidate in (cleaned, *_JSON_RE.findall(cleaned)):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            winner = obj.get("winner")
            if isinstance(winner, str):
                w = winner.strip().lower()
                if w in ("a", "b", "tie"):
                    return w  # type: ignore[return-value]
        raise ValueError(f"could not parse pairwise verdict: {text[:300]!r}")
