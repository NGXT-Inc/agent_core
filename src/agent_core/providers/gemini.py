"""GeminiProvider — LLMProvider implementation for Google Gemini / Vertex AI.

Wraps the ``google-genai`` SDK, handling Content/Part message format,
FunctionDeclaration tool schemas, context caching integration, and
multimodal file attachments in tool responses.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from agent_core.providers.types import ParsedResponse, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class GeminiProvider:
    """LLMProvider for Google Gemini via the ``google-genai`` SDK.

    Args:
        client: Pre-configured ``genai.Client``. If ``None``, a new client
            is created from *project_id* and *location*.
        project_id: Google Cloud project ID (read from env if ``None``).
        location: Google Cloud location (default ``"global"``).
    """

    def __init__(
        self,
        client: genai.Client | None = None,
        project_id: str | None = None,
        location: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            pid = project_id or os.environ.get("GOOGLE_PROJECT_ID")
            loc = location or os.environ.get("GOOGLE_LOCATION", "global")
            if not pid:
                raise ValueError(
                    "GOOGLE_PROJECT_ID must be set in environment or passed explicitly"
                )
            self._client = genai.Client(
                vertexai=True, project=pid, location=loc
            )

    @property
    def client(self) -> genai.Client:
        """The underlying ``genai.Client`` (for backward compat and caching)."""
        return self._client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int,
        tool_schemas: Any | None = None,
        *,
        cache_config: dict | None = None,
    ) -> Any:
        if cache_config and cache_config.get("cache_name"):
            config = types.GenerateContentConfig(
                cached_content=cache_config["cache_name"],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )
            offset = cache_config.get("contents_offset", 0)
            contents = messages[offset:]
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=tool_schemas if tool_schemas else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            )
            contents = list(messages)  # copy — caller may mutate after return

        return self._client.models.generate_content(
            model=model, contents=contents, config=config,
        )

    def parse_response(self, response: Any) -> ParsedResponse:
        # Usage
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            usage.prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            usage.completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
            usage.cached_tokens = getattr(meta, "cached_content_token_count", 0) or 0

        # Empty response
        if not response.candidates or not response.candidates[0].content:
            return ParsedResponse(
                text=response.text or "",
                tool_calls=[],
                raw_message=None,
                usage=usage,
            )

        model_content = response.candidates[0].content

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in model_content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=uuid.uuid4().hex[:12],
                    name=fc.name,
                    args=dict(fc.args) if fc.args else {},
                ))
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        # Determine thinking vs. final text
        thinking_text = None
        text = None
        if text_parts and tool_calls:
            # Text alongside tool calls is intermediate reasoning
            thinking_text = "\n".join(text_parts)
        elif text_parts:
            text = "\n".join(text_parts)

        # Fall back to response.text for the final string if no parts parsed
        if text is None and not tool_calls:
            text = response.text or ""

        return ParsedResponse(
            text=text,
            tool_calls=tool_calls,
            raw_message=model_content,
            usage=usage,
            thinking_text=thinking_text,
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def build_user_message(self, text: str) -> Any:
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )

    def build_tool_result_messages(
        self,
        tool_calls: list[ToolCall],
        results: list[tuple[str, Any]],
    ) -> Any:
        """Build a single Content with function-response Parts.

        Supports multimodal responses — if a tool result contains ``"files"``
        or ``"images"`` keys with file attachments, they are included as
        additional Parts.

        Attachment modes:

        1. **Inline bytes** — ``{"data": bytes, "mime_type": "...", "description": "..."}``
        2. **GCS URI** — ``{"gcs_uri": "gs://...", "mime_type": "...", "description": "..."}``
        """
        parts: list[Any] = []

        for _tc, (func_name, result) in zip(tool_calls, results):
            file_attachments: list[dict] = []

            if isinstance(result, dict):
                file_attachments = (
                    (result.get("files") or []) + (result.get("images") or [])
                )
                response_data = {
                    k: v for k, v in result.items() if k not in ("files", "images")
                }
            else:
                response_data = {"result": str(result)}

            parts.append(
                types.Part.from_function_response(
                    name=func_name, response=response_data,
                )
            )

            for item in file_attachments:
                if not isinstance(item, dict):
                    continue

                mime_type = item.get("mime_type", "application/octet-stream")
                description = item.get("description", "")

                if description:
                    label = "PDF" if mime_type == "application/pdf" else "File"
                    parts.append(
                        types.Part.from_text(
                            text=f"[{label} from {func_name}: {description}]"
                        )
                    )

                if "gcs_uri" in item:
                    gcs_uri = item["gcs_uri"]
                    parts.append(
                        types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
                    )
                    logger.debug(
                        "Attached GCS file to context: %s (%s)", gcs_uri, mime_type
                    )
                elif "data" in item:
                    item_data = item["data"]
                    parts.append(
                        types.Part.from_bytes(data=item_data, mime_type=mime_type)
                    )
                    logger.debug(
                        "Attached inline %s to context: %s (%d bytes)",
                        mime_type, description or func_name, len(item_data),
                    )

        return types.Content(role="user", parts=parts)

    # ------------------------------------------------------------------
    # Tool schemas
    # ------------------------------------------------------------------

    def build_tool_schemas(self, callables: list[Callable]) -> Any | None:
        if not callables:
            return None
        decls = [
            types.FunctionDeclaration.from_callable(
                callable=f, client=self._client._api_client
            )
            for f in callables
        ]
        return [types.Tool(function_declarations=decls)]

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(
        self,
        model: str,
        messages: list[Any],
        system_prompt: str | None = None,
    ) -> int:
        try:
            contents: list[Any] = []
            if system_prompt:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_prompt)],
                    )
                )
            contents.extend(messages)
            resp = self._client.models.count_tokens(model=model, contents=contents)
            return resp.total_tokens
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def is_retryable_error(self, error: Exception) -> bool:
        return isinstance(error, ClientError) and error.code == 429

    def get_retry_delay(
        self,
        error: Exception,
        attempt: int,
        base_delay: float,
        max_delay: float,
    ) -> float:
        return min(base_delay * (2 ** (attempt - 1)), max_delay)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_message(self, message: Any) -> dict:
        """Serialize a Gemini Content object to a JSON-safe dict."""
        serialized_parts = []

        for part in message.parts:
            if hasattr(part, "text") and part.text is not None:
                serialized_parts.append({"type": "text", "text": part.text})

            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                serialized_parts.append({
                    "type": "function_call",
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })

            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                serialized_parts.append({
                    "type": "function_response",
                    "name": fr.name,
                    "response": (
                        fr.response
                        if isinstance(fr.response, dict)
                        else {"result": str(fr.response)}
                    ),
                })

            elif hasattr(part, "inline_data") and part.inline_data:
                serialized_parts.append({
                    "type": "inline_data",
                    "mime_type": getattr(part.inline_data, "mime_type", "unknown"),
                    "skipped": True,
                })

            elif hasattr(part, "thought") and part.thought:
                serialized_parts.append({
                    "type": "thought",
                    "thought": str(part.thought),
                })

        return {
            "_provider": "gemini",
            "role": message.role,
            "parts": serialized_parts,
        }

    def deserialize_message(self, data: dict) -> Any:
        """Deserialize a dict back into a Gemini Content object."""
        parts: list[Any] = []

        for part_data in data.get("parts", []):
            part_type = part_data.get("type")

            if part_type == "text":
                parts.append(types.Part.from_text(text=part_data.get("text", "")))

            elif part_type == "function_call":
                parts.append(types.Part.from_function_call(
                    name=part_data.get("name", ""),
                    args=part_data.get("args", {}),
                ))

            elif part_type == "function_response":
                parts.append(types.Part.from_function_response(
                    name=part_data.get("name", ""),
                    response=part_data.get("response", {}),
                ))

            elif part_type == "thought":
                thought_text = part_data.get("thought", "")
                if thought_text:
                    parts.append(types.Part.from_text(text=thought_text))

            elif part_type == "inline_data":
                pass  # Binary data intentionally skipped during serialization

            else:
                logger.warning(
                    "Unknown part type during deserialization: %s", part_type
                )

        return types.Content(role=data.get("role", "user"), parts=parts)

    def format_message_for_display(self, message: Any) -> dict | None:
        """Format a Gemini Content for human-readable display."""
        if message is None:
            return None

        role = message.role
        parts_text: list[str] = []

        for part in message.parts:
            if hasattr(part, "text") and part.text:
                parts_text.append(part.text)
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                parts_text.append(f"[Tool Call: {fc.name}({args})]")
            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                parts_text.append(f"[Tool Response: {fr.name} -> {fr.response}]")

        if parts_text:
            return {"role": role, "content": "\n".join(parts_text)}
        return None
