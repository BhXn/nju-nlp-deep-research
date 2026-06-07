import json
import re
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
        random_tool_call_id,
    )
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParser,
        ToolParserManager,
    )
    try:
        from vllm.transformers_utils.tokenizer import AnyTokenizer
    except Exception:  # pragma: no cover - version-specific fallback
        AnyTokenizer = Any
except Exception:  # pragma: no cover - fallback for older/newer vLLM layouts
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
        random_tool_call_id,
    )
    from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
    try:
        from vllm.tokenizers import TokenizerLike as AnyTokenizer
    except Exception:  # pragma: no cover - version-specific fallback
        AnyTokenizer = Any


OPEN_TAG = "[unused11]"
CLOSE_TAG = "[unused12]"
THINK_OPEN = "[unused16]"
THINK_CLOSE = "[unused17]"
FENCED_JSON_PATTERN = re.compile(r"```json\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)


@ToolParserManager.register_module(["pangu_deepdiver", "deepdiver_compat"])
class PanguDeepDiverToolParser(ToolParser):
    """Parse DeepDiver/Pangu tool calls with several fallback formats."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self._open_tag = OPEN_TAG
        self._close_tag = CLOSE_TAG

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tool_calls = self._parse_tool_calls(model_output)
        content = self._strip_known_wrappers(model_output)
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Optional[DeltaMessage]:
        if self._open_tag not in current_text:
            return DeltaMessage(content=delta_text)

        first_open = current_text.find(self._open_tag)
        first_close = current_text.find(self._close_tag, first_open + len(self._open_tag))

        if first_close == -1:
            if first_open > len(previous_text):
                new_content = current_text[len(previous_text) : first_open]
                if new_content.strip():
                    return DeltaMessage(content=new_content)
            return None

        if first_open > len(previous_text):
            new_content = current_text[len(previous_text) : first_open]
            if new_content.strip():
                return DeltaMessage(content=new_content)

        tool_calls = self._parse_tool_calls(current_text)
        if not tool_calls:
            return None

        delta_tool_calls = []
        for index, call in enumerate(tool_calls):
            arguments = call.function.arguments if call.function else "{}"
            delta_tool_calls.append(
                DeltaToolCall(
                    index=index,
                    id=random_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.function.name if call.function else "",
                        arguments=arguments,
                    ),
                )
            )

        content = self._strip_known_wrappers(current_text)
        return DeltaMessage(
            content=content if content else None,
            tool_calls=delta_tool_calls,
        )

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        calls: List[ToolCall] = []
        seen = set()
        for payload in self._extract_payloads(text):
            normalized = self._normalize_payload(payload)
            for item in normalized:
                name, arguments = self._normalize_call(item)
                if not name:
                    continue
                dedupe_key = json.dumps(
                    {"name": name, "arguments": arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )
        return calls

    def _extract_payloads(self, text: str) -> List[Any]:
        payloads: List[Any] = []
        candidates: List[str] = []

        for match in re.finditer(
            re.escape(self._open_tag) + r"([\s\S]*?)" + re.escape(self._close_tag),
            text,
            flags=re.IGNORECASE,
        ):
            raw_payload = match.group(1).strip()
            if raw_payload:
                candidates.append(raw_payload)

        for match in FENCED_JSON_PATTERN.finditer(text):
            raw_payload = match.group(1).strip()
            if raw_payload:
                candidates.append(raw_payload)

        stripped = self._strip_tool_markers(self._strip_known_wrappers(text))
        if stripped.startswith("{") or stripped.startswith("["):
            candidates.append(stripped)

        for candidate in candidates:
            parsed = self._safe_json_load(candidate)
            if parsed is not None:
                payloads.append(parsed)
        return payloads

    def _normalize_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            if "tool_calls" in payload:
                return self._normalize_payload(payload["tool_calls"])
            if "name" in payload or "function" in payload or "tool" in payload or "tool_name" in payload:
                return [payload]
        if isinstance(payload, list):
            items: List[Dict[str, Any]] = []
            for entry in payload:
                if isinstance(entry, dict):
                    if "tool_calls" in entry or "name" in entry or "function" in entry or "tool" in entry or "tool_name" in entry:
                        items.extend(self._normalize_payload(entry))
            return items
        return []

    @staticmethod
    def _normalize_call(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if "function" in item:
            if isinstance(item["function"], dict):
                function = item["function"]
                name = function.get(
                    "name",
                    item.get("name", item.get("tool", item.get("tool_name", ""))),
                )
                arguments = function.get(
                    "arguments",
                    item.get("arguments", item.get("parameters", {})),
                )
            else:
                name = str(item.get("function", item.get("name", item.get("tool", item.get("tool_name", "")))))
                arguments = item.get("arguments", item.get("parameters", {}))
        else:
            name = item.get("name", item.get("tool", item.get("tool_name", "")))
            arguments = item.get("arguments", item.get("parameters", {}))

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"raw": arguments}

        if arguments is None:
            arguments = {}

        return name, arguments

    @staticmethod
    def _strip_known_wrappers(text: str) -> str:
        cleaned = re.sub(
            re.escape(OPEN_TAG) + r"[\s\S]*?" + re.escape(CLOSE_TAG),
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            re.escape(THINK_OPEN) + r"[\s\S]*?" + re.escape(THINK_CLOSE),
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = FENCED_JSON_PATTERN.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _strip_tool_markers(text: str) -> str:
        cleaned = re.sub(r"\*\*调用工具：\*\*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"调用工具：", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\*\*tool call:\*\*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"tool call:", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _safe_json_load(raw_text: str) -> Optional[Any]:
        try:
            return json.loads(raw_text)
        except Exception:
            return None
