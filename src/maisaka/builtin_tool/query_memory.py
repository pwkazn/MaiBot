"""query_memory 内置工具。"""

from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Optional, Tuple

import re

from src.common.logger import get_logger
from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.utils.tool_post_execution import with_memory_feedback_task
from src.person_info.person_info import get_person_name_by_person_id, resolve_person_id_for_memory
from src.services.memory_service import MemorySearchResult, memory_service

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_query_memory")

_ALLOWED_QUERY_MODES = {"search", "time", "hybrid", "episode", "aggregate"}
_ISO_QUERY_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?$")
REPLYER_MEMORY_REFERENCE_MARKER = "【长期记忆检索结果-内部参考】"


def get_tool_spec(*, enabled: bool = True) -> ToolSpec:
    """获取 query_memory 工具声明。"""

    return ToolSpec(
        name="query_memory",
        description="检索长期记忆。",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "关键词或问题；非纯时间检索必填。",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回条数。",
                },
                "mode": {
                    "type": "string",
                    "description": "search事实偏好，time时间段，episode经历，aggregate整体，hybrid不确定。",
                    "enum": sorted(_ALLOWED_QUERY_MODES),
                    "default": "search",
                },
                "person_id": {
                    "type": "string",
                    "description": "人物稳定 ID；定向检索优先填写上下文中的 person_id。",
                },
                "person_name": {
                    "type": "string",
                    "description": "仅在不知道 person_id 时按完整名称查找；重名或未知名称会报错。",
                },
                "time_start": {
                    "type": "string",
                    "description": "起始时间。",
                },
                "time_end": {
                    "type": "string",
                    "description": "结束时间。",
                },
                "respect_filter": {
                    "type": "boolean",
                    "description": "是否遵守记忆过滤规则；默认true，模糊来源或整体印象可false。",
                    "default": True,
                },
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        enabled=enabled,
    )


def _normalize_optional_time(raw_value: Any) -> str | float | None:
    """归一化可选时间参数。"""

    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        time_text = raw_value.strip()
        if not time_text:
            return None
        if _ISO_QUERY_TIME_RE.fullmatch(time_text):
            return time_text.replace("-", "/", 2)
        return time_text
    if isinstance(raw_value, (float, int)):
        return float(raw_value)

    time_text = str(raw_value).strip()
    if not time_text:
        return None
    return time_text


def _resolve_person_id(
    *,
    person_id: str = "",
    person_name: str,
    platform: str,
    user_id: str,
    group_id: str,
) -> Tuple[str, str]:
    """优先使用明确的人物 ID，名称查找失败不能改查当前私聊对象。"""

    clean_person_name = str(person_name or "").strip()
    clean_person_id = person_id.strip()
    if clean_person_id:
        return clean_person_id, get_person_name_by_person_id(clean_person_id) or clean_person_name
    if clean_person_name:
        person_id = resolve_person_id_for_memory(
            person_name=clean_person_name,
        )
        if not person_id:
            raise ValueError(f"未找到人物“{clean_person_name}”，请使用上下文中的 person_id。")
        return person_id, get_person_name_by_person_id(person_id) or clean_person_name

    if not group_id and platform and user_id:
        person_id = resolve_person_id_for_memory(
            platform=platform,
            user_id=user_id,
        )
        if person_id:
            return person_id, get_person_name_by_person_id(person_id)

    return "", clean_person_name


def _format_memory_hit_identities(metadata: Dict[str, Any], *, source: str = "") -> str:
    """仅序列化命中自带的人物 ID，不根据记忆正文或昵称猜测归属。"""
    person_ids: List[str] = []
    person_id = metadata.get("person_id")
    if isinstance(person_id, str) and person_id.strip():
        person_ids.append(person_id.strip())
    metadata_person_ids = metadata.get("person_ids")
    if isinstance(metadata_person_ids, list):
        for person_id in metadata_person_ids:
            if isinstance(person_id, str) and person_id.strip() and person_id.strip() not in person_ids:
                person_ids.append(person_id.strip())
    for source_value in (source, metadata.get("source")):
        if isinstance(source_value, str) and source_value.strip().startswith("person_fact:"):
            person_id = source_value.strip()[len("person_fact:") :].strip()
            if person_id and person_id not in person_ids:
                person_ids.append(person_id)
    return " ".join(f'<person person_id="{escape(person_id, quote=True)}"/>' for person_id in person_ids)


def _build_success_content(result: MemorySearchResult, *, limit: int) -> str:
    """构造工具成功时的可读内容。"""

    summary = str(result.summary or "").strip()
    hit_lines: List[str] = []
    for index, hit in enumerate(result.hits[: max(1, int(limit))], start=1):
        identities = _format_memory_hit_identities(hit.metadata, source=hit.source)
        content = hit.content.strip().replace("\n", " ")
        hit_lines.append(f"{index}. {identities} {content}" if identities else f"{index}. {content}")
    snippet = "\n".join(hit_lines)

    if result.hits:
        if snippet:
            return snippet
        if summary:
            return summary
        return "已找到匹配的长期记忆。"

    if result.filtered:
        return "当前请求被聊天过滤策略跳过，未执行长期记忆检索。"
    return "未找到匹配的长期记忆。"


def _build_replyer_memory_reference(structured_content: Dict[str, Any]) -> str:
    """构造自动透传给 replyer 的长期记忆参考。"""

    raw_hits = structured_content.get("hits")
    if not isinstance(raw_hits, list):
        return ""

    lines = [REPLYER_MEMORY_REFERENCE_MARKER]
    person_id = str(structured_content.get("person_id") or "").strip()
    person_name = str(structured_content.get("person_name") or "").strip()
    if person_id:
        lines.append(f"人物：person_id={person_id}；昵称={person_name or '未记录昵称'}")
    query = str(structured_content.get("query") or "").strip()
    mode = str(structured_content.get("mode") or "").strip()
    effective_mode = str(structured_content.get("effective_mode") or "").strip()
    if query:
        lines.append(f"查询：{query}")
    if mode:
        mode_text = mode
        if effective_mode and effective_mode != mode:
            mode_text = f"{mode} -> {effective_mode}"
        lines.append(f"模式：{mode_text}")

    hit_lines: List[str] = []
    for index, raw_hit in enumerate(raw_hits, start=1):
        if not isinstance(raw_hit, dict):
            continue
        content = str(raw_hit.get("content") or "").strip()
        if not content:
            continue
        hit_type = str(raw_hit.get("type") or "").strip()
        title = str(raw_hit.get("title") or "").strip()
        label_parts = [part for part in (title, hit_type) if part]
        label = f"（{' / '.join(label_parts)}）" if label_parts else ""
        metadata = raw_hit.get("metadata")
        source = raw_hit.get("source")
        identities = _format_memory_hit_identities(
            metadata if isinstance(metadata, dict) else {},
            source=source if isinstance(source, str) else "",
        )
        identity_prefix = f"{identities} " if identities else ""
        normalized_content = " ".join(content.split())
        hit_lines.append(f"{index}. {identity_prefix}{label}{normalized_content}")

    if not hit_lines:
        return ""

    lines.append(f"命中：{len(hit_lines)} 条")
    lines.extend(hit_lines)
    return "\n".join(lines)


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 query_memory 内置工具。"""

    del context
    runtime = tool_ctx.runtime
    chat_stream = runtime.chat_stream

    clean_query = str(invocation.arguments.get("query") or "").strip()
    mode = str(invocation.arguments.get("mode") or "search").strip().lower() or "search"
    if mode not in _ALLOWED_QUERY_MODES:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"不支持的检索模式：{mode}。可选值：search/time/hybrid/episode/aggregate。",
        )

    default_limit = max(1, global_config.a_memorix.integration.memory_query_default_limit)
    try:
        limit = int(invocation.arguments.get("limit", default_limit) or default_limit)
    except (TypeError, ValueError):
        limit = default_limit
    limit = max(1, min(limit, 20))

    time_start = _normalize_optional_time(invocation.arguments.get("time_start"))
    time_end = _normalize_optional_time(invocation.arguments.get("time_end"))
    if not clean_query and time_start is None and time_end is None:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "query_memory 需要提供 query，或至少提供 time_start/time_end 中的一个。",
        )

    session_id = str(runtime.session_id or "").strip()
    platform = str(chat_stream.platform or "").strip()
    user_id = str(chat_stream.user_id or "").strip()
    group_id = str(chat_stream.group_id or "").strip()
    try:
        person_id, person_name = _resolve_person_id(
            person_id=str(invocation.arguments.get("person_id") or ""),
            person_name=str(invocation.arguments.get("person_name") or ""),
            platform=platform,
            user_id=user_id,
            group_id=group_id,
        )
    except ValueError as exc:
        return tool_ctx.build_failure_result(invocation.tool_name, str(exc))
    respect_filter = bool(invocation.arguments.get("respect_filter", True))

    logger.info(
        f"{runtime.log_prefix} 触发长期记忆检索工具: "
        f"mode={mode} query={clean_query!r} person_name={person_name!r} person_id={person_id!r}"
    )
    try:
        result = await memory_service.search(
            clean_query,
            limit=limit,
            mode=mode,
            chat_id=session_id,
            person_id=person_id,
            time_start=time_start,
            time_end=time_end,
            respect_filter=respect_filter,
            user_id=user_id,
            group_id=group_id,
        )
    except Exception as exc:
        logger.exception(f"{runtime.log_prefix} 长期记忆检索执行异常: {exc}")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"长期记忆检索失败：{exc}",
        )

    # 保持人物和时间过滤，即使没有命中也不能把其他人物的记忆当成本次结果。
    structured_content: Dict[str, Any] = result.to_dict()
    structured_content.update(
        {
            "query": clean_query,
            "mode": mode,
            "effective_mode": mode,
            "limit": limit,
            "chat_id": session_id,
            "person_name": person_name,
            "person_id": person_id,
            "time_start": time_start,
            "time_end": time_end,
            "respect_filter": respect_filter,
            "user_id": user_id,
            "group_id": group_id,
            "fallback_applied": False,
            "fallback_reason": "",
            "fallback_query": "",
            "primary_hit_count": len(result.hits),
        }
    )

    if not result.success:
        error_message = str(result.error or "").strip() or "长期记忆检索失败。"
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            error_message,
            structured_content=structured_content,
        )

    content = _build_success_content(result, limit=limit)
    if person_id:
        content = f"人物：person_id={person_id}；昵称={person_name or '未记录昵称'}\n{content}"
    metadata: Dict[str, Any] = with_memory_feedback_task()
    replyer_memory_reference = _build_replyer_memory_reference(structured_content)
    if replyer_memory_reference:
        metadata["replyer_memory_reference"] = replyer_memory_reference

    return tool_ctx.build_success_result(
        invocation.tool_name,
        content,
        structured_content=structured_content,
        metadata=metadata,
    )
