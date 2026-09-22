"""Maisaka 规划器消息构造工具。"""

from datetime import datetime
from html import escape
from typing import List, Optional, Sequence, Set

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.message_component_data_model import (
    MessageSequence,
    ReplyComponent,
    TextComponent,
)

from .identity import build_participant_identity, participant_identity_attributes
from .message_adapter import format_speaker_content
from .messages import SessionBackedMessage


def build_planner_prefix(
    *,
    timestamp: datetime,
    user_name: str,
    platform: str = "",
    user_id: str = "",
    group_card: str = "",
    message_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    quote_ids: Optional[Sequence[str]] = None,
    include_message_id: bool = True,
    include_chat_id: bool = False,
    is_self_message: bool = False,
    prefer_nickname: bool = False,
) -> str:
    """构造 Maisaka 规划器使用的统一消息前缀。

    Args:
        timestamp: 消息时间。
        user_name: 展示给规划器的用户名。
        platform: 发送者所在平台。
        user_id: 平台上的稳定账号 ID。
        group_card: 群昵称。
        message_id: 消息 ID。
        chat_id: 聊天流 ID。
        quote_ids: 被引用消息 ID 列表。
        include_message_id: 是否输出 `msg_id` 段。
        include_chat_id: 是否输出 `chat_id` 段。
        is_self_message: 是否显式标注这条消息是 bot 自己发送的。
        prefer_nickname: 回复生成时以称呼为主，同时保留身份元数据。

    Returns:
        str: 拼接完成的规划器前缀。
    """

    identity = build_participant_identity(platform=platform, user_id=user_id, nickname=user_name, group_card=group_card)
    is_self_message = is_self_message or identity.is_self
    # 无真实账号的自身写回消息用明确的 self 标记，不能用配置昵称猜测账号。
    user_reference = identity.person_id or ("self" if is_self_message else "unknown")
    if prefer_nickname and identity.display_name:
        user_reference = identity.display_name
    message_attrs: List[str] = []
    if include_message_id:
        message_attrs.append(f'msg_id="{escape(message_id or "", quote=True)}"')

    if include_chat_id:
        normalized_chat_id = str(chat_id or "").strip()
        if normalized_chat_id:
            message_attrs.append(f'chat_id="{escape(normalized_chat_id, quote=True)}"')

    normalized_quote = _format_quote_ids(quote_ids)
    if normalized_quote:
        message_attrs.append(f'quote="{escape(normalized_quote, quote=True)}"')

    message_attrs.extend(
        [
            f'time="{escape(timestamp.strftime("%H:%M:%S"), quote=True)}"',
            f'user="{escape(user_reference, quote=True)}"',
        ]
    )

    message_attrs.extend(participant_identity_attributes(identity))
    if is_self_message and not identity.is_self:
        message_attrs.append('is_self_message="true"')
    return f"<message {' '.join(message_attrs)}>\n"


def _format_quote_ids(quote_ids: Optional[Sequence[str]]) -> str:
    """将引用消息 ID 列表格式化为 XML 属性值。"""

    if not quote_ids:
        return ""

    normalized_ids: List[str] = []
    seen: Set[str] = set()
    for raw_quote_id in quote_ids:
        quote_id = str(raw_quote_id or "").strip()
        if not quote_id or quote_id in seen:
            continue
        seen.add(quote_id)
        normalized_ids.append(quote_id)
    return ",".join(normalized_ids)


def extract_quote_ids_from_message_sequence(message_sequence: MessageSequence) -> list[str]:
    """从消息片段中提取引用目标 ID，供 prompt 元信息使用。"""

    quote_ids: list[str] = []
    seen: set[str] = set()
    for component in message_sequence.components:
        if not isinstance(component, ReplyComponent):
            continue

        quote_id = component.target_message_id.strip()
        if not quote_id or quote_id in seen:
            continue
        seen.add(quote_id)
        quote_ids.append(quote_id)
    return quote_ids


def build_planner_user_prefix_from_session_message(
    message: SessionMessage,
    *,
    include_message_id: bool = True,
    include_chat_id: bool = False,
    is_self_message: bool = False,
    prefer_nickname: bool = False,
) -> str:
    """根据真实会话消息构造规划器前缀。

    Args:
        message: 原始会话消息。
        include_message_id: 是否输出 `msg_id` 段。
        include_chat_id: 是否输出 `chat_id` 段。
        is_self_message: 是否显式标注这条消息是 bot 自己发送的。
        prefer_nickname: 回复生成使用昵称优先的展示。

    Returns:
        str: 规划器前缀字符串。
    """

    user_info = message.message_info.user_info
    user_name = user_info.user_nickname or user_info.user_id
    return build_planner_prefix(
        timestamp=message.timestamp,
        user_name=user_name,
        platform=message.platform,
        user_id=user_info.user_id,
        group_card=user_info.user_cardname or "",
        message_id=message.message_id,
        chat_id=message.session_id,
        quote_ids=extract_quote_ids_from_message_sequence(message.raw_message),
        include_message_id=include_message_id and not message.is_notify and bool(message.message_id),
        include_chat_id=include_chat_id,
        is_self_message=is_self_message,
        prefer_nickname=prefer_nickname,
    )


def build_session_backed_text_message(
    *,
    speaker_name: str,
    text: str,
    timestamp: datetime,
    source_kind: str,
    platform: str = "",
    user_id: str = "",
    group_card: str = "",
    message_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    quote_ids: Optional[Sequence[str]] = None,
    include_message_id: bool = True,
    include_chat_id: bool = False,
    is_self_message: bool = False,
) -> SessionBackedMessage:
    """构造带规划器前缀的纯文本历史消息。

    Args:
        speaker_name: 发言者名称。
        text: 发言内容。
        timestamp: 发言时间。
        source_kind: 上下文来源类型。
        platform: 发送者平台；只有真实账号已知时提供。
        user_id: 发送者平台账号；不从昵称推断。
        group_card: 群昵称。
        message_id: 消息 ID。
        chat_id: 聊天流 ID。
        quote_ids: 被引用消息 ID 列表。
        include_message_id: 是否输出 `msg_id` 段。
        include_chat_id: 是否输出 `chat_id` 段。
        is_self_message: 是否显式标注这条消息是 bot 自己发送的。

    Returns:
        SessionBackedMessage: 可直接写入历史的上下文消息。
    """

    planner_prefix = build_planner_prefix(
        timestamp=timestamp,
        user_name=speaker_name,
        platform=platform,
        user_id=user_id,
        group_card=group_card,
        message_id=message_id,
        chat_id=chat_id,
        quote_ids=quote_ids,
        include_message_id=include_message_id,
        include_chat_id=include_chat_id,
        is_self_message=is_self_message,
    )
    return SessionBackedMessage(
        raw_message=MessageSequence([TextComponent(f"{planner_prefix}{text}")]),
        visible_text=format_speaker_content(
            speaker_name,
            text,
            timestamp,
            message_id if include_message_id else None,
        ),
        timestamp=timestamp,
        message_id=message_id,
        source_kind=source_kind,
        participant_identity=build_participant_identity(
            platform=platform, user_id=user_id, nickname=speaker_name, group_card=group_card
        ),
    )
