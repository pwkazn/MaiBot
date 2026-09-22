"""供推理、工具和记忆共享的参与者身份表示。"""

from dataclasses import dataclass
from html import escape
from typing import List

from src.person_info.person_info import get_person_id
from src.services.bot_account_service import get_bot_accounts, is_bot_self


@dataclass(frozen=True)
class ParticipantIdentity:
    """账号身份与当前称呼；昵称不参与身份计算。"""

    platform: str
    user_id: str
    person_id: str
    nickname: str
    group_card: str
    is_self: bool

    @property
    def display_name(self) -> str:
        return self.group_card or self.nickname or self.user_id


def build_participant_identity(
    *,
    platform: str,
    user_id: str,
    nickname: str = "",
    group_card: str = "",
) -> ParticipantIdentity:
    """沿用人物档案的稳定 ID，缺少账号时不根据昵称创造身份。"""

    platform = platform.strip()
    user_id = user_id.strip()
    has_account = bool(platform and user_id)
    return ParticipantIdentity(
        platform=platform,
        user_id=user_id,
        person_id=get_person_id(platform, user_id) if has_account else "",
        nickname=nickname.strip(),
        group_card=group_card.strip(),
        is_self=is_bot_self(platform, user_id) if has_account else False,
    )


def participant_identity_attributes(identity: ParticipantIdentity) -> List[str]:
    """序列化模型可见的身份元数据，并转义平台提供的可控文本。"""

    attributes = []
    for key, value in (
        ("person_id", identity.person_id),
        ("platform", identity.platform),
        ("user_id", identity.user_id),
        ("nickname", identity.nickname),
        ("group_card", identity.group_card),
    ):
        if value:
            attributes.append(f'{key}="{escape(value, quote=True)}"')
    if not identity.person_id:
        attributes.append('identity_unknown="true"')
    if identity.is_self:
        attributes.append('is_self_message="true"')
    return attributes


def format_participant_reference(
    *,
    platform: str,
    user_id: str,
    nickname: str = "",
    group_card: str = "",
) -> str:
    """返回以 ID 为主、保留可读称呼的内部人物引用。"""

    identity = build_participant_identity(platform=platform, user_id=user_id, nickname=nickname, group_card=group_card)
    return f"<person {' '.join(participant_identity_attributes(identity))}/>"


def format_bot_identity_context(*, platform: str, nickname: str) -> str:
    """列出当前平台已确认的自身账号，支持首次发言和多账号场景。"""

    references = [
        format_participant_reference(platform=platform, user_id=user_id, nickname=nickname)
        for user_id in sorted(get_bot_accounts(platform))
    ]
    if not references:
        return ""
    return "<bot_accounts>\n" + "\n".join(references) + "\n</bot_accounts>"
