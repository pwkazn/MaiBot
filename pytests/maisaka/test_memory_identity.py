from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest

from src.chat.message_receive.chat_manager import BotChatSession
from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    AtComponent,
    MessageSequence,
    ReplyComponent,
    TextComponent,
)
from src.llm_models.payload_content.context_item import AssistantMessageItem, ContextItemMeta, ContextTextPart
from src.maisaka.context.identity import build_participant_identity
from src.maisaka.context.messages import ModelOutputContextMessage, SessionBackedMessage
from src.maisaka.memory import person_profile
from src.maisaka.memory.heuristic_injector import HeuristicMemoryContext, HeuristicMemoryInjector
from src.maisaka.memory.mid_term import _collect_participants
from src.person_info.person_info import get_person_id
from src.services.memory_service import MemoryHit, MemorySearchResult, memory_service


def _message(user_id: str, nickname: str, *, group_card: Optional[str] = None) -> SessionMessage:
    message = SessionMessage(message_id=f"message:{user_id}", timestamp=datetime(2026, 9, 22), platform="webui")
    message.message_info = MessageInfo(user_info=UserInfo(user_id, nickname, group_card))
    message.raw_message = MessageSequence([TextComponent("讨论周末安排")])
    message.processed_plain_text = "讨论周末安排"
    return message


def _history_message(user_id: str, nickname: str) -> SessionBackedMessage:
    message = _message(user_id, nickname)
    return SessionBackedMessage.from_session_message(
        message,
        raw_message=message.raw_message,
        visible_text="讨论周末安排",
    )


def test_summary_participants_preserve_different_ids_with_the_same_nickname() -> None:
    participants = _collect_participants([_history_message("u1", "小明"), _history_message("u2", "小明")])

    assert len(participants) == 2
    assert get_person_id("webui", "u1") in participants[0]
    assert get_person_id("webui", "u2") in participants[1]
    assert all('nickname="小明"' in participant for participant in participants)


def test_summary_participants_merge_renamed_accounts_by_id() -> None:
    participants = _collect_participants([_history_message("u1", "旧昵称"), _history_message("u1", "新昵称")])

    assert len(participants) == 1
    assert get_person_id("webui", "u1") in participants[0]
    assert 'nickname="新昵称"' in participants[0]


def test_summary_participants_do_not_identify_model_output_as_the_bot() -> None:
    assistant = ModelOutputContextMessage(
        output_item=AssistantMessageItem(
            meta=ContextItemMeta.create(item_id="internal-assistant"),
            parts=(ContextTextPart("内部工具分析"),),
        )
    )

    assert _collect_participants([assistant]) == []
    assert _collect_participants([_history_message("", "麦麦")]) == []


def test_summary_participants_identify_self_from_the_account() -> None:
    participants = _collect_participants([_history_message("self", "任意昵称"), _history_message("u1", "任意昵称")])

    assert 'is_self_message="true"' in participants[0]
    assert 'is_self_message="true"' not in participants[1]


@pytest.mark.parametrize("source_kind", ["focus_at_wakeup", "focus_cooldown_wakeup", "focus_switch"])
def test_summary_participants_ignore_synthetic_focus_senders(source_kind: str) -> None:
    synthetic_message = _history_message("focus-system", "Focus 系统")
    synthetic_message.source_kind = source_kind
    real_message = _history_message("u1", "小明")

    participants = _collect_participants([synthetic_message, real_message])

    assert len(participants) == 1
    assert get_person_id("webui", "u1") in participants[0]
    assert "focus-system" not in participants[0]


@pytest.mark.parametrize("user_id", ["self", ""])
def test_summary_participants_keep_only_confirmed_identity_for_sent_reply(user_id: str) -> None:
    identity = build_participant_identity(platform="webui", user_id=user_id, nickname="麦麦")
    message = SessionBackedMessage(
        raw_message=MessageSequence([TextComponent("我们周末去徒步吧")]),
        visible_text="我们周末去徒步吧",
        timestamp=datetime(2026, 9, 22),
        source_kind="guided_reply",
        participant_identity=identity,
    )

    participants = _collect_participants([message])

    if user_id:
        assert len(participants) == 1
        assert f'person_id="{identity.person_id}"' in participants[0]
        assert 'is_self_message="true"' in participants[0]
    else:
        assert participants == []


def test_profile_candidates_do_not_resolve_identity_from_nickname_only() -> None:
    assert person_profile._resolve_candidate(platform="webui", person_name="小明", source="test") is None


@pytest.mark.asyncio
async def test_profile_injection_keeps_the_queried_person_id(monkeypatch: pytest.MonkeyPatch) -> None:
    anchor = _message("u1", "当前昵称")
    person_id = get_person_id("webui", "u1")
    query = AsyncMock(return_value={"success": True, "person_name": "档案昵称", "profile_text": "喜欢徒步。"})
    monkeypatch.setattr(person_profile.memory_service, "profile_admin", query)
    monkeypatch.setattr(person_profile.global_config.a_memorix.integration, "enable_person_profile_injection", True)

    messages = await person_profile.build_person_profile_injection_messages(anchor_message=anchor)

    query.assert_awaited_once_with(action="query", person_id=person_id, limit=person_profile.PROFILE_QUERY_LIMIT)
    assert len(messages) == 1
    assert f'person_id="{person_id}"' in messages[0]
    assert 'name="档案昵称"' in messages[0]
    assert "喜欢徒步" in messages[0]


def test_heuristic_message_window_preserves_speaker_and_target_identities() -> None:
    message = _message("u1", "同名", group_card="群名片")
    message.raw_message.components.extend(
        [
            AtComponent("u2", target_user_nickname="同名"),
            ReplyComponent("previous", target_message_sender_id="u3", target_message_sender_nickname="同名"),
        ]
    )

    window = HeuristicMemoryInjector._format_message_window([message])

    for user_id in ("u1", "u2", "u3"):
        assert f'person_id="{get_person_id("webui", user_id)}"' in window
    assert 'group_card="群名片"' in window
    assert "提及:" in window
    assert "引用发送者:" in window


def test_heuristic_reference_preserves_person_ids_from_metadata_and_source() -> None:
    hits = [
        MemoryHit(content="小明喜欢徒步", metadata={"person_id": "p1", "person_ids": ["p1", "p2"]}),
        MemoryHit(content="小明想去露营", source="person_fact:p3"),
    ]

    reference = HeuristicMemoryInjector._format_reference(hits, max_chars=900)

    for person_id in ("p1", "p2", "p3"):
        assert reference.count(f'person_id="{person_id}"') == 1


@pytest.mark.parametrize("invalid_id", [None, 123, {}, ""])
def test_heuristic_reference_does_not_invent_ids_from_invalid_metadata(invalid_id: Any) -> None:
    hit = MemoryHit(content="喜欢徒步", metadata={"person_id": invalid_id, "person_ids": [invalid_id, "p1"]})

    reference = HeuristicMemoryInjector._format_reference([hit], max_chars=900)

    assert reference.count("<person ") == 1
    assert 'person_id="p1"' in reference


@pytest.mark.asyncio
async def test_heuristic_search_retains_identity_resolved_from_paragraph_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injector = HeuristicMemoryInjector()
    context = HeuristicMemoryContext(
        session=BotChatSession(session_id="session-1", platform="webui", user_id="u1"),
        recent_messages=[_message("u1", "小明")],
        active_person_ids={"p1"},
    )
    monkeypatch.setattr(
        memory_service,
        "search",
        AsyncMock(return_value=MemorySearchResult(hits=[MemoryHit(content="小明喜欢徒步", hash_value="paragraph-1")])),
    )
    monkeypatch.setattr(
        memory_service,
        "delete_admin",
        AsyncMock(
            return_value={
                "success": True,
                "items": [{"item_type": "paragraph", "item_hash": "paragraph-1", "source": "person_fact:p1"}],
            }
        ),
    )

    hits = await injector._search_related_memory("讨论周末徒步", context)
    reference = injector._format_reference(hits, max_chars=900)

    assert len(hits) == 1
    assert 'person_id="p1"' in reference
