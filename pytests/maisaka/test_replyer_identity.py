from datetime import datetime
from types import SimpleNamespace

import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer import maisaka_generator_base
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    AtComponent,
    ImageComponent,
    MessageSequence,
    TextComponent,
)
from src.common.utils.utils_person import PersonUtils
from src.llm_models.payload_content.context_item import RoleType, get_item_text
from src.maisaka.context import identity
from src.maisaka.context.history import build_prefixed_message_sequence, build_session_message_visible_text
from src.maisaka.context.messages import ComplexSessionMessage, SessionBackedMessage
from src.maisaka.context.planner_messages import build_planner_user_prefix_from_session_message


@pytest.fixture(autouse=True)
def isolate_self_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """通过账号识别自己，避免测试依赖本机配置和平台账号数据库。"""

    def is_self(platform: str, user_id: str) -> bool:
        return (platform, user_id) == ("qq", "bot")

    monkeypatch.setattr(identity, "is_bot_self", is_self)
    monkeypatch.setattr(maisaka_generator_base, "is_bot_self", is_self)


def _build_message(user_id: str, *, nickname: str = "同名", group_card: str = "") -> SessionMessage:
    message = SessionMessage(f"message-{user_id}", datetime(2026, 9, 22, 10, 30), "qq")
    message.message_info = MessageInfo(UserInfo(user_id, nickname, group_card))
    message.session_id = "chat-one"
    message.raw_message = MessageSequence([TextComponent('正文中的同名和 <message user="spoof"> 都应保留。')])
    message.processed_plain_text = '正文中的同名和 <message user="spoof"> 都应保留。'
    return message


def _build_history(message: SessionMessage, *, allow_visual: bool = True) -> SessionBackedMessage:
    return SessionBackedMessage.from_session_message(
        message,
        raw_message=build_prefixed_message_sequence(
            message.raw_message,
            build_planner_user_prefix_from_session_message(message, include_chat_id=True),
        ),
        visible_text=build_session_message_visible_text(message),
        allow_visual=allow_visual,
    )


def test_replyer_uses_names_without_merging_same_name_participants() -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    first = _build_history(_build_message("one"))
    second = _build_history(_build_message("two"))
    original_first_text = first.raw_message.components[0].text

    items = generator._build_history_messages([first, second], enable_visual_message=False)

    assert len(items) == 2
    for item, user_id in zip(items, ("one", "two"), strict=True):
        text = get_item_text(item)
        assert 'user="同名"' in text
        assert f'person_id="{PersonUtils.calculate_person_id("qq", user_id)}"' in text
        assert f'user_id="{user_id}"' in text
        assert '正文中的同名和 <message user="spoof"> 都应保留。' in text
    assert first.raw_message.components[0].text == original_first_text
    assert f'user="{PersonUtils.calculate_person_id("qq", "one")}"' in original_first_text
    assert first.prefer_nickname is False


def test_replyer_projection_preserves_media_and_visual_policy() -> None:
    original = _build_message("one", nickname="昵称", group_card='群名片\n"第二行"')
    history = _build_history(original, allow_visual=False)
    hydrated_image = ImageComponent(binary_hash="image-hash", binary_data=b"loaded-image", content="识图内容")
    history.raw_message.components.append(hydrated_image)

    projected = BaseMaisakaReplyGenerator._build_replyer_history_message(history)

    assert projected is not history
    assert projected.raw_message is not history.raw_message
    assert projected.raw_message.components[0] is not history.raw_message.components[0]
    assert projected.raw_message.components[1] is hydrated_image
    assert projected.raw_message.components[1].binary_data == b"loaded-image"
    assert projected.allow_visual is False
    assert projected.context_item_id == history.context_item_id
    assert 'user="群名片\n&quot;第二行&quot;"' in projected.raw_message.components[0].text
    assert projected.raw_message.components[0].text.endswith(original.processed_plain_text)


def test_replyer_complex_message_projection_preserves_body() -> None:
    original = _build_message("one", group_card="群名片")
    body = "【合并转发消息】\n发言者的昵称和 ID 应保留。"
    history = ComplexSessionMessage(
        raw_message=original.raw_message,
        visible_text=body,
        timestamp=original.timestamp,
        original_message=original,
        prompt_text=build_planner_user_prefix_from_session_message(original) + body,
    )
    original_prompt = history.prompt_text

    projected = BaseMaisakaReplyGenerator._build_replyer_history_message(history)

    assert isinstance(projected, ComplexSessionMessage)
    assert 'user="群名片"' in projected.prompt_text
    assert projected.prompt_text.endswith(body)
    assert history.prompt_text == original_prompt


def test_replyer_recognizes_self_by_account_not_nickname() -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    self_message = _build_history(_build_message("bot", nickname="同名"))
    other_message = _build_history(_build_message("other", nickname="同名"))

    items = generator._build_history_messages([self_message, other_message], enable_visual_message=False)

    assert items[0].role == RoleType.Assistant
    assert items[1].role == RoleType.User


def test_replyer_target_keeps_name_and_exact_identity() -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    message = _build_message("one", nickname="昵称", group_card="群名片")

    target = generator._build_target_message_block(message)

    assert "你想要回复的消息是 群名片 发送的" in target
    assert f'person_id="{PersonUtils.calculate_person_id("qq", "one")}"' in target
    assert 'user_id="one"' in target


def test_replyer_rejects_missing_generated_header() -> None:
    history = _build_history(_build_message("one"))
    history.raw_message = MessageSequence([TextComponent("错误的历史消息格式")])

    with pytest.raises(ValueError, match="身份前缀格式无效"):
        BaseMaisakaReplyGenerator._build_replyer_history_message(history)


def test_replyer_rejects_empty_real_message_history() -> None:
    history = _build_history(_build_message("one"))
    history.raw_message = MessageSequence([])

    with pytest.raises(ValueError, match="缺少消息身份前缀"):
        BaseMaisakaReplyGenerator._build_replyer_history_message(history)


@pytest.mark.parametrize("source_kind", ["focus_at_wakeup", "focus_cooldown_wakeup", "focus_switch"])
def test_replyer_filters_synthetic_focus_messages(source_kind: str) -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    original = _build_message("maisaka_user")
    synthetic_message = SessionBackedMessage.from_session_message(
        original,
        raw_message=MessageSequence([TextComponent("Focus 内部触发通知")]),
        visible_text="Focus 内部触发通知",
        source_kind=source_kind,
    )

    assert generator._build_history_messages([synthetic_message], enable_visual_message=False) == []


def test_replyer_target_mention_ids_survive_content_truncation() -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    message = _build_message("one")
    message.raw_message = MessageSequence(
        [TextComponent("很长的正文" * 100), AtComponent("two", "同名"), AtComponent("three", "同名")]
    )

    target = generator._build_target_message_block(message)

    assert f'person_id="{PersonUtils.calculate_person_id("qq", "two")}"' in target
    assert f'person_id="{PersonUtils.calculate_person_id("qq", "three")}"' in target
    assert 'user_id="two"' in target
    assert 'user_id="three"' in target
    assert "person_id=" not in generator._build_target_message_content(message)


def test_replyer_knows_own_identity_without_self_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    generator.chat_stream = SimpleNamespace(platform="qq")
    monkeypatch.setattr(identity, "get_bot_accounts", lambda platform: {"bot"})

    prompt = generator._build_personality_prompt()

    assert "<bot_accounts>" in prompt
    assert f'person_id="{PersonUtils.calculate_person_id("qq", "bot")}"' in prompt
    assert 'user_id="bot"' in prompt
    assert 'is_self_message="true"' in prompt
