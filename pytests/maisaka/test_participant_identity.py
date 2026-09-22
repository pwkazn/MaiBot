"""验证同名、改名、自身识别和不同上下文中的身份投影。"""

from datetime import datetime
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring

import pytest

from src.chat.message_receive.chat_manager import BotChatSession
from src.common.data_models.message_component_data_model import (
    AtComponent,
    ForwardComponent,
    ForwardNodeComponent,
    MessageSequence,
    TextComponent,
)
from src.llm_models.payload_content.context_item import get_item_text
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.chat_loop_service import MaisakaChatLoopService
from src.maisaka.context.identity import build_participant_identity, format_participant_reference
from src.maisaka.context.messages import _render_at_component_text, build_full_complex_message_content_from_sequence
from src.maisaka.context.planner_messages import build_planner_prefix


@pytest.fixture(autouse=True)
def bot_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.maisaka.context.identity.is_bot_self",
        lambda platform, user_id: (platform, user_id) == ("qq", "bot-account"),
    )


def test_nickname_changes_do_not_change_identity_and_namesakes_stay_distinct() -> None:
    first = build_participant_identity(platform="qq", user_id="one", nickname="同名")
    renamed = build_participant_identity(platform="qq", user_id="one", nickname="改名", group_card="群名片")
    namesake = build_participant_identity(platform="qq", user_id="two", nickname="同名")
    other_platform = build_participant_identity(platform="telegram", user_id="one", nickname="同名")

    assert first.person_id == renamed.person_id
    assert len({first.person_id, namesake.person_id, other_platform.person_id}) == 3
    assert renamed.display_name == "群名片"


def test_planner_uses_id_and_replyer_uses_name_with_identical_identity_metadata() -> None:
    arguments = dict(
        timestamp=datetime(2026, 9, 22, 12, 0),
        platform="qq",
        user_id="one",
        user_name='昵称"<&',
        group_card="群名片",
        message_id="m1",
    )
    planner = fromstring(build_planner_prefix(**arguments) + "</message>")
    replyer = fromstring(build_planner_prefix(**arguments, prefer_nickname=True) + "</message>")

    assert planner.attrib["user"] == planner.attrib["person_id"]
    assert replyer.attrib["user"] == "群名片"
    for key in ("person_id", "platform", "user_id", "nickname", "group_card", "msg_id"):
        assert planner.attrib[key] == replyer.attrib[key]
    assert planner.attrib["nickname"] == '昵称"<&'


def test_self_recognition_uses_account_even_if_someone_copies_bot_name() -> None:
    arguments = dict(timestamp=datetime(2026, 9, 22), platform="qq", user_name="麦麦")
    own = fromstring(build_planner_prefix(**arguments, user_id="bot-account") + "</message>")
    namesake = fromstring(build_planner_prefix(**arguments, user_id="other") + "</message>")

    assert own.attrib["is_self_message"] == "true"
    assert "is_self_message" not in namesake.attrib
    assert own.attrib["person_id"] != namesake.attrib["person_id"]


def test_missing_account_is_explicitly_unknown_and_never_inferred_from_name() -> None:
    reference = fromstring(format_participant_reference(platform="qq", user_id="", nickname="麦麦"))
    assert reference.attrib["identity_unknown"] == "true"
    assert "person_id" not in reference.attrib
    assert "is_self_message" not in reference.attrib


def test_mentions_retain_target_id_in_both_views() -> None:
    mention = AtComponent(target_user_id="two", target_user_nickname="同名")
    planner = _render_at_component_text(mention, platform="qq")
    replyer = _render_at_component_text(mention, platform="qq", prefer_nickname=True)
    identity = build_participant_identity(platform="qq", user_id="two")

    assert planner.startswith("@<person ")
    assert replyer.startswith("@同名 ")
    assert identity.person_id in planner and identity.person_id in replyer


def test_forwarded_namesakes_preserve_separate_account_identities() -> None:
    forward = ForwardNodeComponent(
        [
            ForwardComponent(user_nickname="同名", user_id="one", message_id="m1", content=[TextComponent("第一人")]),
            ForwardComponent(user_nickname="同名", user_id="two", message_id="m2", content=[TextComponent("第二人")]),
        ]
    )
    content = build_full_complex_message_content_from_sequence(MessageSequence([forward]), platform="qq")
    for user_id in ("one", "two"):
        assert build_participant_identity(platform="qq", user_id=user_id).person_id in content
    assert content.count('nickname="同名"') == 2


def test_custom_subagent_prompt_includes_identity_protocol_and_own_account(monkeypatch: pytest.MonkeyPatch) -> None:
    session = BotChatSession(session_id="identity-test", platform="qq", group_id="group", account_id="bot-account")
    monkeypatch.setattr("src.maisaka.context.identity.get_bot_accounts", lambda platform: {"bot-account"})
    monkeypatch.setattr(
        "src.maisaka.chat_loop_service.chat_manager.get_session_by_session_id", lambda session_id: session
    )
    service = MaisakaChatLoopService(chat_system_prompt="自定义子任务", session_id=session.session_id)

    system_text = get_item_text(service._build_request_messages([], enable_visual_message=False)[0])

    assert "自定义子任务" in system_text
    assert "person_id" in system_text
    assert "<bot_accounts>" in system_text
    assert build_participant_identity(platform="qq", user_id="bot-account").person_id in system_text
    assert 'is_self_message="true"' in system_text


def test_synthetic_sent_messages_keep_known_account_without_original_message() -> None:
    session = BotChatSession(session_id="identity-test", platform="qq", group_id="group", account_id="bot-account")
    runtime = SimpleNamespace(
        chat_stream=session,
        session_id=session.session_id,
        _chat_history=[],
        _is_focus_mode_active_for_current_chat=lambda: False,
    )
    context = BuiltinToolRuntimeContext(engine=None, runtime=runtime)
    context.append_guided_reply_to_chat_history("自己的回复")
    context.append_sent_emoji_to_chat_history(emoji_base64="", success_message="微笑")

    assert len(runtime._chat_history) == 2
    for message in runtime._chat_history:
        assert message.original_message is None
        assert message.participant_identity is not None
        assert (
            message.participant_identity.person_id
            == build_participant_identity(platform="qq", user_id="bot-account").person_id
        )
        assert message.participant_identity.is_self
        assert 'is_self_message="true"' in get_item_text(message.to_context_item())
