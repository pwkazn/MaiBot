"""引用原消息保留结构化内容，不使用适配器的纯文本预览。"""

from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest

from src.chat.message_receive.message import MsgIDMapping, SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    EmojiComponent,
    ForwardComponent,
    ForwardNodeComponent,
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    StandardMessageComponents,
    TextComponent,
)
from src.llm_models.payload_content.context_item import get_item_text
from src.maisaka.context.history import build_prefixed_message_sequence
from src.maisaka.context.messages import SessionBackedMessage


def _message(message_id: str, components: List[StandardMessageComponents]) -> SessionMessage:
    message = SessionMessage(message_id, datetime.now(), "test")
    message.session_id = "real-session"
    message.message_info = MessageInfo(UserInfo("alice", "昵称", "群名片"))
    message.raw_message = MessageSequence(components)
    return message


@pytest.mark.asyncio
async def test_reply_uses_structured_source_instead_of_adapter_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _message("original", [ReplyComponent("older"), TextComponent("原文"), ImageComponent(binary_hash="image")])
    target.processed_plain_text = "旧的拼接预览 [image]"
    reference = ReplyComponent("original", target_message_content="[image]")
    message = _message("current", [reference, TextComponent("这是什么？")])
    resolve = AsyncMock(return_value=target)
    monkeypatch.setattr("src.chat.message_receive.reply_resolver.resolve_reply_message", resolve)

    def refresh(components: List[StandardMessageComponents]) -> bool:
        components[1].content = "[图片：一只小猫]"
        return True

    monkeypatch.setattr("src.maisaka.visual.chat_history_refresher.refresh_visual_component_descriptions", refresh)
    await message.resolve_reply_components()

    resolve.assert_awaited_once_with(message, "original")
    assert reference.target_message is not None
    assert [type(component) for component in reference.target_message.components] == [TextComponent, ImageComponent]
    assert reference.target_message_content == "原文 [图片：一只小猫]"
    assert reference.target_message_sender_cardname == "群名片"
    # 展开和识图描述刷新都不能污染原消息，更不能递归展开上一层引用。
    assert target.raw_message.components[2].content == ""
    assert len(target.raw_message.components) == 3
    assert message.raw_message.to_dict()[0] == {"type": "reply", "data": "original"}


@pytest.mark.asyncio
async def test_missing_source_does_not_trust_prepopulated_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = ReplyComponent("missing", target_message_content="[image]")
    message = _message("current", [reference])
    monkeypatch.setattr("src.chat.message_receive.reply_resolver.resolve_reply_message", AsyncMock(return_value=None))

    text = await message.process_reply_component(reference, MsgIDMapping())

    assert text == "[回复了一条消息，但原消息已无法访问]"
    assert reference.target_message is None
    assert reference.target_message_content is None


@pytest.mark.asyncio
async def test_platform_image_is_saved_and_keeps_binary_for_visual_model(monkeypatch: pytest.MonkeyPatch) -> None:
    image = ImageComponent(binary_hash="", binary_data=b"downloaded-image")
    message = _message("current", [])
    save = AsyncMock()
    describe = AsyncMock(return_value="已有描述")
    monkeypatch.setattr("src.chat.image_system.image_manager.image_manager.ensure_image_saved", save)
    monkeypatch.setattr("src.chat.image_system.image_manager.image_manager.get_image_description", describe)

    await message._prepare_reply_media([image])

    save.assert_awaited_once_with(b"downloaded-image")
    assert describe.await_args.kwargs["wait_for_build"] is False
    assert image.binary_data == b"downloaded-image"
    assert image.content == "[图片：已有描述]"


@pytest.mark.asyncio
async def test_quoted_platform_image_respects_size_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config.config import global_config

    monkeypatch.setattr(global_config.visual, "handle_oversized_images", True)
    monkeypatch.setattr(global_config.visual, "max_image_size_mb", 1 / 1024 / 1024)
    monkeypatch.setattr(global_config.visual, "oversized_image_handle_method", "discard")
    target = _message("original", [ImageComponent(binary_hash="", binary_data=b"oversized")])
    reference = ReplyComponent("original")
    message = _message("current", [reference])
    monkeypatch.setattr("src.chat.message_receive.reply_resolver.resolve_reply_message", AsyncMock(return_value=target))
    prepare = AsyncMock()
    monkeypatch.setattr(message, "_prepare_reply_media", prepare)

    await message.resolve_reply_components()

    assert reference.target_message is not None
    assert reference.target_message.components == []
    prepare.assert_awaited_once_with([])
    assert target.raw_message.components[0].binary_data == b"oversized"


@pytest.mark.asyncio
async def test_database_reload_resolves_raw_components_instead_of_processed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _message("original", [ImageComponent(binary_hash="stored-image", content="[图片：已有识图描述]")])
    target.processed_plain_text = "[image]"
    target_record = target.to_db_instance()
    message = _message("current", [ReplyComponent("original"), TextComponent("看看这个")])
    restored = SessionMessage.from_db_instance(message.to_db_instance())
    assert restored.raw_message.components[0].target_message is None
    database = SimpleNamespace(exec=lambda statement: SimpleNamespace(first=lambda: target_record))
    monkeypatch.setattr("src.chat.message_receive.reply_resolver.get_db_session", lambda: nullcontext(database))

    await restored.resolve_reply_components()

    reference = restored.raw_message.components[0]
    assert isinstance(reference.target_message.components[0], ImageComponent)
    assert reference.target_message.components[0].binary_hash == "stored-image"
    assert reference.target_message_content == "[图片：已有识图描述]"


def test_monitor_preview_refreshes_structured_quote_description(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.maisaka.monitor.message_payload import build_monitor_reply_preview

    reference = ReplyComponent(
        "original",
        target_message_content="[image]",
        target_message=MessageSequence([ImageComponent(binary_hash="stored-image")]),
    )
    monkeypatch.setattr(
        "src.maisaka.visual.chat_history_refresher._lookup_cached_image_description", lambda _: "新的描述"
    )

    preview = build_monitor_reply_preview(
        _message("current", [reference]),
        find_source_message_by_id=lambda _: None,
        build_source_content=lambda _: "不应使用的旧预览",
    )

    assert preview["content"] == "[图片：新的描述]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("开心,得意", "[表情包: 开心,得意]"),
        (" [表情包: 开心,得意] ", "[表情包: 开心,得意]"),
        ("", "[表情包]"),
    ],
)
async def test_quoted_emoji_keeps_type_marker_in_preview_and_context(
    monkeypatch: pytest.MonkeyPatch, content: str, expected: str
) -> None:
    """引用自身和引用转发节点中的表情都复用统一标记，不能把情绪标签变成发言。"""
    emoji = EmojiComponent(binary_hash="quoted-emoji", content=content)
    reference = ReplyComponent(
        "original",
        target_message=MessageSequence(
            [
                emoji,
                ForwardNodeComponent(
                    [
                        ForwardComponent(message_id="node", user_nickname="节点作者", content=[emoji.clone()]),
                    ]
                ),
            ]
        ),
    )
    message = _message("current", [reference, TextComponent("看看这个")])
    monkeypatch.setattr("src.maisaka.visual.chat_history_refresher._lookup_cached_emoji_description", lambda _: "")

    await message.resolve_reply_components()
    history = SessionBackedMessage.from_session_message(
        message,
        raw_message=build_prefixed_message_sequence(message.raw_message, '<message quote="original">\n'),
        visible_text="",
    )
    item = history.to_context_item(enable_visual_message=False)

    assert reference.target_message_content.count(expected) == 2
    assert item is not None
    assert get_item_text(item).count(expected) == 2
    assert emoji.content == content
