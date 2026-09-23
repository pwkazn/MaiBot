from datetime import datetime
from io import BytesIO

from PIL import Image

import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.message_component_data_model import EmojiComponent, MessageSequence, TextComponent
from src.llm_models.payload_content.context_item import ContextImagePart, get_item_text
from src.maisaka.context.message_adapter import build_visible_text_from_sequence
from src.maisaka.context.messages import SessionBackedMessage, _render_component_for_prompt
from src.services.send_service import _build_processed_plain_text


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("开心,得意", "[表情包: 开心,得意]"),
        ("  开心,得意  ", "[表情包: 开心,得意]"),
        ("", "[表情包]"),
        ("   ", "[表情包]"),
        ("[表情包]", "[表情包]"),
        (" [表情包: 开心,得意] ", "[表情包: 开心,得意]"),
    ],
)
def test_emoji_labels_stay_marked_across_text_rendering_paths(content: str, expected: str) -> None:
    """发送摘要写回自身历史后，表情描述仍须与正文明确区分。"""
    emoji = EmojiComponent(binary_hash="emoji-hash", content=content)
    sequence = MessageSequence([TextComponent("完成啦"), emoji])
    message = SessionMessage("emoji-reply", datetime(2026, 9, 24), "qq")
    message.raw_message = sequence
    message.processed_plain_text = _build_processed_plain_text(message)
    history = SessionBackedMessage(
        raw_message=sequence,
        visible_text=message.processed_plain_text,
        timestamp=message.timestamp,
        source_kind="guided_reply",
    )
    generator = object.__new__(BaseMaisakaReplyGenerator)

    assert emoji.to_plain_text() == expected
    assert _render_component_for_prompt(emoji) == expected
    assert build_visible_text_from_sequence(sequence) == f"完成啦{expected}"
    assert message.processed_plain_text == f"完成啦 {expected}"
    assert generator._extract_guided_bot_reply(history) == f"完成啦 {expected}"
    assert generator._build_target_message_content(message) == f"完成啦 {expected}"
    assert generator._build_text_from_message_sequence(history) == f"完成啦 {expected}"
    item = history.to_context_item(enable_visual_message=False)
    assert item is not None
    assert expected in get_item_text(item)
    # 渲染只改变文本表示，保留原始描述供序列化和其他业务使用。
    assert emoji.content == content
    assert sequence.components[0].text == "完成啦"


def test_visual_emoji_keeps_image_and_text_only_history_keeps_marker() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(image_buffer, format="PNG")
    emoji = EmojiComponent(binary_hash="", binary_data=image_buffer.getvalue(), content="开心,得意")
    history = SessionBackedMessage(
        raw_message=MessageSequence([emoji]),
        visible_text="",
        timestamp=datetime(2026, 9, 24),
    )

    visual_item = history.to_context_item(enable_visual_message=True)
    assert visual_item is not None
    assert get_item_text(visual_item) == "[消息类型]表情包"
    assert any(isinstance(part, ContextImagePart) for part in visual_item.parts)

    text_item = history.to_context_item(enable_visual_message=False)
    assert text_item is not None
    assert get_item_text(text_item) == "[表情包: 开心,得意]"
    assert not any(isinstance(part, ContextImagePart) for part in text_item.parts)
