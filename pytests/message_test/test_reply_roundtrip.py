"""运行时消息往返保留引用结构，数据库仍只存引用 ID。"""

from datetime import datetime

import pytest

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    TextComponent,
)
from src.plugin_runtime.hook_payloads import deserialize_session_message, serialize_session_message
from src.plugin_runtime.host.message_utils import PluginMessageUtils


def _message() -> SessionMessage:
    message = SessionMessage("current", datetime(2026, 10, 23), "test")
    message.session_id = "real-session"
    message.message_info = MessageInfo(UserInfo("reader", "提问者"))
    message.raw_message = MessageSequence(
        [
            ReplyComponent(
                "original",
                target_message_content="原文 [图片：一只猫]",
                target_message_sender_id="author",
                target_message_sender_nickname="原作者",
                target_message=MessageSequence(
                    [
                        TextComponent("原文"),
                        ImageComponent(binary_hash="", content="[图片：一只猫]", binary_data=b"original-image"),
                    ]
                ),
            ),
            TextComponent("这是什么？"),
        ]
    )
    return message


def test_hook_roundtrip_preserves_resolved_quote_and_image() -> None:
    source = _message()

    restored = deserialize_session_message(serialize_session_message(source))

    quote = restored.raw_message.components[0]
    assert isinstance(quote, ReplyComponent)
    assert quote.target_message_id == "original"
    assert quote.target_message_content == "原文 [图片：一只猫]"
    assert quote.target_message_sender_id == "author"
    assert quote.target_message_sender_nickname == "原作者"
    assert quote.target_message is not None
    assert quote.target_message.components[0].text == "原文"
    image = quote.target_message.components[1]
    assert isinstance(image, ImageComponent)
    assert image.binary_hash == source.raw_message.components[0].target_message.components[1].binary_hash
    assert image.content == "[图片：一只猫]"
    assert image.binary_data == b"original-image"
    assert restored.session_id == "real-session"
    assert restored.raw_message.to_dict()[0] == {"type": "reply", "data": "original"}


def test_runtime_quote_respects_binary_serialization_switch() -> None:
    source = _message()

    payload = PluginMessageUtils._session_message_to_dict(source, include_binary_data=False)
    image_payload = payload["raw_message"][0]["data"]["target_message"][1]
    restored = PluginMessageUtils._build_session_message_from_dict(payload)

    assert "binary_data_base64" not in image_payload
    image = restored.raw_message.components[0].target_message.components[1]
    assert image.binary_hash == image_payload["hash"]
    assert image.content == "[图片：一只猫]"
    assert image.binary_data == b""


def test_runtime_quote_preserves_unresolved_and_empty_states() -> None:
    unresolved = PluginMessageUtils._component_from_dict({"type": "reply", "data": {"target_message_id": "old"}})
    empty = PluginMessageUtils._component_from_dict(
        {"type": "reply", "data": {"target_message_id": "empty", "target_message": []}}
    )

    assert unresolved.target_message is None
    assert empty.target_message is not None
    assert empty.target_message.components == []
    with pytest.raises(ValueError, match="组件列表"):
        PluginMessageUtils._component_from_dict(
            {"type": "reply", "data": {"target_message_id": "bad", "target_message": "plain text"}}
        )
