"""引用原消息的结构化内容应按当前模型模式呈现。"""

from base64 import b64decode
from datetime import datetime
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock

import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator, MaisakaReplyContext
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import (
    ForwardComponent,
    ForwardNodeComponent,
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    TextComponent,
)
from src.llm_models.payload_content.context_item import ContextImagePart, ContextItem, UserMessageItem, get_item_text
from src.maisaka.context.history import build_prefixed_message_sequence
from src.maisaka.context.messages import ComplexSessionMessage, SessionBackedMessage
from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.visual import chat_history_refresher
from src.maisaka.visual.message_limiter import limit_latest_images_in_messages

_PNG_BYTES = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY1cAAAAASUVORK5CYII=")


@pytest.fixture
def visual_loads(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    loaded_hashes: List[str] = []

    async def load_image(component: ImageComponent) -> None:
        loaded_hashes.append(component.binary_hash)
        component.binary_data = _PNG_BYTES

    monkeypatch.setattr(ImageComponent, "load_image_binary", load_image)
    monkeypatch.setattr(chat_history_refresher, "_lookup_cached_image_description", lambda _hash: "")
    return loaded_hashes


def _message(message_id: str = "reply", *, description: str = "[图片：一只猫]") -> SessionMessage:
    message = SessionMessage(message_id, datetime(2026, 10, 23, 10), "test")
    message.session_id = "quoted-images"
    message.message_info = MessageInfo(UserInfo("reader", "提问者"))
    message.raw_message = MessageSequence(
        [
            ReplyComponent(
                "original",
                target_message_sender_id="author",
                target_message_sender_nickname="原作者",
                target_message=MessageSequence(
                    [
                        TextComponent("原消息文字"),
                        ImageComponent(binary_hash="quoted-cat", content=description),
                        ReplyComponent("nested", target_message_content="不要展开的嵌套引用"),
                    ]
                ),
            ),
            TextComponent("这是什么？"),
        ]
    )
    return message


def _engine() -> MaisakaReasoningEngine:
    runtime = SimpleNamespace(
        log_prefix="[引用测试]",
        _is_focus_mode_active_for_current_chat=lambda: False,
    )
    return MaisakaReasoningEngine(runtime)


def test_unresolved_quote_does_not_reuse_flattened_adapter_preview() -> None:
    sequence = MessageSequence(
        [
            ReplyComponent("original", target_message_content="[image]"),
            TextComponent("这是什么？"),
        ]
    )

    expanded = build_prefixed_message_sequence(sequence, '<message quote="original">\n')

    assert len(expanded.components) == 1
    assert expanded.components[0].text == '<message quote="original">\n这是什么？'


@pytest.mark.asyncio
async def test_missing_quote_image_exposes_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: True)

    async def fail_load(component: ImageComponent) -> None:
        raise FileNotFoundError(f"图片文件不存在：{component.binary_hash}")

    monkeypatch.setattr(ImageComponent, "load_image_binary", fail_load)

    with pytest.raises(FileNotFoundError, match="quoted-cat"):
        await _engine()._build_history_message(_message())


@pytest.mark.asyncio
@pytest.mark.parametrize("planner_visual,allow_visual", [(True, True), (False, True), (True, False)])
async def test_quote_keeps_image_components_and_respects_visual_policy(
    monkeypatch: pytest.MonkeyPatch,
    visual_loads: List[str],
    planner_visual: bool,
    allow_visual: bool,
) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: planner_visual)
    message = _message()

    history = await _engine()._build_history_message(message, allow_visual=allow_visual)

    assert isinstance(history, SessionBackedMessage)
    assert visual_loads == (["quoted-cat"] if planner_visual and allow_visual else [])
    item = history.to_context_item(enable_visual_message=True)
    assert isinstance(item, UserMessageItem)
    text = get_item_text(item)
    assert 'quote="original"' in text
    assert '<quoted_message msg_id="original" user_id="author" name="原作者">' in text
    assert "原消息文字" in text
    assert "不要展开的嵌套引用" not in text
    assert text.index("</quoted_message>") < text.index("这是什么？")
    images = [part for part in item.parts if isinstance(part, ContextImagePart)]
    assert len(images) == int(planner_visual and allow_visual)
    assert "[图片：一只猫]" in get_item_text(history.to_context_item(enable_visual_message=False))
    limited = limit_latest_images_in_messages([item], max_image_num=0)
    assert all(not isinstance(part, ContextImagePart) for part in limited[0].parts)
    assert not message.raw_message.components[0].target_message.components[1].binary_data


@pytest.mark.asyncio
async def test_quote_image_stays_visual_beside_forward_summary(
    monkeypatch: pytest.MonkeyPatch, visual_loads: List[str]
) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: True)
    message = _message()
    message.raw_message.components.append(
        ForwardNodeComponent(
            [
                ForwardComponent(
                    message_id="forward-node",
                    user_nickname="转发作者",
                    content=[
                        TextComponent("转发正文"),
                        ImageComponent(binary_hash="forward-image", content="转发图描述"),
                    ],
                )
            ]
        )
    )

    history = await _engine()._build_history_message(message)

    assert isinstance(history, ComplexSessionMessage)
    assert visual_loads == ["quoted-cat"]
    projected = BaseMaisakaReplyGenerator._build_replyer_history_message(history)
    item = projected.to_context_item(enable_visual_message=True)
    assert isinstance(item, UserMessageItem)
    assert len([part for part in item.parts if isinstance(part, ContextImagePart)]) == 1
    assert "转发正文" in get_item_text(item)
    assert "转发图描述" in get_item_text(item)
    assert 'user="提问者"' in get_item_text(item)


@pytest.mark.asyncio
async def test_cached_quote_description_refreshes_history_without_loading_binary(
    monkeypatch: pytest.MonkeyPatch, visual_loads: List[str]
) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: False)
    engine = _engine()
    message = _message(description="")
    history = await engine._build_history_message(message, allow_visual=False)
    assert isinstance(history, SessionBackedMessage)
    old_id = history.context_item_id
    monkeypatch.setattr(chat_history_refresher, "_lookup_cached_image_description", lambda _hash: "猫在窗边")
    history_items = [history]

    async def rebuild(original: SessionMessage, source_kind: str, allow_visual: bool):
        return await engine._build_history_message(original, source_kind=source_kind, allow_visual=allow_visual)

    refreshed = await chat_history_refresher.refresh_chat_history_visual_placeholders(
        chat_history=history_items,
        build_history_message=rebuild,
        build_visible_text=lambda original, source: engine._build_legacy_visible_text(
            original,
            original.raw_message,
            source_kind=source,
        ),
    )

    assert refreshed == 1
    assert visual_loads == []
    assert history_items[0].allow_visual is False
    assert history_items[0].context_item_id == old_id
    assert "[图片：猫在窗边]" in get_item_text(history_items[0].to_context_item(enable_visual_message=True))


@pytest.mark.asyncio
async def test_quote_of_forward_message_includes_node_images(
    monkeypatch: pytest.MonkeyPatch, visual_loads: List[str]
) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: True)
    message = _message()
    reply = message.raw_message.components[0]
    reply.target_message = MessageSequence(
        [
            ForwardNodeComponent(
                [
                    ForwardComponent(
                        message_id="quoted-forward-node",
                        user_id="node-author",
                        user_nickname="节点作者",
                        content=[ImageComponent(binary_hash="forward-image", content="[图片：转发里的猫]")],
                    ),
                ]
            )
        ]
    )

    history = await _engine()._build_history_message(message)

    assert isinstance(history, SessionBackedMessage)
    assert visual_loads == ["forward-image"]
    item = history.to_context_item(enable_visual_message=True)
    assert isinstance(item, UserMessageItem)
    assert len([part for part in item.parts if isinstance(part, ContextImagePart)]) == 1
    assert '<forward_message user_id="node-author" name="节点作者">' in get_item_text(item)
    assert "[图片：转发里的猫]" in get_item_text(history.to_context_item(enable_visual_message=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("replyer_visual", [True, False])
async def test_replyer_hydrates_quotes_independently_of_text_planner(
    monkeypatch: pytest.MonkeyPatch, visual_loads: List[str], replyer_visual: bool
) -> None:
    monkeypatch.setattr("src.maisaka.reasoning_engine.resolve_enable_visual_planner", lambda: False)
    history = await _engine()._build_history_message(_message())
    recalled = await _engine()._build_history_message(_message("recalled"), allow_visual=False)
    assert visual_loads == []
    captured: List[ContextItem] = []

    class Model:
        task_name = "replyer"

        async def generate_response_with_context(self, *, context_factory, options):
            captured.extend(
                await context_factory(
                    SimpleNamespace(api_provider=SimpleNamespace(client_type="test")),
                    SimpleNamespace(name="test-model", visual=replyer_visual),
                )
            )
            raise RuntimeError("测试在请求构建后停止")

    generator = object.__new__(BaseMaisakaReplyGenerator)
    generator.express_model = Model()
    generator.request_type = "maisaka.replyer"
    generator._enable_visual_message = None
    generator._replyer_mode = "auto"
    monkeypatch.setattr(generator, "_build_reply_context", AsyncMock(return_value=MaisakaReplyContext()))
    monkeypatch.setattr(
        generator,
        "_get_runtime_manager",
        lambda: SimpleNamespace(
            invoke_hook=AsyncMock(return_value=SimpleNamespace(kwargs={})),
        ),
    )
    monkeypatch.setattr(
        generator,
        "_build_request_messages",
        lambda *, chat_history, enable_visual_message=False, **kw: generator._build_history_messages(
            chat_history, enable_visual_message
        ),
    )
    monkeypatch.setattr(
        generator,
        "_invoke_before_model_request_hook",
        AsyncMock(
            side_effect=lambda **kw: kw["request_messages"],
        ),
    )

    success, result = await generator.generate_reply_with_context(
        stream_id="quoted-images",
        chat_history=[history, recalled],
    )

    assert success is False
    assert result.error_message == "测试在请求构建后停止"
    assert len(captured) == 2
    assert visual_loads == (["quoted-cat"] if replyer_visual else [])
    assert len([part for part in captured[0].parts if isinstance(part, ContextImagePart)]) == int(replyer_visual)
    assert all(not isinstance(part, ContextImagePart) for part in captured[1].parts)
