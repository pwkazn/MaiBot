"""按当前模型需要加载消息组件中的图片与表情。"""

from typing import Sequence

import asyncio

from src.common.data_models.message_component_data_model import (
    EmojiComponent,
    ImageComponent,
    StandardMessageComponents,
)


async def hydrate_visual_components(components: Sequence[StandardMessageComponents]) -> None:
    """复用已有二进制，只加载仍缺少图片数据的组件。"""

    load_tasks = []
    for component in components:
        if isinstance(component, ImageComponent) and not component.binary_data:
            load_tasks.append(component.load_image_binary())
        elif isinstance(component, EmojiComponent) and not component.binary_data:
            load_tasks.append(component.load_emoji_binary())

    await asyncio.gather(*load_tasks)
