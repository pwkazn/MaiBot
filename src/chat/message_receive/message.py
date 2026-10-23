from asyncio import Task
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

from rich.traceback import install

import asyncio

from src.common.logger import get_logger
from src.common.data_models.mai_message_data_model import MaiMessage, UserInfo
from src.common.data_models.message_component_data_model import (
    AtComponent,
    DictComponent,
    EmojiComponent,
    FileComponent,
    ForwardNodeComponent,
    ImageComponent,
    MessageSequence,
    ReplyComponent,
    StandardMessageComponents,
    TextComponent,
    VoiceComponent,
)


install(extra_lines=3)

logger = get_logger("chat_message")


class MsgIDMapping:
    """回复消息内容缓存。"""

    def __init__(self) -> None:
        """初始化消息 ID 到内容的映射缓存。"""
        self.mapping: Dict[str, Tuple[str | Task[str], UserInfo]] = {}
        self.messages: Dict[str, MessageSequence] = {}
        """转发节点的结构化内容，供引用保留图片等原始组件。"""


class SessionMessage(MaiMessage):
    # 仅记录适配器成功回执明确提供的平台最终 ID；没有回执时保持 None，
    # 不得使用发送前生成的内部 message_id 代替。
    platform_message_id: Optional[str] = None

    #便于调试的打印函数
    def __str__(self) -> str:
        """返回适合日志输出的消息摘要。"""
        return self.to_debug_string()

    def __repr__(self) -> str:
        """返回适合调试场景的消息摘要。"""
        return self.to_debug_string()

    def to_debug_string(self) -> str:
        """构建包含引用信息的调试字符串。

        Returns:
            str: 适合记录日志的消息摘要。
        """
        user_info = self.message_info.user_info
        group_info = self.message_info.group_info
        chat_type = "group" if group_info else "private"
        group_id = group_info.group_id if group_info else None
        group_name = group_info.group_name if group_info else None
        component_summaries = [self._summarize_component(component) for component in self.raw_message.components]
        raw_components = ", ".join(component_summaries) if component_summaries else "empty"

        return (
            "SessionMessage("
            f"message_id={self.message_id!r}, "
            f"platform={self.platform!r}, "
            f"chat_type={chat_type!r}, "
            f"group_id={group_id!r}, "
            f"group_name={group_name!r}, "
            f"user_id={user_info.user_id!r}, "
            f"user_nickname={user_info.user_nickname!r}, "
            f"user_cardname={user_info.user_cardname!r}, "
            f"reply_to={self.reply_to!r}, "
            f"processed_plain_text={self._truncate_text(self.processed_plain_text)}, "
            f"raw_components=[{raw_components}]"
            ")"
        )

    @staticmethod
    def _truncate_text(text: str | None, max_length: int = 120) -> str:
        """截断较长文本，避免日志过长。

        Args:
            text: 原始文本。
            max_length: 最大保留长度。

        Returns:
            str: 截断后的文本表示。
        """
        if text is None:
            return "None"
        normalized_text = text.replace("\r", "\\r").replace("\n", "\\n")
        if len(normalized_text) <= max_length:
            return repr(normalized_text)
        return repr(f"{normalized_text[:max_length]}...")

    def _summarize_component(self, component: StandardMessageComponents) -> str:
        """生成单个消息组件的调试摘要。

        Args:
            component: 消息组件对象。

        Returns:
            str: 组件摘要文本。
        """
        if isinstance(component, TextComponent):
            return f"Text(text={self._truncate_text(component.text, 80)})"
        if isinstance(component, ImageComponent):
            return f"Image(content={self._truncate_text(component.content or None, 60)})"
        if isinstance(component, EmojiComponent):
            return f"Emoji(content={self._truncate_text(component.content or None, 60)})"
        if isinstance(component, AtComponent):
            target_name = component.target_user_cardname or component.target_user_nickname or component.target_user_id
            return f"At(target={target_name!r})"
        if isinstance(component, VoiceComponent):
            return f"Voice(content={self._truncate_text(component.content or None, 60)})"
        if isinstance(component, FileComponent):
            return f"File(name={component.name!r}, size={component.size!r})"
        if isinstance(component, ReplyComponent):
            sender_name = (
                component.target_message_sender_cardname
                or component.target_message_sender_nickname
                or component.target_message_sender_id
            )
            return (
                "Reply("
                f"target_message_id={component.target_message_id!r}, "
                f"target_sender={sender_name!r}, "
                f"target_content={self._truncate_text(component.target_message_content, 80)}"
                ")"
            )
        if isinstance(component, ForwardNodeComponent):
            return f"ForwardNode(count={len(component.forward_components)})"
        return f"{component.__class__.__name__}"
    #便于调试的打印函数end

    async def process(
        self,
        *,
        enable_heavy_media_analysis: bool = True,
        enable_voice_transcription: bool = True,
    ) -> None:
        """处理消息内容并转化为纯文本。

        Args:
            enable_heavy_media_analysis: 是否同步执行图片与表情包描述生成。
            enable_voice_transcription: 是否同步执行语音转写。
        """
        self._remove_voice_placeholder_text_components()
        id_content_map = MsgIDMapping()
        tasks = [
            self.process_single_component(
                component,
                id_content_map,
                enable_heavy_media_analysis=enable_heavy_media_analysis,
                enable_voice_transcription=enable_voice_transcription,
            )
            for component in self.raw_message.components
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_texts: List[str] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"处理消息组件时发生错误: {result}")
            else:
                processed_texts.append(result)
        self.processed_plain_text = " ".join(processed_texts)

    def _remove_voice_placeholder_text_components(self) -> None:
        """移除适配器随语音片段附带的文本占位，避免污染处理后的文本。"""

        has_voice_component = any(isinstance(component, VoiceComponent) for component in self.raw_message.components)
        if not has_voice_component:
            return

        self.raw_message.components = [
            component
            for component in self.raw_message.components
            if not (isinstance(component, TextComponent) and component.text.strip().lower() == "[voice]")
        ]

    async def process_single_component(
        self,
        component: StandardMessageComponents,
        id_content_map: MsgIDMapping,
        recursion_depth: int = 0,
        *,
        enable_heavy_media_analysis: bool = True,
        enable_voice_transcription: bool = True,
    ) -> str:
        """按类型处理单个消息组件。

        Args:
            component: 待处理的消息组件。
            id_content_map: 回复消息解析缓存。
            recursion_depth: 当前递归深度。
            enable_heavy_media_analysis: 是否同步执行图片与表情包描述生成。
            enable_voice_transcription: 是否同步执行语音转写。

        Returns:
            str: 组件对应的文本表示。
        """
        if isinstance(component, TextComponent):
            return component.text
        elif isinstance(component, ImageComponent):
            return await self.process_image_component(
                component,
                enable_heavy_media_analysis=enable_heavy_media_analysis,
            )
        elif isinstance(component, EmojiComponent):
            return await self.process_emoji_component(
                component,
                enable_heavy_media_analysis=enable_heavy_media_analysis,
            )
        elif isinstance(component, AtComponent):
            return await self.process_at_component(component)
        elif isinstance(component, VoiceComponent):
            return await self.process_voice_component(
                component,
                enable_voice_transcription=enable_voice_transcription,
            )
        elif isinstance(component, FileComponent):
            return component.to_plain_text()
        elif isinstance(component, ReplyComponent):
            return await self.process_reply_component(component, id_content_map)
        elif isinstance(component, ForwardNodeComponent):
            return await self.process_forward_component(
                component,
                id_content_map,
                recursion_depth=recursion_depth + 1,
                enable_heavy_media_analysis=enable_heavy_media_analysis,
                enable_voice_transcription=enable_voice_transcription,
            )
        elif isinstance(component, DictComponent):
            return self.process_dict_component(component)
        else:
            raise NotImplementedError(f"暂时不支持的消息组件类型: {type(component)}")

    def process_dict_component(self, component: DictComponent) -> str:
        """处理字典组件，保留非标准消息的可读摘要。"""

        component_data = component.data
        raw_type = str(component_data.get("type") or "dict").strip().lower()
        raw_payload = component_data.get("data", component_data)

        if raw_type == "file" and isinstance(raw_payload, dict):
            return self._build_file_component_text(raw_payload)
        if raw_type == "json" and isinstance(raw_payload, dict):
            return str(raw_payload.get("text") or raw_payload.get("prompt") or "[json]")
        if raw_type:
            return f"[{raw_type}]"
        return "[复杂消息]"

    @staticmethod
    def _build_file_component_text(payload: Dict[str, object]) -> str:
        """构造文件字典组件的可读文本。"""

        file_name = str(
            payload.get("name")
            or payload.get("file")
            or payload.get("file_name")
            or payload.get("filename")
            or ""
        ).strip()
        file_size = str(payload.get("size") or payload.get("file_size") or "").strip()
        file_url = str(payload.get("url") or payload.get("file_url") or "").strip()

        text_parts: List[str] = []
        if file_name:
            text_parts.append(file_name)
        if file_size:
            text_parts.append(f"大小: {file_size}")

        file_text = "[文件]"
        if text_parts:
            file_text = f"[文件] {'，'.join(text_parts)}"
        if file_url:
            file_text = f"{file_text}，链接: {file_url}"
        return file_text

    async def process_image_component(
        self,
        component: ImageComponent,
        *,
        enable_heavy_media_analysis: bool = True,
    ) -> str:
        """处理图片组件。

        Args:
            component: 图片组件。
            enable_heavy_media_analysis: 是否同步执行图片描述生成。

        Returns:
            str: 图片组件对应的文本表示。
        """
        normalized_content = component.content.strip()
        if normalized_content:  # 先检查是否处理过
            component.content = normalized_content
            return normalized_content
        from src.chat.image_system.image_manager import image_manager

        # 获取描述
        try:
            desc = await image_manager.get_image_description(
                image_bytes=component.binary_data,
                wait_for_build=enable_heavy_media_analysis,
            )
        except Exception:
            desc = None  # 失败置空

        # desc 为空时保持 content 为空，表示图片仍处于待识别状态；
        # 展示占位由 Maisaka 渲染层处理，避免把占位符当作已识别内容。
        content = f"[图片：{desc}]" if desc else ""
        component.content = content
        component.binary_data = b""  # 处理完就丢掉二进制数据，节省内存
        return content

    async def process_emoji_component(
        self,
        component: EmojiComponent,
        *,
        enable_heavy_media_analysis: bool = True,
    ) -> str:
        """处理表情包组件。

        Args:
            component: 表情包组件。
            enable_heavy_media_analysis: 是否同步执行表情包描述生成。

        Returns:
            str: 表情包组件对应的文本表示。
        """
        normalized_content = component.content.strip()
        if normalized_content:  # 先检查是否处理过
            component.content = normalized_content
            return normalized_content
        from src.emoji_system.emoji_manager import emoji_manager

        # 获取表情包描述
        try:
            tuple_content = await emoji_manager.get_emoji_description(
                emoji_bytes=component.binary_data,
                session_id=self.session_id,
                wait_for_build=enable_heavy_media_analysis,
            )
        except Exception:
            tuple_content = None  # 失败置空

        if tuple_content:
            desc, _ = tuple_content
            content = f"[表情包: {desc}]"
        else:
            content = "[表情包]"
        component.content = content
        component.binary_data = b""  # 处理完就丢掉二进制数据，节省内存
        return content

    async def process_at_component(self, component: AtComponent) -> str:
        # 如果已经有昵称或备注了，直接使用
        if component.target_user_cardname:
            return f"@{component.target_user_cardname}"
        elif component.target_user_nickname:
            return f"@{component.target_user_nickname}"
        from src.common.utils.system_utils import is_bot_self
        from src.config.config import global_config

        if is_bot_self(self.platform, component.target_user_id):
            bot_nickname = global_config.bot.nickname.strip()
            if bot_nickname:
                component.target_user_nickname = bot_nickname
                component.target_user_cardname = bot_nickname
                return f"@{bot_nickname}"

        from src.common.utils.utils_person import PersonUtils

        # 查询用户信息
        if person_info := PersonUtils.get_person_info_by_user_id_and_platform(component.target_user_id, self.platform):
            component.target_user_nickname = component.target_user_nickname or person_info.user_nickname
            if self.message_info.group_info and person_info.group_cardname_list:
                for group_card in person_info.group_cardname_list:
                    if group_card.group_id == self.message_info.group_info.group_id:
                        component.target_user_cardname = group_card.group_cardname
                        break
        if component.target_user_cardname:  # 优先使用群备注
            return f"@{component.target_user_cardname}"
        elif component.target_user_nickname:  # 其次使用昵称
            return f"@{component.target_user_nickname}"
        else:  # 最后使用用户ID
            return f"@{component.target_user_id}"

    async def process_voice_component(
        self,
        component: VoiceComponent,
        *,
        enable_voice_transcription: bool = True,
    ) -> str:
        """处理语音组件。

        Args:
            component: 语音组件。
            enable_voice_transcription: 是否同步执行语音转写。

        Returns:
            str: 语音组件对应的文本表示。
        """
        normalized_content = component.content.strip()
        if normalized_content:  # 先检查是否处理过
            component.content = normalized_content
            return normalized_content
        if not enable_voice_transcription:
            component.content = "[语音消息]"
            return component.content
        from src.common.utils.utils_voice import get_voice_text

        text = await get_voice_text(component.binary_data)
        content = "[语音消息，转录失败]" if text is None else f"[语音: {text}]"
        component.content = content
        return content

    async def process_reply_component(
        self,
        component: ReplyComponent,
        id_content_map: MsgIDMapping,
    ) -> str:
        """按 ID 解析结构化原消息，纯文本预览不能替代原消息。"""

        if result_item := id_content_map.mapping.get(component.target_message_id):  # ID映射缓存优先
            content, sender_info = result_item
            if isinstance(content, Task):  # 如果是Task，说明是转发组件传入的占位结果，需要等待其完成
                content = await content  # 获取最终结果
                id_content_map.mapping[component.target_message_id] = (content, sender_info)  # 更新为实际内容
            component.target_message = MessageSequence(
                self._copy_reply_components(id_content_map.messages[component.target_message_id].components)
            )
            component.target_message_sender_cardname = sender_info.user_cardname
            component.target_message_sender_nickname = sender_info.user_nickname
            component.target_message_sender_id = sender_info.user_id
        elif component.target_message is None:
            from .image_receive_compressor import process_received_images_in_message
            from .reply_resolver import resolve_reply_message

            target_message = await resolve_reply_message(self, component.target_message_id)
            if target_message is None:
                component.target_message_content = None
                return "[回复了一条消息，但原消息已无法访问]"
            component.target_message = MessageSequence(self._copy_reply_components(target_message.raw_message.components))
            sender_info = target_message.message_info.user_info
            component.target_message_sender_cardname = sender_info.user_cardname
            component.target_message_sender_nickname = sender_info.user_nickname
            component.target_message_sender_id = sender_info.user_id
            # 平台查询回来的图片也遵守入站压缩/丢弃策略，不绕过尺寸限制。
            process_received_images_in_message(component.target_message.components)
            await self._prepare_reply_media(component.target_message.components)

        from src.maisaka.visual.chat_history_refresher import refresh_visual_component_descriptions

        refresh_visual_component_descriptions(component.target_message.components)
        content = self._render_reply_content(component.target_message.components)
        component.target_message_content = content
        sender_name = (
            component.target_message_sender_cardname
            or component.target_message_sender_nickname
            or component.target_message_sender_id
            or "未知用户"
        )
        return f"[回复了{sender_name}的消息: {content}]"

    async def resolve_reply_components(self) -> None:
        """补齐从数据库恢复后只含 ID 的引用，不重跑当前消息的媒体处理。"""

        id_content_map = MsgIDMapping()
        for component in self.raw_message.components:
            if isinstance(component, ReplyComponent):
                await self.process_reply_component(component, id_content_map)

    @staticmethod
    def _copy_reply_components(components: Sequence[StandardMessageComponents]) -> List[StandardMessageComponents]:
        """引用只展开原消息自身的一层，避免嵌套引用和循环引用。"""

        copied = [deepcopy(component) for component in components if not isinstance(component, ReplyComponent)]
        for component in copied:
            if isinstance(component, ForwardNodeComponent):
                for node in component.forward_components:
                    node.content = SessionMessage._copy_reply_components(node.content)
        return copied

    async def _prepare_reply_media(self, components: Sequence[StandardMessageComponents]) -> None:
        """保存平台取回的媒体；本地仅有 hash 的组件不重复识别或转写。"""

        for component in components:
            if isinstance(component, ImageComponent) and component.binary_data:
                from src.chat.image_system.image_manager import image_manager

                await image_manager.ensure_image_saved(component.binary_data)
                if not component.content.strip():
                    description = await image_manager.get_image_description(
                        image_hash=component.binary_hash,
                        image_bytes=component.binary_data,
                        wait_for_build=False,
                    )
                    if description:
                        component.content = f"[图片：{description}]"
            elif isinstance(component, EmojiComponent) and component.binary_data:
                binary_data = component.binary_data
                await self.process_emoji_component(component, enable_heavy_media_analysis=False)
                component.binary_data = binary_data
            elif isinstance(component, ForwardNodeComponent):
                for node in component.forward_components:
                    await self._prepare_reply_media(node.content)

    def _render_reply_content(self, components: Sequence[StandardMessageComponents], depth: int = 0) -> str:
        """从结构化原消息生成预览，图片优先使用已有描述。"""

        parts: List[str] = []
        for component in components:
            if isinstance(component, TextComponent):
                parts.append(component.text)
            elif isinstance(component, ImageComponent):
                parts.append(component.content.strip() or "[图片，识别中.....]")
            elif isinstance(component, EmojiComponent):
                parts.append(component.to_plain_text())
            elif isinstance(component, VoiceComponent):
                parts.append(component.content.strip() or "[语音消息]")
            elif isinstance(component, AtComponent):
                parts.append(f"@{component.target_user_cardname or component.target_user_nickname or component.target_user_id}")
            elif isinstance(component, FileComponent):
                parts.append(component.to_plain_text())
            elif isinstance(component, DictComponent):
                parts.append(self.process_dict_component(component))
            elif isinstance(component, ForwardNodeComponent):
                nodes: List[str] = []
                for node in component.forward_components:
                    sender = node.user_cardname or node.user_nickname or node.user_id or "未知用户"
                    content = self._render_reply_content(node.content, depth + 1)
                    nodes.append(f"{'-' * (depth + 1) * 2} 【{sender}】: {content}")
                parts.append("【合并转发消息: \n" + "\n".join(nodes) + "\n】")
        return " ".join(parts)

    async def process_forward_component(
        self,
        component: ForwardNodeComponent,
        id_content_map: MsgIDMapping,
        recursion_depth: int = 0,
        *,
        enable_heavy_media_analysis: bool = True,
        enable_voice_transcription: bool = True,
    ) -> str:
        """处理合并转发组件。

        Args:
            component: 合并转发组件。
            id_content_map: 回复消息解析缓存。
            recursion_depth: 当前递归深度。
            enable_heavy_media_analysis: 是否同步执行图片与表情包描述生成。
            enable_voice_transcription: 是否同步执行语音转写。

        Returns:
            str: 合并转发组件对应的文本表示。
        """
        task_list: List[Task] = []
        node_user_info_list: List[UserInfo] = []
        for node in component.forward_components:
            task = asyncio.create_task(
                self._process_multiple_components(
                    node.content,
                    id_content_map,
                    recursion_depth + 1,
                    enable_heavy_media_analysis=enable_heavy_media_analysis,
                    enable_voice_transcription=enable_voice_transcription,
                )
            )
            node_user_info = UserInfo(node.user_id or "未知用户", node.user_nickname, node.user_cardname)
            # 传入ID缓存映射，方便Reply组件获取并等待处理结果
            id_content_map.mapping[node.message_id] = (task, node_user_info)
            id_content_map.messages[node.message_id] = MessageSequence(list(node.content))

            task_list.append(task)
            node_user_info_list.append(node_user_info)

        results = await asyncio.gather(*task_list, return_exceptions=True)  # 并行处理节点内容
        forward_texts = []
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"处理转发消息组件时发生错误: {result}")
            else:
                usr_info = node_user_info_list[idx]
                msg_sender_name = usr_info.user_cardname or usr_info.user_nickname or usr_info.user_id or "未知用户"
                forward_texts.append(f"{'-' * recursion_depth * 2} 【{msg_sender_name}】: {result}")
        return "【合并转发消息: \n" + "\n".join(forward_texts) + "\n】"

    async def _process_multiple_components(
        self,
        components: Sequence[StandardMessageComponents],
        id_content_map: MsgIDMapping,
        recursion_depth: int = 0,
        *,
        enable_heavy_media_analysis: bool = True,
        enable_voice_transcription: bool = True,
    ) -> str:
        """并行处理多个消息组件。

        Args:
            components: 待处理的组件序列。
            id_content_map: 回复消息解析缓存。
            recursion_depth: 当前递归深度。
            enable_heavy_media_analysis: 是否同步执行图片与表情包描述生成。
            enable_voice_transcription: 是否同步执行语音转写。

        Returns:
            str: 多个组件拼接后的文本表示。
        """
        tasks = [
            self.process_single_component(
                component,
                id_content_map,
                recursion_depth,
                enable_heavy_media_analysis=enable_heavy_media_analysis,
                enable_voice_transcription=enable_voice_transcription,
            )
            for component in components
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)  # 并行处理多个组件
        processed_texts: List[str] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"处理消息组件时发生错误: {result}")
            else:
                processed_texts.append(result)
        return " ".join(processed_texts)
