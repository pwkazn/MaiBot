"""按当前聊天流与消息 ID 读取引用原消息。"""

from typing import TYPE_CHECKING, Optional, Tuple

from sqlmodel import select

from src.common.database.database import get_db_session
from src.common.database.database_model import Messages
from src.platform_io import get_platform_io_manager
from src.plugin_runtime.host.component_timeout import resolve_component_rpc_timeout_ms

if TYPE_CHECKING:
    from .message import SessionMessage


def _chat_target(message: "SessionMessage") -> Tuple[str, str]:
    """提取聊天目标；私聊发送者可能是机器人，优先使用明确的收件目标。"""

    group_info = message.message_info.group_info
    if group_info is not None:
        return "group", group_info.group_id
    target_user_id = message.message_info.additional_config.get("platform_io_target_user_id")
    return "private", str(target_user_id or message.message_info.user_info.user_id)


async def resolve_reply_message(message: "SessionMessage", message_id: str) -> Optional["SessionMessage"]:
    """优先读取本地结构化消息，未保存时请求当前适配器的标准消息查询 API。

    只使用当前消息已注册的真实聊天流 ID，不注册查询结果，也不写入消息数据库。
    未提供查询能力或平台确认原消息不可访问时返回 ``None``，查询错误直接上抛。
    """

    from .message import SessionMessage

    if not message_id.strip():
        raise ValueError("引用消息 ID 不能为空")
    if not message.session_id:
        return None

    with get_db_session() as session:
        statement = (
            select(Messages)
            .filter_by(session_id=message.session_id, platform=message.platform, message_id=message_id)
            .order_by(Messages.timestamp.desc())
            .limit(1)
        )
        record = session.exec(statement).first()
        if record is not None:
            return SessionMessage.from_db_instance(record)

    from src.plugin_runtime.integration import get_plugin_runtime_manager
    from src.plugin_runtime.host.message_utils import PluginMessageUtils

    platform_io = get_platform_io_manager()
    route_key = platform_io.build_route_key_from_message(message)
    # 接收专用适配器也可以提供查询能力，不能只遍历发送路由。
    drivers = list(platform_io.resolve_drivers(route_key))
    for binding in platform_io.receive_route_table.resolve_bindings(route_key):
        driver = platform_io.driver_registry.get(binding.driver_id)
        if driver is not None:
            drivers.append(driver)
    plugin_ids = {driver.descriptor.plugin_id for driver in drivers if driver.descriptor.plugin_id}
    if not plugin_ids:
        return None
    candidates = []
    for supervisor in get_plugin_runtime_manager().supervisors:
        for plugin_id in plugin_ids:
            entry = supervisor.api_registry.get_api(
                plugin_id,
                "adapter.message.get",
                version="1",
                session_id=message.session_id,
            )
            if entry is not None and entry.public:
                candidates.append((supervisor, entry))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("当前聊天流匹配到多个消息查询适配器，无法确定引用原消息的来源")

    supervisor, entry = candidates[0]
    response = await supervisor.invoke_api(
        plugin_id=entry.plugin_id,
        component_name=entry.handler_name,
        args={"message_id": message_id},
        timeout_ms=resolve_component_rpc_timeout_ms(entry.timeout_ms),
    )
    if response.error:
        raise RuntimeError(f"获取引用原消息失败: {response.error}")
    payload = response.payload
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"获取引用原消息失败: {payload}")
    result = payload.get("result")
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError("适配器消息查询 API 未返回标准消息字典")

    target_message = PluginMessageUtils._build_session_message_from_dict(result)
    if target_message.message_id != message_id or target_message.platform != message.platform:
        raise ValueError("适配器返回的引用原消息 ID 或平台与请求不一致")
    if _chat_target(target_message) != _chat_target(message):
        raise ValueError("适配器返回的引用原消息不属于当前聊天目标")
    target_route = platform_io.build_route_key_from_message(target_message)
    if (target_route.account_id is not None and target_route.account_id != route_key.account_id) or (
        target_route.scope is not None and target_route.scope != route_key.scope
    ):
        raise ValueError("适配器返回的引用原消息不属于当前账号或路由作用域")
    if target_message.session_id and target_message.session_id != message.session_id:
        raise ValueError("适配器返回的引用原消息不属于当前聊天流")
    target_message.session_id = message.session_id
    return target_message
