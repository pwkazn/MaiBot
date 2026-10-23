"""引用原消息查询的会话隔离与适配器边界测试。"""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import importlib.util
import sys

import pytest


def _message(message_id="current", *, session_id="real-session", group_id="room", user_id="alice"):
    return SimpleNamespace(
        message_id=message_id,
        session_id=session_id,
        platform="test-platform",
        raw_message=SimpleNamespace(components=[SimpleNamespace(type="image", binary_data=b"image-data")]),
        message_info=SimpleNamespace(
            group_info=SimpleNamespace(group_id=group_id) if group_id else None,
            user_info=SimpleNamespace(user_id=user_id),
            additional_config={"account_id": "bot"},
        ),
    )


@pytest.fixture
def resolver_env(monkeypatch):
    """只替换 DB 与 RPC 边界，避免测试启动真实插件或项目数据库。"""

    class Query:
        def filter_by(self, **filters):
            self.filters = filters
            return self

        def order_by(self, _order):
            return self

        def limit(self, _limit):
            return self

    class DatabaseSession:
        record = None
        error = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def exec(self, statement):
            if self.error:
                raise self.error
            self.statement = statement
            return self

        def first(self):
            return self.record

    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    db = DatabaseSession()
    entry = SimpleNamespace(plugin_id="adapter-test", public=True, handler_name="get_standard_message", timeout_ms=2000)
    supervisor = SimpleNamespace(
        api_registry=SimpleNamespace(
            get_api=lambda plugin_id, _name, **_kwargs: entry if plugin_id == entry.plugin_id else None
        ),
        invoke_api=AsyncMock(),
    )
    driver = SimpleNamespace(descriptor=SimpleNamespace(plugin_id="adapter-test"))
    routes = SimpleNamespace(
        build_route_key_from_message=lambda message: SimpleNamespace(
            account_id=message.message_info.additional_config.get("account_id"),
            scope=message.message_info.additional_config.get("scope"),
        ),
        resolve_drivers=lambda _route: [driver],
        receive_route_table=SimpleNamespace(resolve_bindings=lambda _route: []),
        driver_registry=SimpleNamespace(get=lambda _driver_id: driver),
    )
    runtime = SimpleNamespace(supervisors=[supervisor])
    stub("src.common.database.database", get_db_session=lambda: db)
    stub("src.common.database.database_model", Messages=SimpleNamespace(timestamp=SimpleNamespace(desc=lambda: None)))
    stub("src.platform_io", get_platform_io_manager=lambda: routes)
    stub("src.plugin_runtime.host.component_timeout", resolve_component_rpc_timeout_ms=lambda value: value or 60000)
    stub(
        "src.chat.message_receive.message",
        SessionMessage=SimpleNamespace(from_db_instance=lambda record: record),
    )
    stub("src.plugin_runtime.integration", get_plugin_runtime_manager=lambda: runtime)
    stub(
        "src.plugin_runtime.host.message_utils",
        PluginMessageUtils=SimpleNamespace(_build_session_message_from_dict=lambda payload: payload["message"]),
    )
    path = Path(__file__).resolve().parents[2] / "src/chat/message_receive/reply_resolver.py"
    spec = importlib.util.spec_from_file_location("src.chat.message_receive._reply_resolver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "select", lambda _model: Query())
    return SimpleNamespace(module=module, db=db, supervisor=supervisor, routes=routes, runtime=runtime, entry=entry)


@pytest.mark.asyncio
async def test_local_lookup_keeps_structured_media_and_is_scoped(resolver_env):
    original = _message("quoted")
    resolver_env.db.record = original

    result = await resolver_env.module.resolve_reply_message(_message(), "quoted")

    assert result is original
    assert result.raw_message.components[0].binary_data == b"image-data"
    assert resolver_env.db.statement.filters == {
        "session_id": "real-session",
        "platform": "test-platform",
        "message_id": "quoted",
    }
    resolver_env.supervisor.invoke_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_only_adapter_returns_media_without_registering_session(resolver_env):
    original = _message("quoted", session_id="")
    resolver_env.routes.resolve_drivers = lambda _route: []
    resolver_env.routes.receive_route_table.resolve_bindings = lambda _route: [SimpleNamespace(driver_id="inbound")]
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
        error=None, payload={"success": True, "result": {"message": original}}
    )

    result = await resolver_env.module.resolve_reply_message(_message(), "quoted")

    assert result is original
    assert result.session_id == "real-session"
    assert result.raw_message.components[0].binary_data == b"image-data"
    resolver_env.supervisor.invoke_api.assert_awaited_once_with(
        plugin_id="adapter-test",
        component_name="get_standard_message",
        args={"message_id": "quoted"},
        timeout_ms=2000,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("message_id", "wrong"), ("platform", "wrong"), ("session_id", "wrong")])
async def test_rejects_message_identity_from_different_resource(resolver_env, field, value):
    original = _message("quoted", session_id="")
    setattr(original, field, value)
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
        error=None, payload={"success": True, "result": {"message": original}}
    )
    with pytest.raises(ValueError):
        await resolver_env.module.resolve_reply_message(_message(), "quoted")


@pytest.mark.asyncio
async def test_rejects_wrong_group_and_private_chat_target(resolver_env):
    for current, original in [
        (_message(), _message("quoted", group_id="another-room")),
        (_message(group_id=""), _message("quoted", group_id="", user_id="someone-else")),
    ]:
        resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
            error=None, payload={"success": True, "result": {"message": original}}
        )
        with pytest.raises(ValueError, match="聊天目标"):
            await resolver_env.module.resolve_reply_message(current, "quoted")


@pytest.mark.asyncio
async def test_private_bot_message_uses_explicit_peer(resolver_env):
    original = _message("quoted", group_id="", user_id="bot")
    original.message_info.additional_config["platform_io_target_user_id"] = "alice"
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
        error=None, payload={"success": True, "result": {"message": original}}
    )
    assert await resolver_env.module.resolve_reply_message(_message(group_id=""), "quoted") is original


@pytest.mark.asyncio
@pytest.mark.parametrize("route_field", ["account_id", "scope"])
async def test_rejects_other_bot_account_or_route_scope(resolver_env, route_field):
    original = _message("quoted")
    original.message_info.additional_config[route_field] = "another-route"
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
        error=None, payload={"success": True, "result": {"message": original}}
    )
    with pytest.raises(ValueError, match="账号或路由作用域"):
        await resolver_env.module.resolve_reply_message(_message(), "quoted")


@pytest.mark.asyncio
async def test_rejects_ambiguous_adapter_without_calling_platform(resolver_env):
    resolver_env.runtime.supervisors.append(resolver_env.supervisor)
    with pytest.raises(RuntimeError, match="多个消息查询适配器"):
        await resolver_env.module.resolve_reply_message(_message(), "quoted")
    resolver_env.supervisor.invoke_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_errors_are_not_treated_as_missing_messages(resolver_env):
    resolver_env.db.error = RuntimeError("database failed")
    with pytest.raises(RuntimeError, match="database failed"):
        await resolver_env.module.resolve_reply_message(_message(), "quoted")
    resolver_env.db.error = None
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(error={"message": "RPC failed"}, payload=None)
    with pytest.raises(RuntimeError, match="RPC failed"):
        await resolver_env.module.resolve_reply_message(_message(), "quoted")


@pytest.mark.asyncio
async def test_unsupported_and_inaccessible_messages_return_none(resolver_env):
    resolver_env.supervisor.invoke_api.return_value = SimpleNamespace(
        error=None, payload={"success": True, "result": None}
    )
    assert await resolver_env.module.resolve_reply_message(_message(), "quoted") is None
    resolver_env.runtime.supervisors = []
    assert await resolver_env.module.resolve_reply_message(_message(), "quoted") is None
