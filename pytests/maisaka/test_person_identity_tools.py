from contextlib import contextmanager
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine
from types import ModuleType, SimpleNamespace
from typing import Dict, Iterator
from unittest.mock import AsyncMock

import json
import pytest

from src.common.database.database_model import PersonInfo
from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import query_memory, query_person_profile
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.person_info import person_info
from src.services.memory_service import MemoryHit, MemorySearchResult


@pytest.fixture(autouse=True)
def isolated_people(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """使用真实查询验证重名和别名，避免接触用户数据库。"""
    engine = create_engine("sqlite://", poolclass=StaticPool)
    PersonInfo.__table__.create(engine)
    with Session(engine) as session:
        session.add_all(
            [
                PersonInfo(
                    person_id="person-1",
                    platform="qq",
                    user_id="100",
                    is_known=True,
                    person_name="小明",
                    user_nickname="爱猫的人",
                    group_cardname=json.dumps([{"group_id": "g1", "group_cardname": "群名片甲"}], ensure_ascii=False),
                ),
                PersonInfo(
                    person_id="person-2",
                    platform="telegram",
                    user_id="200",
                    is_known=True,
                    person_name="小明",
                    user_nickname="小林",
                ),
                PersonInfo(
                    person_id="person-3",
                    platform="qq",
                    user_id="300",
                    is_known=True,
                    person_name="唯一的人",
                    user_nickname="当前昵称",
                ),
            ]
        )
        session.commit()

    @contextmanager
    def database_session(*, auto_commit: bool = False) -> Iterator[Session]:
        del auto_commit
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(person_info, "get_db_session", database_session)
    yield
    engine.dispose()


@pytest.fixture
def tool_context() -> BuiltinToolRuntimeContext:
    runtime = SimpleNamespace(
        session_id="chat-1",
        log_prefix="[测试]",
        chat_stream=SimpleNamespace(platform="qq", user_id="300", group_id=""),
    )
    return BuiltinToolRuntimeContext(engine=SimpleNamespace(), runtime=runtime)


def test_name_resolution_rejects_duplicate_people() -> None:
    with pytest.raises(ValueError, match="对应多个人物") as error:
        person_info.get_person_id_by_person_name("小明")
    assert "person-1" in str(error.value)
    assert "person-2" in str(error.value)


@pytest.mark.parametrize("name", ["爱猫的人", "群名片甲"])
def test_name_resolution_accepts_only_exact_aliases(name: str) -> None:
    assert person_info.get_person_id_by_person_name(name) == "person-1"
    assert person_info.get_person_id_by_person_name(name[:-1]) == ""


@pytest.mark.parametrize("name", ['小"明', "小\\明", "钟离"])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_escaped_group_cardname_cannot_hide_duplicate_nickname(name: str, ensure_ascii: bool) -> None:
    with person_info.get_db_session() as session:
        session.add_all(
            [
                PersonInfo(
                    person_id="card-owner",
                    platform="qq",
                    user_id="400",
                    user_nickname="甲",
                    group_cardname=json.dumps([{"group_id": "g1", "group_cardname": name}], ensure_ascii=ensure_ascii),
                ),
                PersonInfo(person_id="nickname-owner", platform="qq", user_id="500", user_nickname=name),
            ]
        )
        session.commit()
    with pytest.raises(ValueError, match="对应多个人物") as error:
        person_info.get_person_id_by_person_name(name)
    assert "card-owner" in str(error.value)
    assert "nickname-owner" in str(error.value)


@pytest.mark.parametrize("name", ["不存在", "小明", "唯一的人"])
def test_explicit_account_identity_cannot_be_overridden_by_name(name: str) -> None:
    assert person_info.resolve_person_id_for_memory(
        person_name=name, platform="qq", user_id="100"
    ) == person_info.get_person_id("qq", "100")


def test_name_only_resolution_does_not_invent_identity() -> None:
    assert person_info.resolve_person_id_for_memory(person_name="不存在") == ""


@pytest.mark.asyncio
async def test_broad_memory_search_preserves_existing_hit_ids_without_guessing(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
) -> None:
    tool_context.runtime.chat_stream.group_id = "group-1"
    tool_context.runtime.chat_stream.user_id = ""
    search = AsyncMock(
        return_value=MemorySearchResult(
            hits=[
                MemoryHit(
                    content="小明喜欢猫", metadata={"person_id": "person-1", "person_ids": ["person-1", "person-2"]}
                ),
                MemoryHit(content="小明喜欢狗", metadata={"person_id": "person-2"}),
                MemoryHit(content="唯一的人喜欢游戏", metadata={"person_name": "唯一的人"}),
                MemoryHit(content="来源中的人物", source="person_fact:person-source"),
                MemoryHit(content="元数据中的人物", metadata={"source": "person_fact:person-meta"}),
            ]
        )
    )
    monkeypatch.setattr(query_memory.memory_service, "search", search)

    result = await query_memory.handle_tool(
        tool_context,
        ToolInvocation(tool_name="query_memory", arguments={"query": "爱好", "limit": 5}),
    )

    assert result.success
    assert search.await_args.kwargs["person_id"] == ""
    for content in (result.get_history_content(), result.metadata["replyer_memory_reference"]):
        assert '<person person_id="person-1"/> <person person_id="person-2"/> 小明喜欢猫' in content
        assert '<person person_id="person-2"/> 小明喜欢狗' in content
        assert "3. 唯一的人喜欢游戏" in content
        assert "person-3" not in content
        assert '<person person_id="person-source"/> 来源中的人物' in content
        assert '<person person_id="person-meta"/> 元数据中的人物' in content


def test_hit_identity_serialization_escapes_ids_and_skips_absent_ids() -> None:
    rendered = query_memory._format_memory_hit_identities(
        {"person_id": 'person"1', "person_ids": [None, "", 'person"1', "person-2"]}
    )
    assert rendered == '<person person_id="person&quot;1"/> <person person_id="person-2"/>'


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["不存在", "小明"])
@pytest.mark.parametrize("tool_module", [query_memory, query_person_profile], ids=["memory", "profile"])
async def test_invalid_name_never_calls_memory_service(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
    name: str,
    tool_module: ModuleType,
) -> None:
    search = AsyncMock()
    profile = AsyncMock()
    monkeypatch.setattr(query_memory.memory_service, "search", search)
    monkeypatch.setattr(query_person_profile.memory_service, "profile_admin", profile)

    result = await tool_module.handle_tool(
        tool_context,
        ToolInvocation(tool_name=tool_module.get_tool_spec().name, arguments={"query": "爱好", "person_name": name}),
    )

    assert not result.success
    assert "person_id" in result.get_history_content()
    search.assert_not_awaited()
    profile.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_prefers_id_and_keeps_identity_in_model_visible_content(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
) -> None:
    search = AsyncMock(return_value=MemorySearchResult(hits=[MemoryHit(content="喜欢猫")]))
    monkeypatch.setattr(query_memory.memory_service, "search", search)

    result = await query_memory.handle_tool(
        tool_context,
        ToolInvocation(
            tool_name="query_memory", arguments={"query": "爱好", "person_id": "person-1", "person_name": "小明"}
        ),
    )

    assert result.success
    assert search.await_args.kwargs["person_id"] == "person-1"
    assert "person_id=person-1" in result.get_history_content()
    assert "昵称=小明" in result.get_history_content()
    assert "person_id=person-1" in result.metadata["replyer_memory_reference"]


@pytest.mark.asyncio
async def test_empty_person_search_does_not_drop_identity_or_time_filters(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
) -> None:
    search = AsyncMock(return_value=MemorySearchResult())
    monkeypatch.setattr(query_memory.memory_service, "search", search)

    result = await query_memory.handle_tool(
        tool_context,
        ToolInvocation(
            tool_name="query_memory", arguments={"query": "爱好", "person_id": "person-1", "time_start": "2026-09-01"}
        ),
    )

    assert result.success
    search.assert_awaited_once()
    assert search.await_args.kwargs["person_id"] == "person-1"
    assert search.await_args.kwargs["time_start"] == "2026/09/01"
    assert result.structured_content["fallback_applied"] is False
    assert "person_id=person-1" in result.get_history_content()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"person_id": "person-1", "person_name": "小明"}, {"person_name": "群名片甲"}])
async def test_profile_calls_service_by_id_and_exposes_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
    arguments: Dict[str, str],
) -> None:
    profile = AsyncMock(
        return_value={"success": True, "person_id": "person-1", "person_name": "小明", "profile_text": "喜欢猫"}
    )
    monkeypatch.setattr(query_person_profile.memory_service, "profile_admin", profile)

    result = await query_person_profile.handle_tool(
        tool_context,
        ToolInvocation(tool_name="query_person_profile", arguments=arguments),
    )

    assert result.success
    profile.assert_awaited_once_with(action="query", person_id="person-1", limit=8)
    assert "person_id=person-1" in result.get_history_content()
    assert "昵称=小明" in result.get_history_content()


@pytest.mark.asyncio
async def test_profile_rejects_result_for_another_identity(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: BuiltinToolRuntimeContext,
) -> None:
    profile = AsyncMock(return_value={"success": True, "person_id": "person-2", "profile_text": "其他人的画像"})
    monkeypatch.setattr(query_person_profile.memory_service, "profile_admin", profile)

    result = await query_person_profile.handle_tool(
        tool_context,
        ToolInvocation(tool_name="query_person_profile", arguments={"person_id": "person-1"}),
    )

    assert not result.success
    assert "不一致" in result.get_history_content()
