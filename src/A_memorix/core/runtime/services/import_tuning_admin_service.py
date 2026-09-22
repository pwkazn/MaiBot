from __future__ import annotations

from typing import Any, Dict

from .base import KernelServiceBase


class MemoryImportTuningAdminService(KernelServiceBase):
    async def _ensure_vector_runtime_for_task(self) -> None:
        """创建任务前完成待验证的向量加载，并保留真实故障原因。"""

        if self._vector_health["error_code"] == "embedding_fingerprint_unavailable":
            # 冷启动的首次后台探测尚未执行时，由实际导入或调优请求完成验证。
            recovery = await self._recover_embedding_once()
            if not recovery["success"]:
                raise ValueError(f"任务创建前 Embedding 验证失败: {recovery['report']['message']}")

        # Embedding 请求成功不代表向量加载成功，必须检查恢复后的存储状态。
        if self.vector_store is None and self._vector_health["error_code"]:
            raise ValueError(
                "任务创建前向量存储不可用: "
                f"code={self._vector_health['error_code']}, error={self._vector_health['reason']}"
            )

    async def memory_import_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        manager = self.import_task_manager
        if manager is None:
            return {"success": False, "error": "import manager 未初始化"}

        act = str(action or "").strip().lower()
        if act in {"create_upload", "create_paste", "create_raw_scan", "create_lpmm_openie", "create_maibot_migration"}:
            await self._ensure_vector_runtime_for_task()
        if act in {"settings", "get_settings", "get_guide"}:
            return {"success": True, "settings": await manager.get_runtime_settings()}
        if act in {"path_aliases", "get_path_aliases"}:
            return {"success": True, "path_aliases": manager.get_path_aliases()}
        if act in {"resolve_path", "resolve"}:
            return await manager.resolve_path_request(kwargs)
        if act == "create_upload":
            task = await manager.create_upload_task(
                list(kwargs.get("staged_files") or kwargs.get("files") or kwargs.get("uploads") or []),
                kwargs,
            )
            return {"success": True, "task": task}
        if act == "create_paste":
            return {"success": True, "task": await manager.create_paste_task(kwargs)}
        if act == "create_raw_scan":
            return {"success": True, "task": await manager.create_raw_scan_task(kwargs)}
        if act == "create_lpmm_openie":
            return {"success": True, "task": await manager.create_lpmm_openie_task(kwargs)}
        if act == "create_lpmm_convert":
            return {"success": True, "task": await manager.create_lpmm_convert_task(kwargs)}
        if act == "create_temporal_backfill":
            return {"success": True, "task": await manager.create_temporal_backfill_task(kwargs)}
        if act == "create_maibot_migration":
            return {"success": True, "task": await manager.create_maibot_migration_task(kwargs)}
        if act == "list":
            items = await manager.list_tasks(limit=max(1, int(kwargs.get("limit", 50) or 50)))
            return {"success": True, "items": items, "count": len(items)}
        if act == "get":
            task = await manager.get_task(
                str(kwargs.get("task_id", "") or ""),
                include_chunks=bool(kwargs.get("include_chunks", False)),
            )
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act in {"chunks", "get_chunks"}:
            payload = await manager.get_chunks(
                str(kwargs.get("task_id", "") or ""),
                str(kwargs.get("file_id", "") or ""),
                offset=max(0, int(kwargs.get("offset", 0) or 0)),
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
            )
            return {
                "success": payload is not None,
                **(payload or {}),
                "error": "" if payload is not None else "任务或文件不存在",
            }
        if act == "cancel":
            task = await manager.cancel_task(str(kwargs.get("task_id", "") or ""))
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "retry_failed":
            overrides = kwargs.get("overrides") if isinstance(kwargs.get("overrides"), dict) else kwargs
            task = await manager.retry_failed(str(kwargs.get("task_id", "") or ""), overrides=overrides)
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        return {"success": False, "error": f"不支持的 import action: {act}"}

    async def memory_tuning_admin(self, *, action: str, **kwargs) -> Dict[str, Any]:
        await self.initialize()
        manager = self.retrieval_tuning_manager
        if manager is None:
            return {"success": False, "error": "tuning manager 未初始化"}

        act = str(action or "").strip().lower()
        if act in {"settings", "get_settings"}:
            return {"success": True, "settings": manager.get_runtime_settings()}
        if act == "get_profile":
            profile = manager.get_profile_snapshot()
            persistable_profile = manager.get_persistable_profile(profile)
            return {
                "success": True,
                "profile": profile,
                "runtime_profile": profile,
                "persistable_profile": persistable_profile,
                "toml": manager.export_toml_snippet(persistable_profile),
            }
        if act == "apply_profile":
            profile_raw = kwargs.get("profile")
            if isinstance(profile_raw, dict):
                profile_payload: Dict[str, Any] = dict(profile_raw)
            else:
                profile_payload = {key: value for key, value in kwargs.items() if key not in {"reason", "profile"}}
            return {
                "success": True,
                **await manager.apply_profile(
                    profile_payload,
                    reason=str(kwargs.get("reason", "manual") or "manual"),
                    validate=bool(kwargs.get("validate", True)),
                ),
            }
        if act == "rollback_profile":
            return {"success": True, **await manager.rollback_profile()}
        if act == "export_profile":
            profile = manager.get_profile_snapshot()
            persistable_profile = manager.get_persistable_profile(profile)
            return {
                "success": True,
                "profile": profile,
                "runtime_profile": profile,
                "persistable_profile": persistable_profile,
                "toml": manager.export_toml_snippet(persistable_profile),
            }
        if act == "create_task":
            await self._ensure_vector_runtime_for_task()
            payload = kwargs.get("payload") if isinstance(kwargs.get("payload"), dict) else kwargs
            return {"success": True, "task": await manager.create_task(payload)}
        if act == "list_tasks":
            items = await manager.list_tasks(limit=max(1, int(kwargs.get("limit", 50) or 50)))
            return {"success": True, "items": items, "count": len(items)}
        if act == "get_task":
            task = await manager.get_task(
                str(kwargs.get("task_id", "") or ""),
                include_rounds=bool(kwargs.get("include_rounds", False)),
            )
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "get_rounds":
            payload = await manager.get_rounds(
                str(kwargs.get("task_id", "") or ""),
                offset=max(0, int(kwargs.get("offset", 0) or 0)),
                limit=max(1, int(kwargs.get("limit", 50) or 50)),
            )
            return {
                "success": payload is not None,
                **(payload or {}),
                "error": "" if payload is not None else "任务不存在",
            }
        if act == "cancel":
            task = await manager.cancel_task(str(kwargs.get("task_id", "") or ""))
            return {"success": task is not None, "task": task, "error": "" if task is not None else "任务不存在"}
        if act == "apply_best":
            return {
                "success": True,
                **await manager.apply_best(
                    str(kwargs.get("task_id", "") or ""),
                    validate=bool(kwargs.get("validate", True)),
                ),
            }
        if act == "get_report":
            report = await manager.get_report(
                str(kwargs.get("task_id", "") or ""), fmt=str(kwargs.get("format", "md") or "md")
            )
            return {
                "success": report is not None,
                "report": report,
                "error": "" if report is not None else "任务不存在",
            }
        return {"success": False, "error": f"不支持的 tuning action: {act}"}
