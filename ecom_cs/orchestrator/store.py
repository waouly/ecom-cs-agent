"""会话状态存储。

默认内存实现，进程重启即失效——对 Demo 足够。生产环境的差异在「会话要跨实例
共享」，所以接口按 kv 设计（get/put/delete），换成 Redis 只需实现同样的三个方法。
"""

from __future__ import annotations

from typing import Any, Optional

from ..types import SessionState


class SessionStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._data.get(session_id)

    def get_or_create(self, session_id: str, user_id: str) -> SessionState:
        state = self._data.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id, user_id=user_id)
            self._data[session_id] = state
        elif user_id and state.user_id != user_id:
            # 不允许跨用户复用会话，避免状态污染导致的越权
            state.user_id = user_id
        return state

    def put(self, state: SessionState) -> None:
        self._data[state.session_id] = state

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return {sid: s.to_dict() for sid, s in self._data.items()}
