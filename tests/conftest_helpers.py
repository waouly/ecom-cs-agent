"""共用夹具：临时目录 + 隔离数据库 + 一套已装配依赖。"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from ecom_cs.config import PROJECT_ROOT, Settings
from ecom_cs.orchestrator import ChatService, build_deps


def make_settings(tmp: Path) -> Settings:
    base = Settings()
    return dataclasses.replace(
        base,
        data_dir=tmp,
        db_path=tmp / "app.db",
        calibration_path=tmp / "calibration.json",
        policies_path=PROJECT_ROOT / "configs" / "policies",
    )


class TempWorld(unittest.TestCase):
    """每个测试一个干净的库，避免写操作用例互相污染。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmpdir.name)
        cls.settings = make_settings(cls.tmp)
        cls.deps = build_deps(settings=cls.settings)
        cls.service = ChatService(deps=cls.deps)

    @classmethod
    def tearDownClass(cls) -> None:
        # Windows 上打开着的 sqlite 文件不能删，先关连接再清理临时目录
        try:
            cls.deps.repo.conn.close()
        except Exception:
            pass
        cls._tmpdir.cleanup()

    def ask(self, text: str, session: str = "t", user: str = "u_demo") -> dict:
        return self.service.respond(session, user, text)
