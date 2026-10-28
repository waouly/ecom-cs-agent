"""HTTP 接入层：把编排服务暴露成 FastAPI 应用。

设计上刻意做薄：这里不写任何业务判断，只负责
1) 进程启动时构建一次依赖（图、索引、DB），请求间复用；
2) 把 payload 原样返回，前端/联调能看到 decision、阈值、成本这些调试信息；
3) 提供流式接口，让首字延迟不取决于完整答案的生成时间。
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
