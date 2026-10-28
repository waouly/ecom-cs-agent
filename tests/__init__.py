"""测试包。

用标准库 unittest 而不是 pytest：这个工程的目标是「clone 下来零依赖可跑」，
测试也不该额外要求装东西。同时保持 unittest 写法，装了 pytest 也能直接跑
（`pytest tests` 与 `python -m unittest discover tests` 等价）。
"""
