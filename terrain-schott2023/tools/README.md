# tools — Schott 2023 复现的离线验证脚本

全部用 `hython` 跑，**只读不存**（绝不保存 hip），逐项把侵蚀引擎量出来。

| 脚本 | 作用 |
|---|---|
| `snapshot.py` | 全量结构快照：节点树+接线 / 关键参数 / uplift 在各点软硬 / 各 wrangle 角色。换工程也能用。 |
| `verify_m2.py` | M2 侵蚀验收：对比侵蚀前后 height、稳态、`corr(drain, 洼陷度)` 河谷对齐度、三联 hillshade。 |
| `verify_m4.py` | M4 抬升验收：uplift 场 / 侵蚀后 hillshade / 河网三联 + `corr(uplift, Δh)` 抬升驱动度。 |

跑法：`hython verify_m4.py`（路径写死在脚本顶部，换工程改 `HIP`/`GEO`）。

## drainage_gather_cop.cl — GPU 港的 kernel 草稿（未落地）

本次未做 GPU 版（COP 是独立学习项目）。`drainage_gather_cop.cl` 是按 HF Erode v3.0 的 Copernicus COP OpenCL 写法（`@KERNEL/@ixy/@layer.bufferIndex`）写的 **pass1(算D) + pass2(gather汇水)** kernel 草稿，留作将来在 copnet 里港 GPU 的起点。注意 gather 不需 `@WRITEBACK`：COP 节点输入≠输出，一个 opencl 节点 = 一次松弛。
