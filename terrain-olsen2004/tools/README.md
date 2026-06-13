# erosion_review.py — HeightField 侵蚀只读复查工具

非破坏性（**绝不保存 hip**）。逐帧 cook 一个"侵蚀结果节点"，产出复查四件套。
所有分析量（坡度图 / Olsen 侵蚀评分 ε / 守恒 / hillshade）都在 numpy 里从 height
体素直接算，不依赖具体工程的 score 节点 —— 换侵蚀工程也能直接用。

## 跑法（hython）

```
# 本工程默认值已内置，直接跑：
hython tools\erosion_review.py

# 自定义帧 + 输出子目录：
hython tools\erosion_review.py --frames 1 10 30 60 100 --tag run1

# 显式指定节点（换工程时）：
hython tools\erosion_review.py ^
  --hip   F:\path\Scene.hip ^
  --result /obj/.../solver1            # 每帧输出已侵蚀地形的节点 ^
  --base   /obj/.../heightfield_noise1 # 未侵蚀基底（帧无关） ^
  --talus-parm /obj/.../talusangle     # 安息角参数(度)，或用 --repose-deg 33 ^
  --out F:\path\review_renders

# 两版对照（经典版 vs Olsen 版）：叠加评分轨迹 + 并排 hillshade
hython tools\erosion_review.py --result <olsen> --result-b <classic> --label A=Olsen B=Classic
```

hython 路径（本机）：`D:\Program Files (x86)\Houdini\bin\hython.exe`

## 产出

| 文件 | 内容 |
|------|------|
| 控制台表格 | 逐帧 s̄ / σ / ε / mass / **守恒漂移** |
| `score_trajectory.png` | ε 轨迹 + mean/std 双轴（两版时叠加） |
| `slope_histogram.png` | 基底 vs 末帧坡度分布 + 安息角竖线（**碎屑坡 = 在安息角处堆峰**） |
| `hillshade_compare.png` | 基底 \| 末帧 \| 高度差（蓝=沉积 红=侵蚀） |

## 关键参数

- `--result` 必须是"每帧输出当前迭代侵蚀结果"的节点（这里是 Solver SOP）。
- `--base` 是未侵蚀基底，用作守恒基准与对比左图。
- 安息角：优先 `--talus-parm`（读 HDA 参数），否则 `--repose-deg` 直接给度数。
- `--cell` 体素水平尺寸；不给则从 `--gridspacing-node` 的 `gridspacing` 自动探测。

## 已知结论（study1 Olsen2004 复现）

- 纯热侵蚀 **ε 必降**：坡度同质化堆向安息角 → σ 塌 → ε=σ/s̄ 降。这是正确行为，
  非 bug。论文 Fig 15 的 ε 升需要**封闭盆地**被沉积物填平（s̄ 狂掉而 talus 墙保住 σ），
  开放噪声地形演化不出。
- 基底 `elementsize=500` 太平（90% 体素已缓于 33°），侵蚀几乎无活；`~250` 才有料。

## 待办 / 升级方向

- [ ] 抽象成 skill（参数化已就绪，缺触发词 + 自动定位节点的启发式）
- [ ] 支持直接对比两个不同 hip 文件
- [ ] 坡度直方图叠加 HF Erode 原生结果做三方对照
