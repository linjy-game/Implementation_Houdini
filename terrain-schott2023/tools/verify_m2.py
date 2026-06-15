# M2 验收：确认 wrangle5 侵蚀代码 + 参数 -> cook -> 对比侵蚀前后 height -> 三联 hillshade/河网图。只读不存。
import hou
import numpy as np
import os

HIP = r"F:\houdini\C_studyProject\terrain\Schott2023\Schott2023.hip"
GEO = "/obj/Schott2023/landscape/"
OUT_PNG = r"F:\houdini\C_studyProject\terrain\Schott2023\review_renders\m2_panels.png"
hou.hipFile.load(HIP, suppress_save_prompt=True, ignore_load_warnings=True)
print("LOADED:", HIP)

# 迭代次数
re = hou.node(GEO + "repeat_end1")
print("iterations =", re.parm("iterations").eval() if re else "?")

# wrangle5 VEX 头 + 参数
w5 = hou.node(GEO + "volumewrangle5")
code = w5.parm("snippet").eval() if w5 and w5.parm("snippet") else ""
is_erosion = "@height = " in code or "@height=" in code.replace(" ", "")
print("\nwrangle5 是侵蚀代码?", is_erosion, " (含 @height 写入)")
hda = hou.node(GEO.rstrip("/"))   # 参数提到了 HDA 顶层(chf("../..")), 在这读
print("HDA 顶层参数:", {p: (hda.parm(p).eval() if hda.parm(p) else "缺失") for p in ("dt", "u", "n", "m", "kd")})

# cook 基底(侵蚀前) 与 output(侵蚀后)
base_node = hou.node(GEO + "heightfield_noise1")
out = hou.node(GEO + "output0")
print("\n--- COOK ---")
try:
    base_node.cook(force=True)
    out.cook(force=True)
    print("cook 完成")
except Exception as e:
    print("COOK FAILED:", e)
    for n in (w5, re, out):
        if n and n.errors():
            print("  [ERROR]", n.name(), n.errors()[0][:200])
    raise SystemExit(0)


def layer(geo, name):
    for pr in geo.prims():
        if pr.type() == hou.primType.Volume and pr.attribValue("name") == name:
            r = pr.resolution()
            return np.array(pr.allVoxels(), dtype=np.float64).reshape((r[2], r[1], r[0]))[0]
    return None


h0 = layer(base_node.geometry(), "height")     # 侵蚀前
h1 = layer(out.geometry(), "height")           # 侵蚀后
drn = layer(out.geometry(), "drain")

print("\n--- HEIGHT ---")
print("finite:", bool(np.isfinite(h1).all()))
print("侵蚀前 min/mean/max: %.1f / %.1f / %.1f" % (h0.min(), h0.mean(), h0.max()))
print("侵蚀后 min/mean/max: %.1f / %.1f / %.1f" % (h1.min(), h1.mean(), h1.max()))
d = h1 - h0
print("变化量 |Δh| mean=%.3f max=%.3f  (≈0=没侵蚀; 巨大/inf=炸了)" % (np.abs(d).mean(), np.abs(d).max()))
print("净抬升 mean(Δh)=%.3f  (>0 抬升占优, <0 侵蚀占优)" % d.mean())

# 河网-河谷对齐: 高 drain 处应是局部低洼。算 (h - 3x3均值) 的凹陷度 vs log(drain) 相关
from numpy.lib.stride_tricks import sliding_window_view
pad = np.pad(h1, 1, mode="edge")
local_mean = sliding_window_view(pad, (3, 3)).mean(axis=(-1, -2))
concavity = local_mean - h1            # >0 = 比邻居低 = 洼地
inter = slice(1, -1)
c = concavity[inter, inter].ravel()
ld = np.log(np.maximum(drn[inter, inter], 1.0)).ravel()
corr = np.corrcoef(ld, c)[0, 1]
print("\n河网-河谷对齐: corr(log drain, 洼陷度) = %.3f  (>0 且越大=高汇水处越低洼=河谷, 对!)" % corr)

# 三联图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def hs(z, dx=2.0):
        gy, gx = np.gradient(z, dx)
        slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az = np.deg2rad(315); alt = np.deg2rad(45)
        v = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        return np.clip(v, 0, 1)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(hs(h0), cmap="gray", origin="lower"); ax[0].set_title("侵蚀前 base")
    ax[1].imshow(hs(h1), cmap="gray", origin="lower"); ax[1].set_title("侵蚀后 eroded")
    ax[2].imshow(np.log10(np.maximum(drn, 1)), cmap="inferno", origin="lower"); ax[2].set_title("drainage (log)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=100)
    print("\n已存三联图:", OUT_PNG)
except Exception as e:
    print("出图失败:", e)
print("=== END ===")
