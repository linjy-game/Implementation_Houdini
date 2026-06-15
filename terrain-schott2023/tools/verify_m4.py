# M4 验收：参数 + uplift/height/drain 三联图 + uplift 是否驱动了地形。只读不存。
import hou
import numpy as np
import os
HIP = r"F:\houdini\C_studyProject\terrain\Schott2023\Schott2023.hip"
GEO = "/obj/Schott2023/landscape/"
OUT_PNG = r"F:\houdini\C_studyProject\terrain\Schott2023\review_renders\m4_panels.png"
hou.hipFile.load(HIP, suppress_save_prompt=True, ignore_load_warnings=True)

hda = hou.node(GEO.rstrip("/"))
print("HDA 参数:", {p: round(hda.parm(p).eval(), 3) for p in ("dt", "u", "n", "m", "kd")})
print("外层 N =", hou.node(GEO + "repeat_end1").parm("iterations").eval(),
      " 内层 K =", hou.node(GEO + "repeat_end2").parm("iterations").eval())

out = hou.node(GEO + "output0")
base = hou.node(GEO + "heightfield_noise1")


def layer(geo, name):
    for pr in geo.prims():
        if pr.type() == hou.primType.Volume and pr.attribValue("name") == name:
            r = pr.resolution()
            return np.array(pr.allVoxels(), dtype=np.float64).reshape((r[2], r[1], r[0]))[0]
    return None


def hs(z, dx=2.0):
    gy, gx = np.gradient(z, dx)
    sl = np.pi / 2 - np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    return np.clip(np.sin(np.deg2rad(45)) * np.sin(sl) + np.cos(np.deg2rad(45)) * np.cos(sl) * np.cos(np.deg2rad(315) - asp), 0, 1)


base.cook(force=True); out.cook(force=True)
h0 = layer(base.geometry(), "height")
h1 = layer(out.geometry(), "height")
up = layer(out.geometry(), "uplift")
drn = layer(out.geometry(), "drain")

print("\nuplift: mean=%.3f max=%.3f 非零%.1f%%" % (up.mean(), up.max(), 100 * (up != 0).mean()))
print("height 侵蚀前 %.0f~%.0f  侵蚀后 %.0f~%.0f  |Δh|mean=%.2f finite=%s"
      % (h0.min(), h0.max(), h1.min(), h1.max(), np.abs(h1 - h0).mean(), bool(np.isfinite(h1).all())))
# uplift 是否驱动地形: 高 uplift 处是否变高
dh = h1 - h0
corr = np.corrcoef(up.ravel(), dh.ravel())[0, 1]
print("corr(uplift, Δh) = %.3f  (>0 强 = 你刷的地方在隆起 = 起效; ≈0 = 没驱动)" % corr)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
ax[0].imshow(up, cmap="magma", origin="lower"); ax[0].set_title("uplift field (painted)")
ax[1].imshow(hs(h1), cmap="gray", origin="lower"); ax[1].set_title("eroded hillshade")
ax[2].imshow(np.log10(np.maximum(drn, 1)), cmap="inferno", origin="lower"); ax[2].set_title("drainage log")
for a in ax: a.axis("off")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=100)
print("已存:", OUT_PNG)
print("=== END ===")
