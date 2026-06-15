# 全量结构快照: 树+接线 / 关键参数 / uplift 在各点软硬 / 各 wrangle 角色. 只读不存。
import hou
import numpy as np
HIP = r"F:\houdini\C_studyProject\terrain\Schott2023\Schott2023.hip"
hou.hipFile.load(HIP, suppress_save_prompt=True, ignore_load_warnings=True)
GEO = "/obj/Schott2023/landscape/"
ls = hou.node(GEO.rstrip("/"))

print("===== 1. 节点树 + 接线 =====")
for c in ls.children():
    ins = ",".join(i.name() for i in c.inputs() if i)
    fl = ("D" if c.isDisplayFlagSet() else "") + ("B" if c.isBypassed() else "")
    print("  %-22s [%-20s] <- [%s] %s" % (c.name(), c.type().name(), ins, fl))

print("\n===== 2. loop 主链真正读谁 =====")
rb = hou.node(GEO + "repeat_begin1")
print("  repeat_begin1 <- ", [i.name() for i in rb.inputs() if i])

print("\n===== 3. 关键参数 =====")
print("  HDA:", {p: round(ls.parm(p).eval(), 3) for p in ("dt", "u", "n", "m", "kd") if ls.parm(p)})
print("  外层 N =", hou.node(GEO + "repeat_end1").parm("iterations").eval(),
      "| 内层 K =", hou.node(GEO + "repeat_end2").parm("iterations").eval())
b = hou.node(GEO + "heightfield_blur1")
if b:
    print("  blur1: layer=%r radius=%s iterations=%s" % (b.parm("layer").eval(), b.parm("radius").eval(), b.parm("iterations").eval()))
dm = hou.node(GEO + "heightfield_drawmask1")
if dm and dm.parm("masklayer"):
    print("  drawmask1: masklayer=%r" % dm.parm("masklayer").eval())
viz = hou.node(GEO + "heightfield_visualize1")
if viz:
    for p in viz.parms():
        if "layer" in p.name().lower() and isinstance(p.eval(), str) and p.eval().strip():
            print("  visualize1: %s=%r" % (p.name(), p.eval()))


def up_trans(name):
    n = hou.node(GEO + name)
    if not n:
        return "(无此节点)"
    try:
        n.cook(force=True)
        for pr in n.geometry().prims():
            if pr.type() == hou.primType.Volume and pr.attribValue("name") == "uplift":
                r = pr.resolution()
                a = np.array(pr.allVoxels(), dtype=np.float64).reshape((r[2], r[1], r[0]))[0]
                return "max=%.2f 过渡带=%.1f%% (%s)" % (a.max(), 100 * ((a > 0.05) & (a < 0.95)).mean(),
                                                     "软" if ((a > 0.05) & (a < 0.95)).mean() > 0.05 else "硬0/1")
    except Exception as e:
        return "cook失败:%s" % str(e)[:60]
    return "无uplift层"


print("\n===== 4. uplift 软硬 (过渡带>5%=软) =====")
for nm in ("heightfield_drawmask1", "heightfield_blur1", "repeat_begin1", "output0"):
    print("  %-22s %s" % (nm, up_trans(nm)))

print("\n===== 5. wrangle 角色 (snippet 关键词) =====")
for nm in ("volumewrangle3", "volumewrangle4", "volumewrangle5"):
    n = hou.node(GEO + nm)
    if n and n.parm("snippet"):
        code = n.parm("snippet").eval()
        role = []
        if "f@slope" in code: role.append("写slope(D)")
        if "f@drain" in code: role.append("写drain(gather)")
        if "@height =" in code or "@height=" in code.replace(" ", ""): role.append("写height(erode)")
        if "@uplift" in code: role.append("读uplift")
        print("  %-16s %s" % (nm, " + ".join(role) or "?"))
print("=== END ===")
