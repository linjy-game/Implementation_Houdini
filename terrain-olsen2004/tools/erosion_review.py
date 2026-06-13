# -*- coding: utf-8 -*-
"""
erosion_review.py — HeightField 侵蚀只读复查工具（非破坏性，绝不保存 hip）

用 hython 跑。对一个"侵蚀结果节点"逐帧采样，产出四件套：
  1) 侵蚀评分轨迹  score_trajectory.png   (Olsen ε = σ_s / s̄，以及 mean/std)
  2) 守恒检查      控制台表格            (每帧 height 体素总和漂移)
  3) 坡度直方图    slope_histogram.png    (基底 vs 末帧，标出安息角)
  4) 地形对比      hillshade_compare.png  (基底 | 末帧 | 高度差)

所有分析量(坡度/评分/守恒/hillshade)都在 numpy 里从 height 体素直接算，
不依赖工程里特定的 score 节点 —— 换一个侵蚀工程也能直接用。

典型用法(本工程默认值已内置):
  hython erosion_review.py
显式指定:
  hython erosion_review.py --hip F:\...\Study1.hip ^
      --result /obj/LandScape/landscape/solver1 ^
      --base   /obj/LandScape/landscape/heightfield_noise1 ^
      --talus-parm /obj/LandScape/landscape/talusangle ^
      --frames 1 10 30 60 100 --out F:\...\review_renders
两版对照(经典版 vs Olsen 版,叠加评分轨迹 + 并排 hillshade):
  hython erosion_review.py --result <olsenSolver> --result-b <classicSolver> --label A=Olsen B=Classic
"""
import os, sys, argparse
import hou
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- 默认值(本工程) -----------------
D_HIP    = r"F:\houdini\C_studyProject\study1\Study1.hip"
D_RESULT = "/obj/LandScape/landscape/solver1"            # 每帧输出已侵蚀地形的节点
D_BASE   = "/obj/LandScape/landscape/heightfield_noise1" # 未侵蚀基底(帧无关)
D_TALUS  = "/obj/LandScape/landscape/talusangle"         # 安息角参数(度)
D_GSNODE = "/obj/LandScape/landscape/heightfield1"       # 取 gridspacing 的节点
D_OUT    = r"F:\houdini\C_studyProject\study1\review_renders"

def parse_args():
    p = argparse.ArgumentParser(description="HeightField 侵蚀只读复查")
    p.add_argument("--hip", default=D_HIP)
    p.add_argument("--result", default=D_RESULT, help="逐帧输出侵蚀结果的节点路径")
    p.add_argument("--result-b", default=None, help="第二版结果节点(两版对照)")
    p.add_argument("--label", nargs="*", default=["A=A", "B=B"], help="形如 A=Olsen B=Classic")
    p.add_argument("--base", default=D_BASE, help="未侵蚀基底节点")
    p.add_argument("--talus-parm", default=D_TALUS, help="安息角参数路径(度);没有就用 --repose-deg")
    p.add_argument("--repose-deg", type=float, default=None, help="直接给安息角(度),覆盖 --talus-parm")
    p.add_argument("--gridspacing-node", default=D_GSNODE)
    p.add_argument("--cell", type=float, default=None, help="直接给体素水平尺寸,覆盖自动探测")
    p.add_argument("--height-layer", default="height")
    p.add_argument("--conserve-layers", default="height",
                   help="守恒检查累加的层(逗号分隔)。热侵蚀=height；水力侵蚀=height,sediment")
    p.add_argument("--extra-layer", default=None,
                   help="额外可视化的标量场(如 water)，单独出一张 <名>_field.png")
    p.add_argument("--frames", type=int, nargs="*", default=[1,5,10,20,30,50,80])
    p.add_argument("--final", type=int, default=None, help="用于直方图/hillshade 的帧(默认 frames 最大值)")
    p.add_argument("--out", default=D_OUT)
    p.add_argument("--tag", default=None, help="输出子目录名")
    return p.parse_args()

# ----------------- 几何工具 -----------------
def find_volume(geo, name):
    if geo is None: return None
    for prim in geo.prims():
        if isinstance(prim, hou.Volume):
            nm = prim.attribValue("name") if geo.findPrimAttrib("name") else ""
            if nm == name:
                return prim
    return None

def height_array(node, layer):
    prim = find_volume(node.geometry(), layer)
    if prim is None:
        raise RuntimeError(f"节点 {node.path()} 上找不到体素层 '{layer}'")
    rx, ry, rz = prim.resolution()
    return np.array(prim.allVoxels(), dtype=np.float64).reshape((ry, rx))

def height_mass(node, layer):
    prim = find_volume(node.geometry(), layer)
    r = prim.resolution()
    return prim.volumeAverage() * r[0] * r[1] * r[2]

def layers_mass(node, layers):
    """守恒检查：把若干层的体素总和相加（缺失的层按 0 计）。
    热侵蚀守恒 height；水力侵蚀守恒 height+sediment（水溶解后悬浮，仍是物质）。"""
    geo = node.geometry(); total = 0.0
    for ly in layers:
        prim = find_volume(geo, ly)
        if prim is not None:
            r = prim.resolution()
            total += prim.volumeAverage() * r[0] * r[1] * r[2]
    return total

# ----------------- 分析(纯 numpy) -----------------
def slope_olsen(z, cell):
    """Olsen 坡度图 s = max(4 邻域 |Δh|) / 水平尺寸;裁掉 1 像素边避免环绕误差。"""
    up = np.abs(z - np.roll(z, -1, 0)); dn = np.abs(z - np.roll(z, 1, 0))
    lf = np.abs(z - np.roll(z, 1, 1));  rt = np.abs(z - np.roll(z, -1, 1))
    s = np.maximum.reduce([up, dn, lf, rt]) / cell
    return s[1:-1, 1:-1]

def erosion_score(z, cell):
    s = slope_olsen(z, cell)
    sb = float(s.mean()); sg = float(s.std())
    return sb, sg, (sg/sb if sb > 1e-9 else 0.0)

def hillshade(z, cell, az=315.0, alt=45.0):
    az = np.radians(360.0 - az + 90.0); zen = np.radians(90.0 - alt)
    dy, dx = np.gradient(z, cell, cell)
    slope = np.arctan(np.hypot(dx, dy)); aspect = np.arctan2(-dx, dy)
    return np.clip(np.cos(zen)*np.cos(slope) + np.sin(zen)*np.sin(slope)*np.cos(az-aspect), 0, 1)

# ----------------- 主流程 -----------------
def sweep(result_node, base_mass, height_layer, conserve_layers, cell, frames, final_frame):
    """逐帧 cook,返回 {frame: (s_bar,sigma,eps,mass,drift)} 与末帧 height 数组。
    评分/坡度用 height_layer；守恒用 conserve_layers 的总和。"""
    rows = {}; final_h = None
    for f in sorted(set(frames + [final_frame])):
        hou.setFrame(f)
        z = height_array(result_node, height_layer)
        sb, sg, ep = erosion_score(z, cell)
        m = layers_mass(result_node, conserve_layers)
        rows[f] = (sb, sg, ep, m, m - base_mass)
        if f == final_frame:
            final_h = z
    return rows, final_h

def main():
    a = parse_args()
    hou.hipFile.load(a.hip, suppress_save_prompt=True, ignore_load_warnings=True)

    result = hou.node(a.result)
    base   = hou.node(a.base)
    if result is None: sys.exit(f"找不到结果节点: {a.result}")
    if base is None:   sys.exit(f"找不到基底节点: {a.base}")

    # gridspacing
    if a.cell is not None:
        cell = a.cell
    else:
        gsn = hou.node(a.gridspacing_node)
        cell = gsn.parm("gridspacing").eval() if (gsn and gsn.parm("gridspacing")) else 1.0

    # 安息角
    if a.repose_deg is not None:
        angle = a.repose_deg
    else:
        tp = hou.parm(a.talus_parm) if a.talus_parm else None
        angle = tp.eval() if tp else None
    repose_slope = np.tan(np.radians(angle)) if angle is not None else None

    out = a.out if not a.tag else os.path.join(a.out, a.tag)
    os.makedirs(out, exist_ok=True)

    final_frame = a.final if a.final is not None else max(a.frames)
    conserve_layers = [s.strip() for s in a.conserve_layers.split(",") if s.strip()]

    base_h = height_array(base, a.height_layer)
    base_mass = layers_mass(base, conserve_layers)
    bsb, bsg, bep = erosion_score(base_h, cell)

    print(f"=== 侵蚀复查 ===")
    print(f"hip      : {a.hip}")
    print(f"result   : {a.result}")
    print(f"base     : {a.base}")
    print(f"cell={cell}  安息角={angle}°  repose slope tan={repose_slope}")
    print(f"守恒层: {'+'.join(conserve_layers)}")
    print(f"基底: s_bar={bsb:.4f} sigma={bsg:.4f} eps={bep:.4f}  mass={base_mass:.4f}\n")

    # --- 版本 A ---
    rowsA, hA = sweep(result, base_mass, a.height_layer, conserve_layers, cell, a.frames, final_frame)
    print(f"{'frame':>5} {'s_bar':>8} {'sigma':>8} {'eps':>8} {'mass':>15} {'守恒漂移':>12}")
    print(f"{'base':>5} {bsb:8.4f} {bsg:8.4f} {bep:8.4f} {base_mass:15.4f} {'-':>12}")
    for f in sorted(rowsA):
        sb,sg,ep,m,dr = rowsA[f]
        print(f"{f:5d} {sb:8.4f} {sg:8.4f} {ep:8.4f} {m:15.4f} {dr:12.2e}")
    max_drift = max(abs(rowsA[f][4]) for f in rowsA)
    print(f"\n守恒: 最大 |Δmass| = {max_drift:.3e}  (相对 {max_drift/abs(base_mass):.2e})")

    # --- 版本 B(可选) ---
    rowsB, hB = (None, None)
    resultB = hou.node(a.result_b) if a.result_b else None
    if resultB:
        rowsB, hB = sweep(resultB, base_mass, a.height_layer, conserve_layers, cell, a.frames, final_frame)

    labels = {kv.split("=")[0]: kv.split("=")[1] for kv in a.label if "=" in kv}
    lblA = labels.get("A", "A"); lblB = labels.get("B", "B")

    # ---------- 图1:评分轨迹 ----------
    def series(rows):
        fr = sorted(rows);
        return ([0]+fr,
                [bsb]+[rows[f][0] for f in fr],
                [bsg]+[rows[f][1] for f in fr],
                [bep]+[rows[f][2] for f in fr])
    frA,sbA,sgA,epA = series(rowsA)
    fig, ax1 = plt.subplots(figsize=(8.5,5))
    ax1.plot(frA, epA, 'o-', color='crimson', label=f'eps [{lblA}]')
    ax1.set_xlabel('iteration (frame)'); ax1.set_ylabel('epsilon = sigma/mean', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson'); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(frA, sbA, 's--', color='navy', alpha=.7, label=f'mean [{lblA}]')
    ax2.plot(frA, sgA, '^--', color='green', alpha=.7, label=f'std [{lblA}]')
    ax2.set_ylabel('mean / std of slope')
    if rowsB:
        frB,sbB,sgB,epB = series(rowsB)
        ax1.plot(frB, epB, 'o-', color='darkorange', label=f'eps [{lblB}]')
        ax2.plot(frB, sbB, 's:', color='royalblue', alpha=.6, label=f'mean [{lblB}]')
        ax2.plot(frB, sgB, '^:', color='limegreen', alpha=.6, label=f'std [{lblB}]')
    ttl = f'Erosion score vs iteration  (repose={angle}deg, cell={cell})'
    ax1.set_title(ttl)
    l1,a1 = ax1.get_legend_handles_labels(); l2,a2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, a1+a2, loc='best', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out,"score_trajectory.png"), dpi=110); plt.close(fig)

    # ---------- 图2:坡度直方图 ----------
    s0 = slope_olsen(base_h, cell).flatten()
    s1 = slope_olsen(hA, cell).flatten()
    fig, ax = plt.subplots(figsize=(8.5,5))
    hi = np.percentile(s0, 99.5); bins = np.linspace(0, hi, 80)
    ax.hist(s0, bins=bins, alpha=.55, color='steelblue', density=True, label='base')
    ax.hist(s1, bins=bins, alpha=.55, color='indianred', density=True, label=f'{lblA} @f{final_frame}')
    if rowsB:
        s1b = slope_olsen(hB, cell).flatten()
        ax.hist(s1b, bins=bins, alpha=.45, color='darkorange', density=True, label=f'{lblB} @f{final_frame}')
    if repose_slope is not None:
        ax.axvline(repose_slope, color='k', ls='--', lw=2, label=f'repose tan({angle}deg)={repose_slope:.2f}')
    ax.set_xlabel('slope = max|dh|/cell'); ax.set_ylabel('density')
    ax.set_title('Slope distribution: pile-up at the repose angle = talus')
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out,"slope_histogram.png"), dpi=110); plt.close(fig)

    # ---------- 图3:hillshade 对比 ----------
    ncol = 3 if not rowsB else 4
    fig, ax = plt.subplots(1, ncol, figsize=(5.4*ncol, 5.6))
    ax[0].imshow(hillshade(base_h, cell), cmap='gray', origin='lower'); ax[0].set_title('base')
    ax[1].imshow(hillshade(hA, cell), cmap='gray', origin='lower'); ax[1].set_title(f'{lblA} @f{final_frame}')
    diff = hA - base_h; vmax = np.percentile(np.abs(diff), 99.5)
    im = ax[2].imshow(diff, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    ax[2].set_title('height change (blue=deposit/red=erode)')
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    if rowsB:
        ax[3].imshow(hillshade(hB, cell), cmap='gray', origin='lower'); ax[3].set_title(f'{lblB} @f{final_frame}')
    for x in ax: x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out,"hillshade_compare.png"), dpi=110); plt.close(fig)

    # ---------- 图4(可选):额外标量场(如 water) ----------
    extra_png = None
    if a.extra_layer:
        hou.setFrame(final_frame)
        try:
            ef = height_array(result, a.extra_layer)
            fig, ax = plt.subplots(figsize=(6,5.6))
            im = ax.imshow(ef, cmap='Blues', origin='lower')
            ax.set_title(f'{a.extra_layer} @f{final_frame}  (max={ef.max():.3g})')
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            extra_png = f"{a.extra_layer}_field.png"
            fig.savefig(os.path.join(out, extra_png), dpi=110); plt.close(fig)
        except Exception as e:
            print(f"(extra-layer '{a.extra_layer}' 跳过: {e})")

    # ---------- 小结 ----------
    below = (slope_olsen(hA, cell) <= (repose_slope*1.05)).mean() if repose_slope is not None else float('nan')
    below0 = (slope_olsen(base_h, cell) <= (repose_slope*1.05)).mean() if repose_slope is not None else float('nan')
    print(f"安息角以下体素占比: 基底 {below0*100:.1f}% -> {lblA} {below*100:.1f}%")
    print(f"总搬运量 |Δh|/2 @f{final_frame} = {np.abs(diff).sum()/2:.1f}")
    print(f"\n输出目录: {out}")
    outs = ["score_trajectory.png","slope_histogram.png","hillshade_compare.png"]
    if extra_png: outs.append(extra_png)
    for fn in outs:
        print("  -", fn)
    print("=== 完毕(hip 未保存)===")

if __name__ == "__main__":
    main()
