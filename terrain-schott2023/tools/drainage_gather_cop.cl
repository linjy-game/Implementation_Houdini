// Schott 2023 汇水面积 — Copernicus COP OpenCL kernel 草稿 (未落地, 留作 GPU 港的起点)
// 写法参照 HF Erode v3.0 的 relax_erode: @KERNEL / @ixy / @layer.bufferIndex
// gather 不需 @WRITEBACK: COP 节点输入≠输出, 一个 opencl 节点 = 一次松弛; 迭代靠 copnet 里套 For-Each.

// ============ PASS 1: 每格下游坡度幂和 D (式4 分母) -> slope ============
#bind layer height float
#bind layer slope float        // 写 D
#bind parm  p float val=4

@KERNEL
{
    int2  off[8] = { (int2)(1,0),(int2)(-1,0),(int2)(0,1),(int2)(0,-1),
                     (int2)(1,1),(int2)(1,-1),(int2)(-1,1),(int2)(-1,-1) };
    float dst[8] = { 1.f,1.f,1.f,1.f, 1.41421356f,1.41421356f,1.41421356f,1.41421356f };
    float hi = @height, D = 0.f;
    for (int k = 0; k < 8; k++) {
        int2 n = @ixy + off[k];
        if (n.x<0 || n.x>=@xres || n.y<0 || n.y>=@yres) continue;
        float s = (hi - @height.bufferIndex(n)) / dst[k];   // 我->邻居
        if (s > 0.f) D += pow(s, p);                         // 只算更低邻居
    }
    @slope.set(D);
}

// ============ PASS 2: gather 汇水面积 a (式3/5) -> drain ============
// 单独一个 opencl 节点; @drain.bufferIndex 读输入(邻居旧a), @drain.set 写输出(自己新a)
#bind layer height float
#bind layer slope  float       // = D
#bind layer drain  float       // 读输入 + 写输出
#bind parm  p float val=4

@KERNEL
{
    int2  off[8] = { (int2)(1,0),(int2)(-1,0),(int2)(0,1),(int2)(0,-1),
                     (int2)(1,1),(int2)(1,-1),(int2)(-1,1),(int2)(-1,-1) };
    float dst[8] = { 1.f,1.f,1.f,1.f, 1.41421356f,1.41421356f,1.41421356f,1.41421356f };
    float hi = @height, a = 1.0f;                            // 式3 的 +1
    for (int k = 0; k < 8; k++) {
        int2 n = @ixy + off[k];
        if (n.x<0 || n.x>=@xres || n.y<0 || n.y>=@yres) continue;
        float s = (@height.bufferIndex(n) - hi) / dst[k];    // 邻居->我
        if (s > 0.f) {                                       // 只收更高邻居
            float Dj = @slope.bufferIndex(n);
            if (Dj > 0.f) a += (pow(s, p) / Dj) * @drain.bufferIndex(n);
        }
    }
    @drain.set(a);
}
