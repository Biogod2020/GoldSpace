#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
TS="$(date +%Y%m%d_%H%M%S)"

echo "==> Repo root: ${REPO_ROOT}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1"; exit 1; }; }
need python3

# ---------- helper: python-based safe editor ----------
py_edit() {
python3 - "$@" <<'PYEDIT'
import sys, re, pathlib, io, os

def backup_write(path: pathlib.Path, new_text: str):
    bak = path.with_suffix(path.suffix + f".bak_{os.environ.get('TS','')}")
    if not bak.exists():
        bak.write_text(path.read_text())
    path.write_text(new_text)

def patch_params(pyfile: pathlib.Path):
    s = pyfile.read_text()
    if "--gnn-mode" in s and "--text-pooler" in s:
        print(f"[OK] {pyfile} already has --gnn-mode / --text-pooler")
        return
    # 插到 --freeze-omiclip 这一段 parser.add_argument(...) 之后
    m = re.search(r'parser\.add_argument\(\s*"--freeze-omiclip"[\s\S]*?\)\s*', s)
    if not m:
        print(f"[WARN] cannot find --freeze-omiclip block in {pyfile}, skip")
        return
    insert_after = m.end()
    add_block = '''
    parser.add_argument(
        "--gnn-mode",
        type=str,
        choices=["dual", "image_only", "text_only", "none"],
        default="dual",
        help="Which sides apply GNN: 'dual' (both), 'image_only', 'text_only', or 'none'."
    )
    parser.add_argument(
        "--text-pooler",
        type=str,
        choices=["mean", "center"],
        default="mean",
        help="When text GNN is disabled, 'center' picks the center-spot node embedding."
    )
'''
    new_s = s[:insert_after] + add_block + s[insert_after:]
    backup_write(pyfile, new_s)
    print(f"[PATCHED] {pyfile}: inserted --gnn-mode / --text-pooler")

def patch_spaglam_model(pyfile: pathlib.Path):
    s = pyfile.read_text()

    # 1) 替换投影头两行 -> 条件输入维度（img_proj_in / txt_proj_in）
    proj_pat = re.compile(
        r'self\.image_proj_head\s*=\s*MLPProjectionHead\([^\n]*\)\s*\n\s*'
        r'self\.gene_proj_head\s*=\s*MLPProjectionHead\([^\n]*\)'
    )
    proj_new = (
        'img_proj_in = gnn_hidden_dim if getattr(config, "gnn_mode", "dual") in ("dual", "image_only") else gnn_input_dim\n'
        '        txt_proj_in = gnn_hidden_dim if getattr(config, "gnn_mode", "dual") in ("dual", "text_only") else gnn_input_dim\n'
        '        self.image_proj_head = MLPProjectionHead(img_proj_in,  gnn_hidden_dim, gnn_output_dim)\n'
        '        self.gene_proj_head  = MLPProjectionHead(txt_proj_in,  gnn_hidden_dim, gnn_output_dim)'
    )
    s2, n_repl = proj_pat.subn(proj_new, s, count=1)
    if n_repl == 0:
        print(f"[WARN] {pyfile}: projection-head pattern not found; please check file version")
        s2 = s
    else:
        print(f"[PATCHED] {pyfile}: projection heads updated")

    # 2) 插入 _select_centers 方法（若不存在）
    if "_select_centers(" not in s2:
        # 在 def forward(...) 之前插入
        pos = s2.find('\n    def forward(')
        if pos == -1:
            print(f"[WARN] {pyfile}: cannot locate def forward(...); skip center selector insertion")
        else:
            center_method = """
    @staticmethod
    def _select_centers(x_nodes: torch.Tensor, batch) -> torch.Tensor:
        \"\"\"为每个子图选出一个中心节点的特征，返回 [G, C]。优先使用 batch.center_mask；否则用 ptr 或 batch 边界。\"\"\"
        N = x_nodes.size(0)
        device = x_nodes.device
        if hasattr(batch, "center_mask") and batch.center_mask is not None:
            mask = batch.center_mask
            assert mask.dtype == torch.bool and mask.numel() == N, "center_mask 必须是 [N] 的 bool"
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            return x_nodes[idx]
        if hasattr(batch, "ptr") and batch.ptr is not None:
            starts = batch.ptr[:-1]
            return x_nodes[starts]
        assert hasattr(batch, "batch"), "需要 batch.batch 或 ptr 才能定位子图"
        graph_ids: torch.Tensor = batch.batch
        starts = []
        prev = -1
        for i in range(N):
            g = int(graph_ids[i].item())
            if g != prev:
                starts.append(i); prev = g
        starts = torch.tensor(starts, device=device, dtype=torch.long)
        return x_nodes[starts]
"""
            s2 = s2[:pos] + center_method + s2[pos:]
            print(f"[PATCHED] {pyfile}: inserted _select_centers()")

    # 3) 替换 forward(...) 为 image_only+center 分支版本（保持接口不变）
    fstart = s2.find('def forward(self, batch:')
    if fstart == -1:
        print(f"[WARN] {pyfile}: cannot find forward(...); skip")
    else:
        # 取到下一个文件标记或函数边界（保守：到下一个文件标记）
        fend = s2.find('\n# ===== File:', fstart)
        if fend == -1:
            fend = len(s2)
        old_forward = s2[fstart:fend]
        # 直接覆盖整个 forward 定义
        new_forward = '''def forward(self, batch: "torch_geometric.data.Batch") -> dict:
        """
        The default forward pass for end-to-end training.
        这个函数接口和功能都保持不变。
        """
        # 1. 获取节点级别的 GNN 特征
        img_feat, gene_feat = self.forward_gnn(batch)

        # 2. 图级别读出 (Readout) 和 投影 (Projection)
        gnn_mode = getattr(self.config, "gnn_mode", "dual")
        text_pooler = getattr(self.config, "text_pooler", "mean")

        if gnn_mode == "image_only" and text_pooler == "center":
            # image：走 GNN + mean 池化
            Z_image = global_mean_pool(img_feat, batch.batch)

            # text：不进 GNN，直接取中心 spot（不池化）
            if self.use_precomputed_embeddings:
                E_txt_nodes = getattr(batch, "x_text", None)
                assert E_txt_nodes is not None, "需要 batch.x_text 以支持文本分支（precomputed 模式）"
            else:
                with torch.set_grad_enabled(not self.config.freeze_omiclip):
                    E_txt_nodes = self.omiclip_model.encode_text(batch.x_text)
            Z_gene = self._select_centers(E_txt_nodes, batch)
        else:
            # 其他模式保持原逻辑：双侧 mean 池化
            Z_image = global_mean_pool(img_feat, batch.batch)
            Z_gene = global_mean_pool(gene_feat, batch.batch)

        final_image_features = self.image_proj_head(Z_image)
        final_text_features  = self.gene_proj_head(Z_gene)

        # 3. 返回与训练代码兼容的输出
        return {
            "image_features": F.normalize(final_image_features, dim=-1),
            "text_features": F.normalize(final_text_features, dim=-1),
            "logit_scale": self.logit_scale.exp(),
        }
'''
        s2 = s2.replace(old_forward, new_forward)
        print(f"[PATCHED] {pyfile}: forward() replaced")

    backup_write(pyfile, s2)

def add_flags_to_common_args(shfile: pathlib.Path):
    s = shfile.read_text()
    if "--gnn-mode" in s and "--text-pooler" in s:
        print(f"[OK] {shfile}: flags already present")
        return
    key = 'COMMON_ARGS="'
    i = s.find(key)
    if i == -1:
        print(f"[WARN] {shfile}: cannot find COMMON_ARGS=...; skip")
        return
    # 找到该字符串的结束引号（处理内部有转义引号的情况）
    start = i + len('COMMON_ARGS=')
    j = start
    while j < len(s):
        if s[j] == '"' and s[j-1] != '\\':
            break
        j += 1
    if j >= len(s) or s[j] != '"':
        print(f"[WARN] {shfile}: unterminated COMMON_ARGS string; skip")
        return
    content = s[i+len(key):j]
    if "--gnn-mode" in content:
        print(f"[OK] {shfile}: flags already present inside COMMON_ARGS")
        return
    new_content = content.rstrip() + " \\\n  --gnn-mode            image_only \\\n  --text-pooler         center"
    new_s = s[:i+len(key)] + new_content + s[j:]
    backup_write(shfile, new_s)
    print(f"[PATCHED] {shfile}: appended --gnn-mode image_only --text-pooler center to COMMON_ARGS")

# ---------------- main ----------------
repo = pathlib.Path(sys.argv[1]).resolve()
os.environ['TS'] = sys.argv[2]

# 1) params.py
candidates_params = list(repo.glob("src/open_clip_train/params.py"))
if not candidates_params:
    # 兜底：全仓搜含有 --use-spaglam-model 的文件
    candidates_params = [p for p in repo.glob("**/*.py") if '--use-spaglam-model' in p.read_text(errors='ignore')]
for p in candidates_params:
    try:
        patch_params(p)
    except Exception as e:
        print(f"[ERROR] patching {p}: {e}")

# 2) spaglam_model.py
candidates_model = list(repo.glob("src/open_clip/spaglam_model.py"))
if not candidates_model:
    # 兜底：全仓搜 class SpaGLaM
    candidates_model = [p for p in repo.glob("**/*.py") if 'class SpaGLaM' in p.read_text(errors='ignore')]
for p in candidates_model:
    try:
        patch_spaglam_model(p)
    except Exception as e:
        print(f"[ERROR] patching {p}: {e}")

# 3) sequential scripts
for sh in repo.glob("**/sequentially_run_spaglam*.sh"):
    try:
        add_flags_to_common_args(sh)
    except Exception as e:
        print(f"[ERROR] patching {sh}: {e}")

print("\n[Done] Patching complete.")
PYEDIT
}

# ---------- run patchers ----------
export TS="${TS}"
py_edit "${REPO_ROOT}" "${TS}"

echo -e "\n==> Summary:"
echo "  - Added CLI flags: --gnn-mode, --text-pooler (params.py)"
echo "  - Updated SpaGLaM: projection heads (conditional in/out dims), _select_centers(), forward()"
echo "  - Appended training flags to sequential scripts (if found): --gnn-mode image_only --text-pooler center"
echo -e "\n✅ Now you can train as usual. If you run your sequential script, it should already include the new flags."

