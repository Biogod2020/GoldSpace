TARGET="sequentially_run_spaglam_v4_robust.sh"   # ← 换成你的实际脚本名
cp "$TARGET" "${TARGET}.bak.$(date +%Y%m%d-%H%M%S)"  # 备份

awk '
BEGIN{inblk=0; done=0}
# 命中 EXPERIMENTS 块的开始
/^EXPERIMENTS=\(/{
  print "EXPERIMENTS=("
  print "    \"Exp6_TransformerConv_with-res;transformerconv;1e-4;true\""
  print ")"
  inblk=1; done=1; next
}
# 在原 EXPERIMENTS 块里：一直跳过，直到遇到唯一一行的右括号
inblk && /^\)/{ inblk=0; next }
inblk { next }

# 其他行原样输出
{ print }

END{
  if(!done){
    printf("ERROR: 未找到 EXPERIMENTS=() 定义，脚本未改动。\n") > "/dev/stderr"; exit 1
  }
}
' "$TARGET" > "$TARGET.tmp" && mv "$TARGET.tmp" "$TARGET" && chmod +x "$TARGET"

