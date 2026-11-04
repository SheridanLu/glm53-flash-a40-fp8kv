#!/bin/bash
# 只读干跑:把容器里正在运行的 vllm 的 .py 文件复制到临时目录,逐个检查补丁能否干净打上。不改容器、不改服务。
set -u
T=/data/glm_test/tree
rm -rf $T; mkdir -p $T
docker exec glm53flash sh -c "cd /usr/local/lib/python3.12/dist-packages && find vllm -name '*.py' | tar czf - -T -" | tar xzf - -C $T
echo "tree: $(find $T -name '*.py' | wc -l) py files"
cd $T
git init -q . 2>/dev/null; git add -A >/dev/null 2>&1; git -c user.email=a@b -c user.name=t commit -qm base >/dev/null 2>&1
P=/data/glm_test/patches
echo "=== 逐个单独检查(各自对原始树)"
for n in 56524 56525 37152 42359 43569 54076 55898 47123 56756 57477 51540 54368 57128; do
  out=$(git apply --check --exclude='tests/*' $P/$n.diff 2>&1); rc=$?
  nfail=$(echo "$out" | grep -c 'error:')
  echo "#$n rc=$rc  错误行数=$nfail  $(echo "$out" | grep 'error:' | head -2 | cut -c1-140 | tr '\n' ' ')"
done
echo "=== 备注: 若某补丁涉及的文件在本树不存在,会显示 No such file"
