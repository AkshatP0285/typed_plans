#!/usr/bin/env bash
set -u
cd /fsxvision_new/aryanjain.intern/typed_plans
export TAU2_DATA_DIR=/fsxvision_new/aryanjain.intern/tau2-bench-src/data
export VLLM_4B_URL=http://10.20.213.80:8002/v1
export HF_HOME=/fsxvision_new/aryanjain.intern/.cache/huggingface
PY=/fsxvision_new/aryanjain.intern/tau2_venv/bin/python
OUT=outputs/tau2
DOMAINS=${DOMAINS:-"airline retail"}
SET=${SET:-formats}

run() {
  local domain=$1 tag=$2 fmt=$3 post=$4 port=$5 extra=${6:-}
  local base=http://10.20.213.80:$port/v1
  local user_args='{"api_base":"'$base'","api_key":"x","temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
  VLLM_27B_URL=$base $PY scripts/run_tau2.py benchmark=tau2 benchmark.domain=$domain benchmark.num_tasks=null \
    benchmark.agent_llm_label=qwen3-27b "benchmark.user_llm=openai/Qwen/Qwen3.8-27B" \
    "benchmark.extra_args=[--user-llm-args,'$user_args',--max-concurrency,6,--communicate-scorer,substring${TAU2_EXTRA:-},--save-to,$PWD/$OUT/${domain}_$tag.json]" \
    planner=prompted planner.few_shot=false plan_format=$fmt gate=jev gate.post_steps=$post \
    ir_editor=llm judge=llm model@models.planner=qwen3_27b_vllm model@models.editor=qwen3_27b_vllm \
    model@models.judge=qwen3_27b_vllm model@models.converter=qwen3_4b_vllm model@models.responder=qwen3_27b_vllm \
    $extra hydra.run.dir=$OUT/${domain}_$tag > $OUT/${domain}_$tag.log 2>&1
  echo "$domain $tag rc=$?"
}

mkdir -p $OUT
if [ "$SET" = fix ]; then
  TAU2_EXTRA=",--task-ids,$FIX_IDS"
  for t in $FIX_TAGS; do
    case $t in ir) run $DOMAINS ir_fix typed_ir all 8004 ;; ir_danger) run $DOMAINS ir_danger_fix typed_ir danger 8006 ;; graph) run $DOMAINS graph_fix graph all 8006 ;; free_text) run $DOMAINS free_text_fix free_text_rule all 8004 ;; esac &
  done
  wait
  echo ALL_DONE
  exit 0
fi
if [ "$SET" = irtext ]; then
  run airline ir_irtext typed_ir all 8004 gate.plan_view=ir_text &
  run retail ir_irtext typed_ir all 8005 gate.plan_view=ir_text &
  wait
  echo ALL_DONE
  exit 0
fi
for domain in $DOMAINS; do
  P3a=8004; P3b=8005
  if curl -s -m3 http://10.20.213.80:8006/v1/models | grep -q Qwen; then P3a=8006; P3b=8006; fi
  run $domain ir typed_ir all 8004 &
  run $domain ir_danger typed_ir danger $P3a &
  run $domain graph graph all 8005 &
  run $domain free_text free_text_rule all $P3b &
  wait
done
echo ALL_DONE
