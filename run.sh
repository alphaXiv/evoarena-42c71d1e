#!/usr/bin/env bash
###############################################################################
# EvoArena / EvoMem — minimal proof-of-concept (PersonaMem-Evo subset)
#
# Illustrates the paper's core claim on its smallest, self-contained subset:
# a patch-based memory (EvoMem) tracks how a user's preferences *evolve* across
# a long implicit chat history and beats a "consolidate-to-latest" robust
# baseline that suffers from "state collapse".
#
# We run BOTH agents — the robust baseline (test_persona_robust.py) and the
# EvoMem patch agent (test_persona_patch.py) — over the committed compact
# benchmark (data/personamem-evo-10p.csv), restricted to ONE persona (the
# smallest chat history, 113 messages) so the whole thing finishes on a CPU
# box in minutes. Both agents share the same A-Mem base memory and the same
# hosted LLM (OpenRouter); the only difference is EvoMem's patch trace.
#
# Output: EVAL.md (+ the two result CSVs and metric files) under
# .openresearch/artifacts/ , with the head-to-head MCQ + chain accuracy.
#
# This is API-based + CPU-embedding work: NO GPU needed. The LLM runs behind
# OpenRouter; the only local compute is the all-MiniLM-L6-v2 retriever.
###############################################################################
set -euo pipefail

# ---- knobs (overridable from the environment) -------------------------------
MODEL="${MODEL:-openai/gpt-4o-mini}"      # real, cheap OpenRouter model
BACKEND="${BACKEND:-openrouter}"
PERSONA_IDS="${PERSONA_IDS:-18}"          # smallest chat history in the 10p split
MAX_ITEMS="${MAX_ITEMS:-}"                # empty = all questions for the persona
RETRIEVE_K="${RETRIEVE_K:-10}"
PATCH_TOP_K="${PATCH_TOP_K:-3}"
MIN_PATCH_SIMILARITY="${MIN_PATCH_SIMILARITY:-0.4}"
BATCH="${BATCH:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBDIR="$REPO_ROOT/EvoMem-PersonaMem-Evo"
ART="$REPO_ROOT/.openresearch/artifacts"
mkdir -p "$ART"

echo "=============================================================="
echo "EvoMem PersonaMem-Evo PoC"
echo "  model=$MODEL backend=$BACKEND persona=$PERSONA_IDS max_items=${MAX_ITEMS:-ALL}"
echo "  retrieve_k=$RETRIEVE_K patch_top_k=$PATCH_TOP_K min_patch_sim=$MIN_PATCH_SIMILARITY"
echo "=============================================================="

# ---- API key ----------------------------------------------------------------
# litellm/openai read these from the environment. Accept the common names and
# normalize so both robust (openai client) and patch (litellm openrouter) work.
KEY="${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-${API_KEY:-}}}"
if [[ -z "$KEY" ]]; then
  echo "FATAL: no API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY)." >&2
  exit 3
fi
export OPENROUTER_API_KEY="$KEY"
export OPENAI_API_KEY="$KEY"
export API_KEY="$KEY"

# ---- environment ------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export LITELLM_LOG="${LITELLM_LOG:-WARNING}"
PY="${PYTHON:-python3}"

# ---- dependencies (install once) --------------------------------------------
if ! "$PY" -c "import sentence_transformers, litellm, openai, rank_bm25, nltk" >/dev/null 2>&1; then
  echo ">> installing dependencies (CPU torch)…"
  "$PY" -m pip install --quiet --upgrade pip
  # CPU-only torch keeps the image small and avoids CUDA wheels on a CPU box.
  "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install --quiet -r "$SUBDIR/requirements.txt"
fi
"$PY" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

# ---- run both agents --------------------------------------------------------
cd "$SUBDIR"
export MODEL_LIST="$MODEL" BACKEND PERSONA_IDS RETRIEVE_K PATCH_TOP_K MIN_PATCH_SIMILARITY BATCH
export BENCHMARK_FILE="data/personamem-evo-10p.csv"
export API_BASE="https://openrouter.ai/api/v1"
export OUTPUT_DIR="results"
export RESUME=1 FORCE_REINGEST_PATCHES=0
[[ -n "$MAX_ITEMS" ]] && export MAX_ITEMS

echo ">> running robust baseline + EvoMem patch (this calls the hosted LLM)…"
# Use the two-stage gated patch policy (PATCH_GATING_PROMPT →
# PATCH_DETAIL_REVISION_PROMPT) in PersonaPatchAgent._answer_with_patch_policy
# instead of unconditionally injecting the top-3 retrieved patches. This tests
# whether gating cuts the off-topic-patch regressions called out in
# patch_prompts.py while also reducing prompt tokens.
export PATCH_USAGE="${PATCH_USAGE:-gated}"
bash scripts/run_persona_baseline_patch.sh both

# ---- locate the two result CSVs ---------------------------------------------
ROBUST_CSV="$(ls -t results/persona_robust_*.csv 2>/dev/null | head -1 || true)"
PATCH_CSV="$(ls -t results/persona_patch_*.csv 2>/dev/null | head -1 || true)"
echo "robust csv: $ROBUST_CSV"
echo "patch  csv: $PATCH_CSV"
if [[ -z "$ROBUST_CSV" || -z "$PATCH_CSV" ]]; then
  echo "FATAL: missing result CSV(s)." >&2
  exit 4
fi

# ---- chain accuracy ---------------------------------------------------------
"$PY" scripts/evaluate_persona_chain_acc.py "$ROBUST_CSV" "$PATCH_CSV" \
  --size 32k --output results/chain_acc_summary.csv || true

# ---- copy artifacts + build EVAL.md -----------------------------------------
cp -f "$ROBUST_CSV" "$PATCH_CSV" "$ART"/ 2>/dev/null || true
cp -f "${ROBUST_CSV%.csv}_metrics.txt" "${PATCH_CSV%.csv}_metrics.txt" "$ART"/ 2>/dev/null || true
cp -f results/chain_acc_summary.csv "$ART"/ 2>/dev/null || true

"$PY" - "$ROBUST_CSV" "$PATCH_CSV" "$MODEL" "$PERSONA_IDS" <<'PY' > "$ART/EVAL.md"
import csv, sys
csv.field_size_limit(sys.maxsize)
robust_csv, patch_csv, model, persona = sys.argv[1:5]

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def mcq_acc(rows):
    col = "is_correct_mcq_32k"
    ok = sum(1 for r in rows if r.get(col) == "True")
    n  = sum(1 for r in rows if r.get(col) in ("True", "False"))
    return ok, n, (ok / n if n else 0.0)

def chain_acc(rows):
    col = "is_correct_mcq_32k"
    groups = {}
    for r in rows:
        cid = (r.get("chain_id") or "").strip()
        if not cid or r.get(col) not in ("True", "False"):
            continue
        groups.setdefault(cid, []).append(r.get(col) == "True")
    if not groups:
        return 0, 0, 0.0
    good = sum(1 for v in groups.values() if all(v))
    return good, len(groups), good / len(groups)

rb, pb = load(robust_csv), load(patch_csv)
ro, rn, ra = mcq_acc(rb)
po, pn, pa = mcq_acc(pb)
rcg, rcn, rca = chain_acc(rb)
pcg, pcn, pca = chain_acc(pb)

print(f"# EvoMem PersonaMem-Evo — minimal PoC\n")
print(f"- model: `{model}` (OpenRouter) · persona: {persona} · subset: data/personamem-evo-10p.csv\n")
print("## MCQ step accuracy (head-to-head)\n")
print("| agent | correct / answered | accuracy |")
print("|---|---|---|")
print(f"| robust baseline | {ro} / {rn} | {ra:.3f} |")
print(f"| EvoMem (patch)  | {po} / {pn} | {pa:.3f} |")
print(f"\n**MCQ delta (EvoMem − baseline): {pa-ra:+.3f}**\n")
print("## Chain (exact-match) accuracy\n")
print("| agent | chains solved / total | accuracy |")
print("|---|---|---|")
print(f"| robust baseline | {rcg} / {rcn} | {rca:.3f} |")
print(f"| EvoMem (patch)  | {pcg} / {pcn} | {pca:.3f} |")
print(f"\n**Chain delta (EvoMem − baseline): {pca-rca:+.3f}**\n")
verdict = "EvoMem >= baseline" if pa >= ra else "EvoMem < baseline (small-N noise possible)"
print(f"_Verdict: {verdict}._")
PY

echo "=============================================================="
cat "$ART/EVAL.md"
echo "=============================================================="
echo "DONE. Artifacts in $ART"
