# KAB-MoE

Static visual expert skipping for MoE-MLLMs: reward matrix search → policy solving → evaluation.  
Currently supports Kimi-VL.

## Environment

```bash
conda activate kab-moe
cd /path/to/KAB-MoE
export MODEL_PREFIX=/path/to/models   # directory containing models, e.g. Kimi-VL-A3B-Instruct
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
Dependencies are listed in requirements.txt.

KAB-MoE/
├── models/           # model hooks and static MoE logic
├── tasks/            # dataset transforms and task-specific preprocessing
├── eval/             # evaluation scripts or lmms_eval integration
├── utils.py          # shared utilities
├── search.py         # reward matrix search with budget-only sampling
├── solve_matrix.py   # solve final layer-wise policy from logged reward matrix
├── run.sh            # search script
├── eval_kimi.sh      # evaluation script
├── README.md
└── requirements.txt

Usage

1. Search reward matrix
Run policy search on the calibration set. The search samples fixed-budget policies, evaluates them against the full-expert baseline, and updates a layer-action reward matrix.
```bash
bash run.sh
```
2. Solve final policy
Extract the reward matrix from the search log and solve the final layer-wise policy under a target skipping ratio.
```bash
python solve_matrix.py \
  --log-file /path/to/log \
  --skip-ratio $target_ratio$
```
3. Evaluation
Use the solved pattern for downstream benchmark evaluation.
```bash
export VISUAL_K="" #match the pattern by solve_matrix.py
bash eval_kimi.sh $task_name$
```
