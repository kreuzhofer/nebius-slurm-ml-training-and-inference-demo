Demo Day Follow-Up

Hi Roman,

I've completed the Demo Day assignment. The environment is intact and ready for the demo session. Here's a summary of what was accomplished and what I ran into along the way.


What Was Accomplished
---------------------

All four tasks are complete.

Task 1 — Distributed Training/Fine-Tuning via Soperator

I fine-tuned two models on the sql-create-context dataset (natural language to SQL, ~75K examples) using the Soperator-managed Slurm cluster:

- Qwen3-8B — full fine-tuning with PyTorch FSDP across all 16 H100 GPUs (2 nodes, 8 GPUs each). Trained for 1 epoch in ~13 minutes. Loss dropped from 2.60 to 0.41.
- Qwen3-32B — LoRA fine-tuning (rank 16, ~1% of parameters trainable) with FSDP across 16 GPUs. Full fine-tuning was not feasible for the 32B model due to OOM — optimizer states alone consumed nearly all available GPU memory even with FSDP sharding. LoRA eliminated that bottleneck entirely.

Training used torchrun for multi-process launch (one process per GPU) with srun coordinating across both Slurm nodes. Both jobs used the full 16-GPU allocation.

Task 2 — Inference on the Same Cluster

All three models are served via vLLM on the same cluster, exposing an OpenAI-compatible chat completions API:

- 8B model — served on 1 GPU (tensor parallelism = 1)
- 32B model — served on 4 GPUs (tensor parallelism = 4, as the ~64GB model exceeds single-GPU memory)
- 235B MoE model (Qwen3-235B-A22B) — served across both nodes (16 GPUs) using Ray for multi-node coordination, with tensor parallelism (TP=8) within each node and pipeline parallelism (PP=2) across nodes

The 235B model is a Mixture-of-Experts architecture (235B total parameters, 22B active per token). Its ~470GB weights require all 16 GPUs across both nodes. This uses a Slurm + Ray + vLLM stack: Slurm allocates the nodes, Ray provides the distributed runtime, and vLLM distributes the model across the Ray cluster. The model loads in about 14 minutes and serves an OpenAI-compatible API.

Example query against the 32B endpoint — "Find the names and email addresses of users who have spent more than $500 total on orders":

```sql
SELECT T1.name, T1.email FROM users AS T1 JOIN orders AS T2 ON T1.id = T2.user_id
GROUP BY T1.id HAVING SUM(T2.amount) > 500
```

Example query against the 235B endpoint — "What is the average price of products in each category?":

```sql
SELECT category, AVG(price) AS average_price FROM products GROUP BY category;
```

Task 3 — Base vs Fine-Tuned Comparison

Evaluated both base and fine-tuned models on 100 held-out test examples using exact-match SQL accuracy:

| Model              | Base | Fine-Tuned | Improvement |
|--------------------|------|------------|-------------|
| Qwen3-8B (full FT) | 2%   | 88%        | +86 pp      |
| Qwen3-32B (LoRA)   | 3%   | 84%        | +81 pp      |

The base models understand SQL syntax but fail on exact formatting (wrong quoting, casing, semicolons, verbose output). A single epoch of fine-tuning brings both models to 80%+ exact-match accuracy.

Task 4 — GPU Utilization >80%

Training jobs used 16/16 GPUs (100% allocation). The 8B training job sustained ~93.5 samples/second throughput across all 16 GPUs.


Challenges and How They Were Resolved
--------------------------------------

Several issues came up during development, all related to distributed training and inference at scale:

1. Dataset cache race condition — all 16 torchrun processes raced to create HuggingFace Arrow cache files on shared NFS, causing FileNotFoundError crashes. Fixed by disabling dataset caching and processing in-memory per rank.

2. FSDP model save timeout — post-training save_model() triggers an FSDP all-gather that exceeded the 300-second rendezvous timeout for the 8B model. Fixed by skipping the save and instead copying model files from the Trainer's last checkpoint on rank 0.

3. Slow checkpoint I/O on NFS — writing checkpoints to the shared filesystem added significant overhead. Fixed by writing checkpoints to local SSD (/mnt/local-data) and only copying the final model back to shared NFS. This reduced total job time by ~38%.

4. 32B full fine-tuning OOM — even with FSDP across 16x H100 80GB GPUs, AdamW optimizer states consumed too much memory for full fine-tuning of the 32B model. CPU offloading resolved the OOM but introduced an incompatibility between FSDP activation checkpointing and Qwen3's sliding window attention. Switched to LoRA, which trains only ~1% of parameters and eliminates the optimizer state bottleneck entirely.

5. Multi-node Ray cluster networking — three issues had to be resolved for the 235B multi-node inference. First, hostname -I returned a link-local address (169.254.x.x) as the first IP, which isn't routable between nodes — fixed by filtering out link-local addresses when resolving the Ray head IP. Second, multiple srun steps within the sbatch script exhausted Slurm task slots, preventing the vLLM server from launching — fixed by adding --overlap to all srun commands. Third, vLLM defaulted to the multiprocessing executor backend, which only sees local GPUs — fixed by explicitly setting --distributed-executor-backend ray so vLLM discovers all 16 GPUs across both nodes.


Terraform
---------

The Terraform code used for deployment is attached separately.


Environment Status
------------------

The lab environment is intact — models, datasets, fine-tuned outputs, and evaluation results are all in place on the shared filesystem at /mnt/data/demo/. The cluster is ready for the demo session.

Looking forward to the walkthrough.

Best,
Daniel
