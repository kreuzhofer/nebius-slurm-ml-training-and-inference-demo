# Qwen3 SQL Fine-Tuning: Demo Summary

## Objective

Fine-tune Qwen3-8B and Qwen3-32B on the `sql-create-context` dataset (natural language to SQL) using distributed training across 2 nodes with 16 H100 GPUs, then evaluate and serve the models.

**Fine-tuning** means taking a pre-trained language model (one that already understands language from training on massive text data) and continuing to train it on a smaller, task-specific dataset so it learns to perform that task well.

## Cluster Setup

- **Infrastructure**: Nebius Cloud, 2 worker nodes managed by **Slurm** (a cluster job scheduler — you submit jobs to it and it allocates nodes/GPUs, via Soperator)
- **GPUs**: 8x NVIDIA H100 80GB per node (16 total), connected via **InfiniBand** (a high-speed network fabric for GPU-to-GPU communication across nodes; **GPUDirect RDMA** lets GPUs transfer data directly over InfiniBand without going through the CPU)
- **Storage**: Shared **NFS** (Network File System — a networked disk visible to all nodes, `/mnt/data`, 2TB) for models/data + local **SSD** (`/mnt/local-data`, 1TB per node) for fast checkpoint I/O
- **Software**: PyTorch 2.x with **FSDP** (Fully Sharded Data Parallel — splits model weights, gradients, and optimizer states across GPUs so each GPU only holds a fraction), HuggingFace Transformers, **torchrun** (PyTorch's launcher for starting one training process per GPU across multiple nodes)

### Terraform Configuration (`terraform.tfvars`)

The cluster is provisioned using Terraform with **Soperator** (Nebius's Slurm-on-Kubernetes operator). Before deploying, the following changes were made to `terraform.tfvars`:

| Setting | Original Value | Changed To | Why |
|---|---|---|---|
| `production` | `true` | `false` | Setting `production = true` requires an `iam_merge_request_url` (an IAM approval workflow URL). Since this is a demo/non-production cluster, setting it to `false` bypasses that validation gate. |
| `active_checks_scope` | `""` (empty) | `"prod_quick"` | Enables GPU and InfiniBand health checks after provisioning. These run short benchmarks (all-reduce, IB bandwidth, CUDA tests) on each node to verify hardware is healthy before accepting workloads. Takes ~10 minutes on H100 nodes. |
| `backups_password` | `"password"` | (a real password) | The default placeholder is insecure. Changed to a proper encryption key for jail backup encryption. |

The worker nodeset `size` was changed from `128` (the template default) to `2` — we only need 2 nodes (16 GPUs total) for this demo. The rest of the worker config was already correct:

```hcl
slurm_nodeset_workers = [{
  name = "worker"
  size = 2                              # 2 worker nodes
  autoscaling = { enabled = false }     # fixed size, no scale-down
  resource = {
    platform = "gpu-h100-sxm"           # H100 SXM GPUs
    preset   = "8gpu-128vcpu-1600gb"    # 8 GPUs, 128 vCPUs, 1.6TB RAM per node
  }
  gpu_cluster = {
    infiniband_fabric = "fabric-2"      # InfiniBand network for cross-node GPU comms
  }
}]
```

Two shared filesystems were created manually in the Nebius Console before deploying Terraform, then attached to the cluster by referencing their IDs:

- **`filestore_jail`** — the root shared filesystem (the **jail** — the base filesystem visible to all Slurm nodes).
- **`filestore_jail_submounts`** — an additional shared filesystem mounted at `/mnt/data` inside the jail. This is where models, datasets, and outputs are stored.

Both are referenced by ID in the tfvars (`existing = { id = "computefilesystem-..." }`) rather than having Terraform create them, since they were already provisioned via the console.

Additionally, `node_local_jail_submounts` creates a 1TB local SSD (`/mnt/local-data`) on each worker node for fast checkpoint I/O. Other relevant settings: `slurm_shared_memory_size_gibibytes = 1024` (1 TiB shared memory per node) and `slurm_login_public_ip = true` (public IP on login nodes for SSH access).

### Deployed Virtual Machines

The Terraform template deploys several types of VMs (running as Kubernetes pods under Soperator), each with a specific role in the cluster:

| Node Type | Count | Platform / Preset | Role |
|---|---|---|---|
| **Worker** | 2 | gpu-h100-sxm / 8gpu-128vcpu-1600gb | The GPU compute nodes. All training and inference jobs run here. Each has 8x H100 80GB GPUs, 128 vCPUs, 1.6TB RAM, and is connected to the InfiniBand fabric. |
| **Login** | 2 | cpu-d3 / 32vcpu-128gb | SSH entry points to the cluster. You connect here to submit Slurm jobs (`sbatch`), check job status (`squeue`, `sacct`), and access the shared filesystem. Has a public IP. |
| **Controller** | 1 | cpu-d3 / 4vcpu-16gb | Runs the Slurm controller daemon (`slurmctld`). Manages the job queue, schedules jobs onto workers, and tracks node state. Users don't interact with it directly. |
| **NFS** | 1 | cpu-d3 / 32vcpu-128gb | Serves the NFS `/home` filesystem to all nodes. Separate from the Nebius filestores used for `/mnt/data`. |
| **Accounting** | 1 | cpu-d3 / 8vcpu-32gb | Runs the Slurm accounting database (`slurmdbd`). Records job history, resource usage, and enables `sacct` queries for completed jobs. |
| **System** | 3–9 | cpu-d3 / 8vcpu-32gb | Kubernetes system nodes managed by Soperator. Run cluster infrastructure (operators, monitoring, DCGM exporter, health checks). Auto-scaled based on cluster size. |

In total, the cluster runs ~10 VMs: 2 GPU workers doing the actual compute, and ~8 supporting nodes handling scheduling, storage, access, monitoring, and accounting.

### Deploying the Cluster

Deployment followed the [Soperator guide](https://github.com/nebius/nebius-solutions-library/tree/main/soperator). Prerequisites: Terraform, Nebius CLI (`nebius`), kubectl, jq, and coreutils.

**Step 1 — Create the installation directory** from the example template:

```bash
cd soperator
export INSTALLATION_NAME=dkreuzh001
mkdir -p installations/$INSTALLATION_NAME
cd installations/$INSTALLATION_NAME
cp -r ../example/* ../example/.* .
```

**Step 2 — Create shared filesystems** in the Nebius Console. The jail (root shared filesystem) and data filesystem (`/mnt/data`) were created manually and their IDs noted for use in `terraform.tfvars`.

**Step 3 — Configure and source `.envrc`**. The copied `.envrc` template needs the correct Nebius credentials before sourcing:

- `NEBIUS_TENANT_ID` — your Nebius tenant ID
- `NEBIUS_PROJECT_ID` — the project ID where the cluster will be deployed

Once these are set, sourcing the file handles the rest automatically:

1. Authenticating with Nebius IAM (`nebius iam get-access-token`)
2. Looking up the VPC subnet ID for the project
3. Creating (or reusing) a service account (`slurm-terraform-sa`) and adding it to the `editors` group
4. Generating a temporary S3 access key for Terraform state storage
5. Creating (or reusing) an S3 bucket for the Terraform backend and writing `terraform_backend_override.tf`
6. Exporting all required `TF_VAR_*` environment variables (region, IAM token, project ID, subnet ID, etc.)

```bash
source .envrc
nebius iam whoami   # verify authentication
```

**Step 4 — Configure `terraform.tfvars`** with cluster settings (see changes table above).

**Step 5 — Deploy**:

```bash
terraform init      # initialize providers and modules
terraform plan      # review what will be created
terraform apply     # deploy the cluster (~40 minutes)
```

`terraform apply` provisions the Kubernetes cluster, installs Soperator via Helm, and deploys all Slurm node sets. The GPU health checks (`active_checks_scope = "prod_quick"`) run automatically after workers come up.

**Step 6 — Verify** the Kubernetes cluster:

```bash
kubectl config get-contexts                          # confirm context was added
kubectl config use-context nebius-dkreuzh001-slurm   # switch to cluster context
kubectl get pods --all-namespaces                    # check all pods are running
kubectl get pods --all-namespaces -o wide            # also shows which node each pod runs on
```

### Connecting to the Cluster

After deployment, retrieve the login node's public IP and SSH in:

```bash
terraform output -json slurm_login_public_ips
ssh root@<login_node_ip> -i ~/.ssh/<private_key>
```

### Ranks in Distributed Training

Each GPU runs its own process, and every process is assigned a unique **rank** (0–15 across the job) and a **local rank** (0–7 within each node). These are set automatically by `torchrun`. All 16 ranks participate equally in training — **forward pass** (feeding data through the model to get a prediction), **backward pass** (computing how much each parameter contributed to the error, producing **gradients**), and **gradient synchronization** (averaging gradients across all GPUs so they agree on how to update the model) via FSDP. **Rank 0** (first GPU on the first node) acts as the leader for housekeeping tasks that should only happen once: logging, dataset preprocessing/caching, and saving the final model. It has no special role during the actual training computation.

---

## Chapter 1: Fine-Tuning Qwen3-8B

### Training Configuration

| Parameter | Value |
|---|---|
| Model | Qwen3-8B (**bfloat16** — a 16-bit number format that uses half the memory of standard 32-bit floats with minimal quality loss) |
| Dataset | sql-create-context (74,659 train / 3,930 eval) |
| Parallelism | FSDP full_shard auto_wrap across 16 GPUs |
| Batch size | 4 per GPU, 2 **gradient accumulation** steps (effective 128) — instead of updating the model after every 4 samples, we accumulate gradients over 2 mini-batches before updating, simulating a larger batch without needing more GPU memory |
| Learning rate | 2e-5, **cosine schedule** (learning rate starts high and gradually decreases following a cosine curve) with 3% **warmup** (learning rate ramps up slowly at the start to stabilize early training) |
| **Epochs** | 1 (584 steps) — one epoch means the model sees every training example exactly once |
| Training time | **798 seconds (~13 minutes)** |
| Throughput | 93.5 samples/second |

### Training Results

- **Initial loss**: 2.60 — **Final loss**: 0.41 (**loss** measures how wrong the model's predictions are — lower is better; it starts high because the model hasn't learned the task yet)
- **Eval loss**: 0.416 — measured on a held-out set of examples the model never trains on, to check it's genuinely learning rather than memorizing
- Loss dropped sharply in the first 50 steps, then gradually converged through the rest of the epoch

### Issues Discovered and Resolved

#### 1. Dataset Cache Race Condition

**Problem**: All 16 torchrun processes simultaneously called `train_test_split()` and `dataset.map()` on the shared NFS filesystem. The HuggingFace `datasets` library writes **Arrow cache files** (pre-processed data saved to disk so it doesn't need to be recomputed) during these operations, and 16 processes racing to create/read the same cache files caused `FileNotFoundError` crashes.

**Solution**: Disabled HuggingFace dataset caching entirely (`datasets.disable_caching()`). Each rank processes the dataset independently in memory. The dataset is small enough (~75K examples) that this adds negligible overhead, and it eliminates the cross-node race condition completely.

#### 2. FSDP Model Save Timeout

**Problem**: After training completes, calling `trainer.save_model()` triggers an FSDP **all-gather** (a collective operation where every GPU sends its shard of the model to rank 0, reconstructing the full model in one place). For an 8B model across 16 GPUs, this operation exceeded the torchrun **rendezvous timeout** (the deadline for all processes to complete a coordinated operation, 300 seconds), killing the job before the model could be saved.

**Solution**: Skip `save_model()` entirely. The HuggingFace Trainer already saves a full **checkpoint** (a snapshot of the model weights saved to disk, in **safetensors** format — a safe, fast file format for storing model weights) during training at `save_steps` intervals. After training, rank 0 copies the model files from the last checkpoint to the shared output directory, filtering out unnecessary files (**optimizer states** — extra data the optimizer needs to track training momentum, FSDP shards, RNG states).

#### 3. Slow Checkpoint I/O on Shared Storage

**Problem**: Writing checkpoints to shared NFS (`/mnt/data`) during training was slow due to network filesystem overhead, adding significant time to each checkpoint save.

**Solution**: Configured the Trainer to write checkpoints to local SSD (`/mnt/local-data`) on each worker node. Only the final model is copied back to shared NFS after training. This reduced total job time by approximately 38% compared to writing checkpoints directly to NFS.

---

## Chapter 2: Evaluating Qwen3-8B

### Evaluation Results (100 test examples, exact-match accuracy)

| Model | Accuracy |
|---|---|
| Base Qwen3-8B | **2%** (2/100) |
| Fine-tuned Qwen3-8B | **88%** (88/100) |
| **Improvement** | **+86 percentage points** |

The base model understands SQL syntax but fails on exact formatting (wrong quoting, casing, trailing semicolons, verbose output). The fine-tuned model learns the precise output format and produces exact matches on 88% of test queries after just one epoch.

**Exact-match accuracy** means the model's SQL output must be character-for-character identical to the expected answer (after normalizing whitespace, casing, and trailing semicolons). This is a strict metric — a query that returns the correct result but uses different formatting counts as wrong.

---

## Chapter 3: Serving Qwen3-8B (Inference)

The fine-tuned model is served using **vLLM** (a high-performance inference engine optimized for serving large language models) which exposes an **OpenAI-compatible API** — meaning you can query it with the same format used for ChatGPT/OpenAI APIs.

```bash
sbatch /mnt/data/demo/scripts/serve.sbatch
```

This starts an HTTP server on a worker node. To query it:

```bash
curl http://<worker_node>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b-sql", "messages": [{"role": "user", "content": "Schema: CREATE TABLE t (id INT, name TEXT)\nQuestion: How many rows?"}]}'
```

The 8B model needs 1 GPU for inference (**tensor parallelism** = 1 — the entire model fits on a single GPU).

---

## Chapter 4: Fine-Tuning Qwen3-32B (LoRA)

### The Problem: Full Fine-Tuning Exceeds GPU Memory

Qwen3-32B full fine-tuning failed with **OOM** (out-of-memory) errors on 16x H100 80GB GPUs, despite FSDP sharding the model across all of them. Here's why:

FSDP distributes the model **parameters** (the learned weights of the neural network), **gradients** (how much to adjust each parameter, computed during the backward pass), and **optimizer states** across GPUs. For a 32B parameter model in bfloat16:

| Component | Per-GPU Memory (16 GPUs) | Explanation |
|---|---|---|
| Model parameters (sharded) | ~4 GB | The 32B weights split across 16 GPUs |
| Gradients (sharded) | ~4 GB | One gradient per parameter, also split |
| Optimizer states (**AdamW**) | **~16–20 GB** | AdamW (the most common optimizer for LLMs) maintains two extra float32 copies per parameter: **momentum** (running average of recent gradients) and **variance** (running average of squared gradients), used to make smarter updates |
| **Activations** | ~20–30 GB | Intermediate values computed during the forward pass that must be kept in memory for the backward pass |
| FSDP all-gather buffers | ~10–15 GB | During forward/backward, FSDP temporarily reconstructs full layers by gathering shards from all GPUs — these buffers hold the reassembled layer |
| CUDA allocator overhead | ~5–8 GB | GPU memory **fragmentation** — like a hard drive with scattered free blocks, the GPU may have 5GB free total but not in one contiguous chunk |
| **Total** | **~60–77 GB** | |

This leaves almost no headroom on 80GB GPUs. During FSDP initialization, a single 932MB allocation request fails because the remaining memory is too fragmented. The dominant consumer is **optimizer states** — AdamW's two float32 copies total ~32GB even after sharding.

We attempted **CPU offloading** (`fsdp="full_shard offload auto_wrap"` — moving optimizer states from GPU to CPU RAM, which is much larger) to free ~16–20GB per GPU. This resolved the OOM but introduced a secondary failure: FSDP's **activation checkpointing** (a technique that saves memory by not storing all activations during the forward pass, instead recomputing them during the backward pass — trades compute time for memory) is incompatible with Qwen3's **SDPA** (Scaled Dot-Product Attention — PyTorch's built-in efficient attention implementation) because Qwen3 uses **sliding window attention** (a variant where each token only attends to a fixed window of nearby tokens, not the entire sequence) with varying key-value lengths that cause shape mismatches during recomputation. **Flash Attention 2** (a faster, more memory-efficient attention implementation that handles causal masking internally) would avoid this issue but was not installed in the environment.

### The Solution: LoRA (Low-Rank Adaptation)

Instead of updating all 32 billion parameters, **LoRA** freezes the entire base model and injects small trainable **adapter** matrices into each layer. These adapters use a **low-rank decomposition**: for a weight matrix W of size (d × d), LoRA adds a bypass path B × A where B is (d × r) and A is (r × d), with **rank** r much smaller than d (e.g., r=16 vs d=4096). Only B and A are trained — this captures the important adjustments the model needs for the new task using a tiny fraction of the original parameters.

Think of it like this: instead of rewriting an entire textbook (full fine-tuning), you write a small set of sticky notes that modify key pages (LoRA adapters). The textbook stays unchanged, and the sticky notes are cheap to create and store.

**Why this solves the memory problem:**

| Component | Full Fine-Tuning | LoRA (r=16) |
|---|---|---|
| Trainable parameters | 32B (100%) | ~300M (~1%) |
| Optimizer states | ~32 GB total | ~2.4 GB total |
| Gradients | ~4 GB/GPU | negligible |
| Base model (frozen, sharded) | ~4 GB/GPU | ~4 GB/GPU |

With LoRA, the optimizer only tracks the ~1% of parameters that are actually trained. This eliminates the dominant memory consumer entirely, freeing ~20+ GB per GPU.

### 32B LoRA Training Configuration

| Parameter | Value |
|---|---|
| Model | Qwen3-32B (bfloat16, frozen) |
| LoRA rank (r) | 16 |
| **LoRA alpha** | 32 — a scaling factor that controls how much influence the adapters have; typically set to 2× the rank |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj — these are the linear layers inside each **transformer block** (the repeating building block of the model); q/k/v/o_proj are the **attention** layers (how the model decides which parts of the input to focus on), and up/down/gate_proj are the **MLP** layers (feed-forward networks that process each position independently) |
| Trainable parameters | ~1% of total |
| Batch size | 2 per GPU, 4 gradient accumulation steps (effective 128) |
| Learning rate | 2e-4 (10x higher than full FT — LoRA adapters are randomly initialized and small, so they need larger updates to learn quickly) |
| Parallelism | FSDP full_shard auto_wrap with `use_orig_params: True` — this flag tells FSDP to preserve the original parameter structure, which **PEFT** (Parameter-Efficient Fine-Tuning, the HuggingFace library that implements LoRA) requires to correctly identify and update only the adapter parameters |

### Post-Training: Merging Adapters

LoRA training outputs only the small adapter weights (~200MB). To use the model with existing evaluation and serving scripts, the adapters are merged back into the base model — the adapter matrices are mathematically combined with the original weight matrices, producing a standard model that behaves as if it were fully fine-tuned:

```bash
srun --nodes=1 --gpus-per-node=1 --exclusive \
    python /mnt/data/demo/scripts/merge_lora.py \
    /mnt/data/demo/output/qwen3-32b-sql-lora \
    /mnt/data/demo/models/Qwen3-32B \
    /mnt/data/demo/output/qwen3-32b-sql
```

This produces a standard model directory that works identically with `evaluate.py` and `serve.sbatch`. The single GPU is only needed to get a node with enough CPU RAM (~64GB) — the merge itself runs on CPU.

---

## Chapter 5: Evaluating Qwen3-32B

### Evaluation Results (100 test examples, exact-match accuracy)

| Model | Accuracy |
|---|---|
| Base Qwen3-32B | **3%** (3/100) |
| Fine-tuned Qwen3-32B (LoRA) | **84%** (84/100) |
| **Improvement** | **+81 percentage points** |

The base 32B model exhibits the same failure mode as the base 8B — it understands SQL but produces outputs with wrong quoting, casing, trailing semicolons, and verbose formatting that fail exact-match. After LoRA fine-tuning with only ~1% of parameters trained, the 32B model achieves 84% accuracy, comparable to the 8B model's 88% with full fine-tuning.

The slightly lower accuracy compared to the 8B model is expected: LoRA trains a small fraction of the model's parameters, so it has less capacity to adapt to the exact output format. Despite this, the 32B LoRA model handles more complex queries better — for example, correctly mapping ambiguous column references and generating multi-table JOINs that the 8B model sometimes misses.

```bash
sbatch --gpus-per-node=4 /mnt/data/demo/scripts/evaluate.sbatch \
    /mnt/data/demo/models/Qwen3-32B /mnt/data/demo/output/qwen3-32b-sql
```

---

## Chapter 6: Serving Qwen3-32B (Inference)

The fine-tuned 32B model is served using vLLM with **tensor parallelism** across 4 GPUs (splitting each layer's weight matrices across GPUs so they compute in parallel — unlike FSDP which is for training, tensor parallelism is used at inference time):

```bash
sbatch --gpus-per-node=4 /mnt/data/demo/scripts/serve.sbatch \
    /mnt/data/demo/output/qwen3-32b-sql 8000 4
```

Once the server is running, query it from the login node:

```bash
curl http://<worker_node>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/data/demo/output/qwen3-32b-sql",
    "messages": [
      {"role": "system", "content": "You are a SQL expert. Given a database schema and a question, write the correct SQL query. Output only the SQL query, nothing else."},
      {"role": "user", "content": "Schema: CREATE TABLE users (id INT, name TEXT, age INT, email TEXT)\nCREATE TABLE orders (id INT, user_id INT, product TEXT, amount DECIMAL, order_date DATE)\n\nQuestion: Find the names and email addresses of users who have spent more than $500 total on orders"}
    ]
  }'
```

Example response from the fine-tuned 32B model:

```sql
SELECT T1.name, T1.email FROM users AS T1 JOIN orders AS T2 ON T1.id = T2.user_id GROUP BY T1.id HAVING SUM(T2.amount) > 500
```

The model correctly generates a multi-table JOIN with GROUP BY and HAVING — a non-trivial query that demonstrates genuine SQL reasoning, not just pattern matching. The 32B model needs 4 GPUs with tensor parallelism (TP=4) because the model weights (~64GB in bfloat16) exceed a single H100's 80GB memory when accounting for KV cache and runtime overhead.

---

## Key Takeaways

1. **Distributed training on shared filesystems requires explicit rank coordination** for dataset preprocessing. Never assume cache files will be safely shared across processes without barriers.
2. **FSDP all-gather is unreliable at scale** for post-training saves. Copying from Trainer checkpoints is a practical workaround that avoids the collective operation entirely.
3. **Local SSD for intermediate I/O** is a significant performance win on NFS-backed clusters. Use shared storage only for inputs and final outputs.
4. **One epoch is sufficient** for task-specific fine-tuning on a well-structured dataset. The 8B model went from 2% to 88% accuracy in under 14 minutes of training.
5. **Full fine-tuning has hard memory limits** — optimizer states dominate GPU memory and can make training infeasible even with FSDP sharding across many GPUs.
6. **LoRA is the practical solution for large models** — by training only ~1% of parameters, it eliminates the optimizer state bottleneck and makes 32B training feasible on the same hardware with larger batch sizes.
7. **Multi-node inference requires three fixes over the naive setup** — filtering link-local IPs from `hostname -I`, adding `--overlap` to srun commands to avoid task slot exhaustion, and explicitly setting `--distributed-executor-backend ray` for vLLM to discover cross-node GPUs.
8. **TP within nodes + PP across nodes minimizes cross-node traffic** — keeping the expensive all-reduce on NVLink and only sending pipeline stage outputs over InfiniBand is the optimal parallelism split for multi-node serving.
9. **MoE models are practical at scale** — Qwen3-235B-A22B (235B total, 22B active) loads in ~14 minutes across 16 H100s and generates correct SQL without any fine-tuning, demonstrating the power of large pre-trained models.

---

## Appendix A: Slurm for Training vs. Slurm for Inference

### What Slurm Actually Does

Slurm is a **job scheduler** — it manages who gets to use which machines and GPUs. Think of it as a receptionist at a hotel: you request "I need 2 rooms with 8 beds each," and Slurm finds available rooms, gives you the keys, and kicks you out when your time is up. It doesn't care *what* you do inside the rooms.

This means Slurm's role is identical for training and inference: allocate nodes, allocate GPUs, run your script, clean up when done. The difference is what runs *inside* the Slurm job.

### The Three-Layer Stack

Both training and inference on a GPU cluster have three layers:

```
┌─────────────────────────────────────────────────┐
│  Layer 1: RESOURCE ALLOCATION (Slurm)           │
│  "Which nodes and GPUs do I get?"               │
│  Same for training and inference.               │
│  sbatch → srun                                  │
├─────────────────────────────────────────────────┤
│  Layer 2: DISTRIBUTED RUNTIME                   │
│  "How do the GPUs coordinate with each other?"  │
│  DIFFERENT for training vs inference.            │
│  Training: torchrun + PyTorch                   │
│  Inference: Ray + vLLM                          │
├─────────────────────────────────────────────────┤
│  Layer 3: THE ACTUAL WORK                       │
│  Training: forward pass, backward pass, update  │
│  Inference: receive request, generate tokens    │
└─────────────────────────────────────────────────┘
```

Layer 1 (Slurm) is always the same. The interesting difference is Layer 2.

### Training: Slurm → torchrun → PyTorch

For distributed training, the software stack is:

1. **Slurm** (`sbatch`/`srun`) allocates nodes and launches one process per node
2. **torchrun** (PyTorch's built-in launcher) spawns 8 GPU processes on each node
3. **PyTorch FSDP** handles the distributed training logic — sharding model weights, synchronizing gradients across all 16 GPUs via NCCL over InfiniBand

The key: **PyTorch has its own distributed runtime built in** (`torch.distributed`). It uses NCCL (NVIDIA's GPU communication library) to move data between GPUs. It doesn't need any external framework — torchrun sets up the coordination, and PyTorch handles everything from there.

```bash
# Training launch: Slurm → torchrun → PyTorch
srun torchrun --nnodes=2 --nproc_per_node=8 train.py
```

Each of the 16 processes is equal — they all do the same work (forward pass, backward pass, gradient sync) on different data. The communication pattern is **symmetric**: every GPU talks to every other GPU in the same way.

### Inference: Slurm → Ray → vLLM

For distributed inference, the software stack is:

1. **Slurm** (`sbatch`/`srun`) allocates nodes — same as training
2. **Ray** (a separate distributed computing framework) creates a cluster across the nodes
3. **vLLM** connects to the Ray cluster and distributes model layers across all GPUs

The key: **vLLM does not use PyTorch's built-in distributed runtime**. Instead, it uses Ray because inference has fundamentally different coordination needs:

- An inference server must handle **incoming HTTP requests** and route them to GPU workers
- It must manage a **KV cache** (memory that stores previously computed attention values so the model doesn't recompute them for every new token) across multiple GPUs
- It must do **dynamic batching** (grouping multiple requests together to process them more efficiently, even when they arrive at different times)
- Workers are **asymmetric**: there's a scheduler that decides which GPU handles which request, unlike training where every GPU does the same thing

Ray provides the infrastructure for this kind of coordination. PyTorch's `torch.distributed` was designed for training's symmetric all-reduce patterns, not for inference's request-routing patterns.

```bash
# Inference launch: Slurm → Ray → vLLM
ray start --head           # on node 1
ray start --address=...    # on node 2
python -m vllm ... --tensor-parallel-size 8 --pipeline-parallel-size 2
```

### Side-by-Side Comparison

| | Training | Inference |
|---|---|---|
| **Slurm's role** | Allocate 2 nodes × 8 GPUs | Allocate 2 nodes × 8 GPUs |
| **Distributed runtime** | torchrun + PyTorch (`torch.distributed`) | Ray |
| **GPU communication** | NCCL (via PyTorch) | NCCL (via Ray/vLLM) |
| **Process model** | 16 equal workers, all doing the same thing on different data | 1 scheduler + 16 GPU workers, handling different requests |
| **Communication pattern** | Symmetric — all-reduce, all-gather (every GPU sends to every GPU) | Asymmetric — scheduler routes requests, GPUs do tensor/pipeline parallel |
| **Why not just use Slurm?** | You *can* — torchrun is just a launcher, not a framework. `srun` alone could work with the right env vars. | You *can't* — vLLM needs Ray's actor model for worker management, KV cache coordination, and request scheduling. Slurm can't provide that. |
| **Extra dependency** | None beyond PyTorch | Ray (must be installed and started as a cluster) |

### The Short Version

**Slurm is the bouncer** — it decides who gets in (which nodes/GPUs your job gets). **The distributed runtime is the crew inside** — for training, PyTorch's built-in crew (torchrun + FSDP) handles everything. For inference, vLLM brings its own crew (Ray) because serving requests is a fundamentally different job than training a model. You always need the bouncer (Slurm), but the crew changes depending on the task.

---

## Chapter 7: Serving Qwen3-235B-A22B (Multi-Node Inference)

### Why Multi-Node Is Required

Qwen3-235B-A22B is a **Mixture-of-Experts (MoE)** model — 235 billion total parameters with 22 billion active per token. In bfloat16, the weights alone are ~470GB, far exceeding a single node's 8x H100 80GB (640GB total, minus KV cache and runtime overhead). Multi-node inference across both nodes (16 GPUs total) is mandatory.

### Architecture: Slurm + Ray + vLLM

vLLM uses Ray to coordinate GPU workers across machines. The serving architecture has three layers:

1. **Ray cluster** — one node runs `ray start --head`, the others join with `ray start --address=<head_ip>:6379`. This gives vLLM a unified pool of GPUs across nodes.
2. **Tensor parallelism (TP=8)** — splits each layer's weight matrices across 8 GPUs within each node. Every GPU computes a slice of every layer in parallel, then they **all-reduce** (combine partial results from all GPUs into a final result) to merge. Uses NVLink within the node for fast GPU-to-GPU communication.
3. **Pipeline parallelism (PP=2)** — splits the model by layers across the 2 nodes. Each node handles half the layers. Cross-node traffic occurs only between pipeline stages (not every layer), using InfiniBand with GPUDirect RDMA.

This TP=8/PP=2 split is optimal — it keeps the expensive all-reduce within each node (NVLink, ~900 GB/s) and only sends pipeline stage outputs across nodes (InfiniBand, ~400 Gb/s).

### Running It

```bash
sbatch /mnt/data/demo/scripts/serve_235b.sbatch
```

### Issues Discovered and Resolved

#### 1. Link-Local IP Address in Ray Cluster Setup

**Problem**: The `hostname -I` command returns all IP addresses for a node. On this cluster, the first IP returned is `169.254.0.1` — a **link-local** address that is not routable between nodes. The original script used `awk '{print $1}'` to grab the first IP, causing worker nodes to fail to connect to the Ray head with `Failed to connect to GCS at address 169.254.0.1:6379`.

**Solution**: Filter out link-local addresses before selecting the IP:
```bash
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | tr ' ' '\n' | grep -v '^169\.254\.' | head -n1)
```

#### 2. Slurm Task Slot Exhaustion

**Problem**: The sbatch script uses three `srun` commands (Ray head, Ray worker, vLLM server), but only allocates `--ntasks-per-node=1` (2 total task slots for 2 nodes). The two backgrounded Ray `srun` commands consume both slots, causing the third `srun` (vLLM) to block indefinitely waiting for a free slot. The model never starts loading.

**Solution**: Add `--overlap` to all `srun` commands, which allows multiple srun steps to share the same task slots within the job:
```bash
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" ray start --head ...
srun --overlap --nodes=1 --ntasks=1 -w "$NODE" ray start --address=...
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" python -m vllm ...
```

#### 3. Missing `--distributed-executor-backend ray`

**Problem**: vLLM 0.16.0 defaults to the `mp` (multiprocessing) executor backend, which only sees GPUs on the local node. With TP=8 and PP=2 (world size 16), it detects only 8 local GPUs and raises a `ValidationError`: "World size (16) is larger than the number of available GPUs (8)."

**Solution**: Explicitly set `--distributed-executor-backend ray` so vLLM uses the Ray cluster to discover GPUs across all nodes.

### Startup Timeline

| Phase | Duration | Description |
|---|---|---|
| Ray head start | ~7s | Ray runtime initializes on worker-0 |
| Worker join | ~4s | worker-1 connects to the Ray cluster |
| vLLM initialization | ~20s | Resolves model architecture, creates Ray placement groups |
| Model weight loading | **~14 minutes** | Loads 118 safetensors shards (~470GB) from NFS across 16 GPUs |
| CUDA graph capture | ~30s | Captures optimized decode graphs for various batch sizes |
| **Total startup** | **~16 minutes** | From job start to server ready |

### Inference Results

The base Qwen3-235B-A22B model (not fine-tuned) correctly generates SQL queries using its built-in reasoning:

```
Q: What is the average price of products in each category?
A: SELECT category, AVG(price) AS average_price FROM products GROUP BY category;

Q: Which department has the most employees?
A: SELECT department FROM employees GROUP BY department ORDER BY COUNT(*) DESC LIMIT 1;
```

The model uses Qwen3's **thinking mode** (`<think>` tags) by default, providing step-by-step reasoning before producing the final SQL. This is useful for complex queries but consumes tokens from the context window.

### Key Configuration Notes

- **`--max-model-len 32768`** — The model supports large context windows. With 16x H100 80GB GPUs, there is ample KV cache headroom for 32K token sequences.
- **`--distributed-executor-backend ray`** — Required for multi-node inference; without this, vLLM only sees local GPUs.
- **`--overlap` on srun** — Required when running multiple srun steps within a single sbatch job to avoid task slot exhaustion.
- The Soperator health checks (`all-redu`, `ib-gpu-p`, `cuda-sam`) run periodically in the `hidden` partition. These short benchmark jobs test GPU all-reduce, InfiniBand, and CUDA functionality. They may delay job scheduling by a few minutes while they cycle through the nodes.

---

## Appendix A: Slurm for Training vs. Slurm for Inference

### What Slurm Actually Does

Slurm is a **job scheduler** — it manages who gets to use which machines and GPUs. Think of it as a receptionist at a hotel: you request "I need 2 rooms with 8 beds each," and Slurm finds available rooms, gives you the keys, and kicks you out when your time is up. It doesn't care *what* you do inside the rooms.

This means Slurm's role is identical for training and inference: allocate nodes, allocate GPUs, run your script, clean up when done. The difference is what runs *inside* the Slurm job.

### The Three-Layer Stack

Both training and inference on a GPU cluster have three layers:

```
┌─────────────────────────────────────────────────┐
│  Layer 1: RESOURCE ALLOCATION (Slurm)           │
│  "Which nodes and GPUs do I get?"               │
│  Same for training and inference.               │
│  sbatch → srun                                  │
├─────────────────────────────────────────────────┤
│  Layer 2: DISTRIBUTED RUNTIME                   │
│  "How do the GPUs coordinate with each other?"  │
│  DIFFERENT for training vs inference.            │
│  Training: torchrun + PyTorch                   │
│  Inference: Ray + vLLM                          │
├─────────────────────────────────────────────────┤
│  Layer 3: THE ACTUAL WORK                       │
│  Training: forward pass, backward pass, update  │
│  Inference: receive request, generate tokens    │
└─────────────────────────────────────────────────┘
```

Layer 1 (Slurm) is always the same. The interesting difference is Layer 2.

### Training: Slurm → torchrun → PyTorch

For distributed training, the software stack is:

1. **Slurm** (`sbatch`/`srun`) allocates nodes and launches one process per node
2. **torchrun** (PyTorch's built-in launcher) spawns 8 GPU processes on each node
3. **PyTorch FSDP** handles the distributed training logic — sharding model weights, synchronizing gradients across all 16 GPUs via NCCL over InfiniBand

The key: **PyTorch has its own distributed runtime built in** (`torch.distributed`). It uses NCCL (NVIDIA's GPU communication library) to move data between GPUs. It doesn't need any external framework — torchrun sets up the coordination, and PyTorch handles everything from there.

```bash
# Training launch: Slurm → torchrun → PyTorch
srun torchrun --nnodes=2 --nproc_per_node=8 train.py
```

Each of the 16 processes is equal — they all do the same work (forward pass, backward pass, gradient sync) on different data. The communication pattern is **symmetric**: every GPU talks to every other GPU in the same way.

### Inference: Slurm → Ray → vLLM

For distributed inference, the software stack is:

1. **Slurm** (`sbatch`/`srun`) allocates nodes — same as training
2. **Ray** (a separate distributed computing framework) creates a cluster across the nodes
3. **vLLM** connects to the Ray cluster and distributes model layers across all GPUs

The key: **vLLM does not use PyTorch's built-in distributed runtime**. Instead, it uses Ray because inference has fundamentally different coordination needs:

- An inference server must handle **incoming HTTP requests** and route them to GPU workers
- It must manage a **KV cache** (memory that stores previously computed attention values so the model doesn't recompute them for every new token) across multiple GPUs
- It must do **dynamic batching** (grouping multiple requests together to process them more efficiently, even when they arrive at different times)
- Workers are **asymmetric**: there's a scheduler that decides which GPU handles which request, unlike training where every GPU does the same thing

Ray provides the infrastructure for this kind of coordination. PyTorch's `torch.distributed` was designed for training's symmetric all-reduce patterns, not for inference's request-routing patterns.

```bash
# Inference launch: Slurm → Ray → vLLM
ray start --head           # on node 1
ray start --address=...    # on node 2
python -m vllm ... --tensor-parallel-size 8 --pipeline-parallel-size 2
```

### Side-by-Side Comparison

| | Training | Inference |
|---|---|---|
| **Slurm's role** | Allocate 2 nodes × 8 GPUs | Allocate 2 nodes × 8 GPUs |
| **Distributed runtime** | torchrun + PyTorch (`torch.distributed`) | Ray |
| **GPU communication** | NCCL (via PyTorch) | NCCL (via Ray/vLLM) |
| **Process model** | 16 equal workers, all doing the same thing on different data | 1 scheduler + 16 GPU workers, handling different requests |
| **Communication pattern** | Symmetric — all-reduce, all-gather (every GPU sends to every GPU) | Asymmetric — scheduler routes requests, GPUs do tensor/pipeline parallel |
| **Why not just use Slurm?** | You *can* — torchrun is just a launcher, not a framework. `srun` alone could work with the right env vars. | You *can't* — vLLM needs Ray's actor model for worker management, KV cache coordination, and request scheduling. Slurm can't provide that. |
| **Extra dependency** | None beyond PyTorch | Ray (must be installed and started as a cluster) |

### The Short Version

**Slurm is the bouncer** — it decides who gets in (which nodes/GPUs your job gets). **The distributed runtime is the crew inside** — for training, PyTorch's built-in crew (torchrun + FSDP) handles everything. For inference, vLLM brings its own crew (Ray) because serving requests is a fundamentally different job than training a model. You always need the bouncer (Slurm), but the crew changes depending on the task.
