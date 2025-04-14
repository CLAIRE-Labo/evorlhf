# 🧪 Jupyter Notebook on CSCS Cluster (with Conda)

Follow these steps to run Jupyter Notebook on a compute node and access it from your local machine.

---

## 🖥️ Step 1: Request a Compute Node and SSH In

Use your cluster’s scheduler (e.g., `srun`, `salloc`, or `qsub`) to get an interactive shell on a compute node:

```bash
sbatch --time 4:00:00 -A a-a10 --gres=gpu:1 --wrap "sleep infinity" --output=/dev/null --error=/dev/null
```
or
```bash
sbatch --partition=debug --time=01:00:00 -A a-a10 --wrap "sleep infinity" --output=/dev/null --error=/dev/null
```

Once assigned, login the compute node like `<JOB-ID>`.

```bash
srun --overlap --pty --jobid=<JOB-ID> bash
```
---


## ✅ Step 2: Activate Your Conda Environment

```bash
conda activate your_env_name
```
---

## 🚀 Step 3: Start Jupyter Notebook Server

Run this **on the compute node**:

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

You’ll see output like:

```
http://0.0.0.0:8888/?token=abc123...
```

---

## 🔁 Step 4: Forward Port From Local Machine

In a separate terminal **on your local machine**, run:

```bash
ssh -L 8888:nid007024:8888 nevali@clariden
```

This tunnels the compute node’s Jupyter port to your local machine.

---

## 🌐 Step 5: Open Jupyter in Your Browser

Now open this in your browser:

```
http://localhost:8888
```

Paste the token URL you got in step 3 if prompted.

---

✅ You’re now running Jupyter Notebook on the cluster with access to GPU resources!
