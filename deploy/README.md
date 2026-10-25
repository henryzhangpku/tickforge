# Deploying a distributed sweep

One container image, one sharding contract: every task runs
`tickforge_cli sweep ... --shard-from-env`, which reads its slice of the
global task list from whichever scheduler started it:

| Scheduler | Index variable | Count variable |
|---|---|---|
| any (explicit) | `TICKFORGE_SHARD` | `TICKFORGE_SHARDS` |
| Kubernetes Indexed Job | `JOB_COMPLETION_INDEX` | `TICKFORGE_SHARDS` |
| Google Cloud Run Jobs | `CLOUD_RUN_TASK_INDEX` | `CLOUD_RUN_TASK_COUNT` |
| AWS Batch array job | `AWS_BATCH_JOB_ARRAY_INDEX` | `TICKFORGE_SHARDS` |

Task *i* belongs to shard *i mod N*. Results are deterministic, so the merged
output of N shards is byte-identical to a single-machine run.

* **Image:** `ghcr.io/henryzhangpku/tickforge:latest`, published by CI on every push to `main`.
* **Kubernetes** (`k8s/sweep-job.yaml`): deployed to a throwaway kind cluster in CI on every push;
  CI merges the pod logs and diffs them against a local run.
* **Cloud Run Jobs** (`cloudrun/job.yaml`) and **AWS Batch** (`aws-batch/job-definition.json`):
  ready-to-apply definitions with the commands in their headers. They are not run by CI, which has
  no cloud account; they rely on the same image and the same contract the Kubernetes job verifies.

AWS Batch:

```sh
aws batch register-job-definition --cli-input-json file://deploy/aws-batch/job-definition.json
aws batch submit-job --job-name tickforge-sweep --job-queue YOUR_FARGATE_QUEUE \
  --job-definition tickforge-sweep --array-properties size=4
```

Merging shard output from any scheduler:

```sh
python3 scripts/merge_shards.py shard-*.log > sweep.csv
```
