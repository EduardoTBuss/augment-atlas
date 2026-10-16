# Recovering a stopped worker attempt

A stopped training container can outlive its host worker's last queue update.
In that case the Git queue may still show `running`. Reconcile the original
attempt before considering an explicit training resume.

The host API `augment_atlas.runtime.reconcile_stopped_attempt(settings,
active_path, reason=...)` performs this recovery without starting training or
running a CUDA probe. The operator must first stop the original host worker,
confirm the container is stopped, and exclude concurrent access to its journal
(for example with the existing GPU worker lock). Preserve copies of the journal
and spool before a recovery operation.

The API checks the journal's attempt, worker owner, runtime provenance, stopped
container image, labels, environment and output mount. It uses the original
attempt/token and versioned queue events. An unavailable Git transport leaves a
durable event for a later call; replay keeps the same event ID. Ownership or
version conflicts require inspection. Recovery never claims another attempt or
requests a training retry.

Exit code zero with validated completion artifacts establishes `succeeded`.
An interrupted exit (75, 130, 137 or 143) with a compatible checkpoint archive,
no OOM and no trainer failure marker permits `interrupted`. Other exits,
trainer failure markers and missing or invalid artifacts are reported as
`failed`. The checkpoint check uses ZIP integrity and the job's saved
configuration/split/progress provenance. It does **not** deserialize tensors or
establish model/optimizer resumability. Before an explicit resume, also load the
trusted private checkpoint in the pinned training environment on CPU and check
its internal configuration, split, completed epoch, model, optimizer, scheduler
and RNG state. Keep checkpoints, raw logs and local journals private.

The `configs/smoke/recovery-v1/synthetic/none.json` fixture is a separate
infrastructure job for bounded recovery tests. Its identity must remain stable
through checkpoint/resume and live-container reattachment. It is synthetic,
provisional, and provides no clinical generalization evidence. A new code or
configuration identity is a new technical attempt, not proof of resume.

Temperature ceilings and preventive triggers belong to the operator's test
manifest. A preventive thermal emergency should stop GPU work promptly using
the last committed atomic checkpoint when necessary; waiting for an epoch to
finish can consume the temperature margin. Software monitoring is not a physical
temperature guarantee. Do not change fans, power limits or system settings as
part of these recovery tests.
