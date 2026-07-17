# Synthetic benchmark data

The source directory is the human-authored source of truth for the canonical synthetic
benchmark suite. Every case contains a versioned case.json plus complete before and
after trees. Golden line numbers are never authored directly; generation derives them
from unique added-line anchors.

The cases and their code are original synthetic material created for this repository.
They use only Python's standard library and contain no copied third-party project code,
credentials, personal information, customer data, environment files, or production
artifacts.

Generated files live under benchmarks/tasks: task manifests at the root, diffs in
fixtures, exact post-change workspaces in workspaces, deterministic review responses
in replays, and suite manifests in suites.

Regenerate and validate with:

    codereviewops benchmark generate --source benchmarks/source --output-root benchmarks/tasks
    codereviewops benchmark generate --source benchmarks/source --output-root benchmarks/tasks --check
    codereviewops benchmark validate --suite benchmarks/tasks/suites/m4_25.json

The canonical comparison matrix is matrices/m4_replay_transport_v1.json. It runs the
full suite through replay using direct and MCP stdio transports, gates every quality and
semantic-trace metric at the strict profile, and compares against
baselines/m4_replay_v1.json. Replay establishes deterministic transport parity; it is
not evidence of prompt-quality differences. Baselines contain only stable semantic data
and hashes, excluding timestamps, host paths, process identifiers, latency, and tokens.

Run the strict regression harness after building the pinned Docker test image:

    codereviewops benchmark run --matrix benchmarks/matrices/m4_replay_transport_v1.json --output-dir artifacts/m4_replay