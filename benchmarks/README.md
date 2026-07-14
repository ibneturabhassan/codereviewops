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
