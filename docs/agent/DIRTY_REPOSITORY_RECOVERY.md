# Dirty Repository Recovery and Safe Isolation

Use this procedure whenever the working tree is not clean, the ownership of changes is uncertain, or governance work cannot be applied safely in place. The purpose is to preserve all work, establish evidence, and continue in an isolated location without asking the human to review code or choose routine Git commands.

This procedure does not authorize discarding, rewriting, publishing, or merging any change.

## Required outcome

Before changing repository files, produce a durable recovery record that answers:

- What repository, branch, commit, and remotes were observed?
- Which files are staged, unstaged, untracked, ignored-but-relevant, conflicted, or submodule-modified?
- Can every observed change be recovered independently of the current checkout?
- Which base commit and isolation route will be used for the approved task?
- Why can the task proceed without modifying or concealing the existing work?

## Fail-closed rules

- Do not use `git reset`, `git checkout --`, `git restore`, `git clean`, destructive rebases, force operations, or deletion of unknown files.
- Do not drop, pop, or overwrite a stash. Do not use a stash as the only recovery copy.
- Do not commit unknown work merely to make the tree clean.
- Do not amend or rewrite commits that may belong to another task.
- Do not push, merge, open a pull request, or publish a patch unless the exact action is authorized.
- Do not copy credentials, private keys, tokens, large raw datasets, or protected outcomes into the recovery bundle. Record excluded paths and reasons.
- Do not inspect protected strategy outcomes while classifying repository state. Path, size, status, and hash metadata are sufficient unless the approved task explicitly requires more.

## Phase 1: Identify the repository and instruction chain

Record, using repository-supported read-only commands:

- resolved repository root;
- current branch or detached-HEAD state;
- current commit and one-line subject;
- configured remotes without credential-bearing URLs;
- linked worktrees and their branches;
- applicable `AGENTS.md` files from root to the task paths;
- submodule status, sparse-checkout state, and large-file tooling when present.

Redact credentials from any remote URL before archiving output. If repository identity cannot be established, stop before writes and record the exact failure.

## Phase 2: Inventory the dirty state

Capture machine-readable and human-readable inventories of:

- porcelain status with branch data;
- staged name/status and binary summary;
- unstaged name/status and binary summary;
- untracked paths with sizes and SHA-256 where safe;
- merge conflicts and unmerged stages;
- changed submodules;
- relevant ignored files that appear to be task artifacts rather than caches;
- large or sensitive paths excluded from content capture.

Classify each item as `staged`, `unstaged`, `untracked`, `conflicted`, `submodule`, `generated`, `sensitive_excluded`, or `unknown`. Do not infer authorship from timestamps alone.

## Phase 3: Create a recovery bundle

Use the repository's existing archive convention. If none is verified, create a new task-local directory such as:

```text
docs/agent/task_archive/<YYYYMMDD>_dirty-repository-recovery/
```

Preserve, as applicable:

- `REPOSITORY_STATE.md` with root, branch, commit, remotes, worktrees, and instruction chain;
- `STATUS_PORCELAIN_V2.txt`;
- `STAGED.patch` produced from the index;
- `UNSTAGED.patch` produced from the working tree;
- an untracked-file manifest with relative path, size, SHA-256, and capture decision;
- safe copies of small untracked text/config files when they contain no secrets or protected outcomes;
- a binary/large-file inventory and recovery location instead of embedding large payloads;
- `RECOVERY_MANIFEST.json` with path, size, SHA-256, purpose, and exclusion reason;
- `RECOVERY_VALIDATION.md` showing that patch parsing and manifest hashes passed.

Where supported, add a Git bundle or another independent object-level recovery artifact. A patch alone does not preserve untracked files, file modes in every case, submodule content, ignored files, or Git LFS objects. Use more than one recovery mechanism when the state is complex.

Validate the recovery bundle before relying on it. Do not alter the original checkout to test restoration.

## Phase 4: Choose a safe isolation route

Prefer a new worktree and task branch from a verified commit that does not modify the dirty checkout. The base must come from repository evidence, such as the current `HEAD`, a verified task branch, or an authorized remote-tracking commit. Record the base commit exactly.

Use this decision order:

1. If the approved task is documentation-only and the intended paths do not exist at the verified base, create a new isolated worktree from that base and apply only the approved package there.
2. If intended paths already exist, compare them in the isolated worktree with the package and integrate surgically. Preserve repository-native content and instruction precedence.
3. If the existing dirty changes touch the same paths, do not overlay them. Keep the dirty checkout intact, integrate against the verified base in isolation, and record the overlap for later reconciliation by an agent review.
4. If no safe base can be proven, stop before writes and report the missing evidence. Do not ask the human to interpret the diff; ask only for the missing authority or repository identity needed to proceed.

The isolated worktree must have its own branch name and task archive. Never point two concurrent agents at the same writable worktree.

## Phase 5: Integrate and review

In the isolated worktree:

- read the complete applicable instruction chain;
- write and maintain the task plan;
- apply only the approved documentation, instruction, skill, or test scope;
- discover commands from repository files and CI;
- run non-economic documentation, skill, lint, and focused test checks supported by the repository;
- inspect the final diff for unrelated changes, secrets, protected-period leakage, broken links, and authority conflicts;
- obtain a separate agent review when available;
- update the task archive, manifest, repository authority, and continuity records.

Do not run a backtest, economic screen, protected-outcome inspection, capture action, live action, push, or merge unless separately authorized.

## Human role and escalation

The human should receive:

- a short statement of what was preserved;
- the safe isolation route and base commit;
- material risks or unresolved authority conflicts;
- trading or research sanity-check questions;
- any consequential approval that is actually required;
- links or paths to the recovery and task archives.

Do not ask the human to decide which hunks are correct, reconcile routine documentation, maintain the logs, or choose a Git command. Escalate only a decision that cannot be resolved from repository evidence and would materially change, destroy, or publish work.

## Completion record

Report:

- recovery bundle path and SHA-256 manifest;
- original checkout left unchanged: yes/no;
- repository root, branch, and commit;
- staged, unstaged, untracked, conflict, and excluded counts;
- isolated worktree path, branch, and base commit;
- package files integrated or superseded;
- validation commands and results;
- archive and Drive handoff paths and hashes;
- economic runs launched: yes/no;
- protected outcomes inspected: yes/no;
- pushes or merges: yes/no;
- remaining blockers and exact approval required.
