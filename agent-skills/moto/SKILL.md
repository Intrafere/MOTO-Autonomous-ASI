---
name: moto
description: Control MOTO Autonomous ASI (autonomous math/science research, paper writing, and Lean 4 proofs) from the terminal with the `moto` command - hand it a hard question or decision to think about in depth and check back for its answer (`moto think`), launch it, set up providers, start or stop research runs, wait for results, read papers, proofs, and final answers, and follow live activity.
metadata: {"openclaw":{"requires":{"anyBins":["moto","moto-asi"]}}}
---

# MOTO command line

MOTO is a local desktop app that runs long autonomous research workflows. The
`moto` command drives a running MOTO instance on this computer. If `moto` is not
found, use `moto-asi` (same program; installed when another tool already owns
the name `moto`).

## Rules for agents

- Add `--json` to every command and parse stdout. Errors are also JSON, with
  `error`, `exit_code`, and `detail`.
- Exit codes: `0` ok, `1` error (including `wait` timeouts), `2` usage problem,
  `3` MOTO not running, `4` MOTO rejected the request (read `detail`, for example
  "another workflow is running" or a missing setting).
- Commands never prompt. Destructive commands need `--yes`.
- Never put API keys on the command line. Pipe them on stdin or use `--from-env VAR`.
- Only one workflow runs at a time: Autonomous Research (`autonomous`),
  Aggregator (`aggregator`), Single Paper Writer (`compiler`), or the LeanOJ
  Proof Solver (`leanoj`). Stop the active one before starting another.
- Runs are long (hours or days) and continue until stopped. Starting returns
  immediately; use `moto wait` or `moto events` instead of polling in a loop.
- Stopping keeps progress. Starting the same workflow again resumes it.
- After any failure, run `moto status --json` and read `attention`: it explains
  why a run stopped and what fixes it.

## First-time setup

```sh
moto status --json                       # exit 3 means MOTO is not running
moto launch --no-browser --json          # takes a few minutes the first time
moto keys set openrouter --from-env OPENROUTER_API_KEY --json
moto keys status --json                  # what is connected
moto vllm set http://127.0.0.1:8001 --json   # use a vLLM server; `moto vllm status` lists detected models
```

Subscription sign-in (OpenAI Codex, xAI Grok) needs the user's browser:
`moto login codex --no-wait --json` returns `authorization_url` and `state`.
Give the link to the user, then run `moto login codex --wait-only --json`. If
their browser ends on an error page, ask for that page's address and run
`moto login codex --callback-url URL --state STATE --json`.

## Deep thinking: hand MOTO a hard question

Use `moto think` when a decision, design, or idea deserves far more thought than
one reply can give it. MOTO explores the question in depth (many parallel
attempts, each checked by a separate validator, building on the ideas that
survive), then makes one model call that answers your question from what it
explored. You carry on with other work and check back.

```sh
moto think start --prompt-file question.txt --max-runtime 2h --json
moto think wait --timeout 600 --json   # or `moto think status --json` now and then
moto think answer --json               # read answer.content
```

- Write the question so it stands alone: the goal, the constraints, what is
  already known, and what a useful answer looks like. MOTO sees only that text.
- MOTO always ends with an answer. It writes it when the exploration finishes
  on its own, when `--max-runtime` is reached, or when you run
  `moto think finish` (stop exploring and answer now from what it has).
- Expect a long run that spends the user's provider usage the whole time. Set
  `--max-runtime` (`45m`, `2h`, `1h30m`, `1d`; at least 60 seconds) unless the
  user wants it open-ended. MOTO enforces the limit itself, so it holds even if
  you never check back. The answer is written just after the limit.
- `state` is one of: `thinking`, `answering` (writing the answer), `answered`,
  `needs_attention` (stopped on a problem such as a provider error or no
  credits; read `attention`), `stopped`, `idle`. `next` says what to do.
- `moto think answer`: when `answer_ready` is true, `answer.content` is MOTO's
  answer, in whatever form the question called for. `answer.reason` is
  `exploration_complete`, `time_limit`, or `finish_requested`. `--details` adds
  any papers and verified proofs behind it.
- `moto think wait --timeout SECONDS` blocks until MOTO stops. It exits 0 either
  way; `timed_out: true` means it is still working.
- By default MOTO only explores and answers. `--proofs` also searches for
  Lean-verified proofs (needs Lean; much more usage) and `--papers` also writes
  research papers first (much longer). Add them only when the user wants them.
- `moto think resume [--max-runtime 1h]` explores further and answers again
  (when `can_resume` is true). `moto think stop` stops at once with no answer;
  prefer `finish`.
- Exit 4 with `detail.reason`: `workflow_running` (one workflow at a time),
  `paused_session_exists` (a different question is paused; `detail` shows it),
  or `not_a_think_run`. Pass `--replace` only when that paused question is your
  own earlier one or the user agreed: it stays in history but can no longer be
  resumed.
- It reuses the model settings saved for Autonomous Research (on a fresh
  install add `--model`, see below) and never changes them.

## Starting workflows

`moto <mode> start` reuses the settings from the last start of that workflow.
On a fresh install, choose a model:

```sh
moto autonomous start --model openai/gpt-5 --prompt-file prompt.txt --json
```

- `--model` uses one model for every role. `--provider` is `openrouter`
  (default), `lm_studio`, `vllm`, `codex`, `grok`, `kimi-code`, `sakana-fugu`,
  `claude-api`, or `amazon-bedrock`. Context and output limits are filled in for
  OpenRouter and vLLM models; for the others pass `--context` and `--max-tokens`. `--submitters N` (1-10, default 3).
- `--set key=value` changes one field (value parsed as JSON). Dotted paths reach
  nested fields: `--set validator_model=other/model`,
  `--set submitter_configs.*.model_id=other/model`.
- `moto config init <mode> --model ... -o settings.json` writes the settings to a
  file to edit; start with `--config settings.json`. `moto config show <mode>`
  prints the last-used settings.
- `--prompt TEXT` or `--prompt-file FILE` (`-` reads stdin). Prefer
  `--prompt-file` for long prompts or prompts with quotes, `%`, `&`, or `|`.
- `--proofs/--no-proofs`, `--papers/--no-papers` (autonomous, compiler): Lean
  proof output and research paper output. At least one must stay on. Proofs need
  Lean (`moto proofs status`, `moto proofs settings --lean on`; `--single-shot`/`--master on|off` pick proof types, `--parallel N` sets concurrent attempts).
- Aggregator: `--file FILE` uploads reference files (.txt/.lean) for the run.
- LeanOJ: `--lean-template FILE` is required.
- Autonomous: `--max-runtime 2h` stops the run after that long (never
  remembered for later starts); `--final-answer/--no-final-answer` turns the
  final answer stage on or off.
- `--dry-run` prints the request without starting.

## Waiting and watching

| Goal | Command |
|------|---------|
| Until the running workflow stops | `moto wait --json` |
| Until N papers exist | `moto wait --papers N --json` |
| Until the Tier 3 final answer exists | `moto wait --final-answer --json` |
| Until a proof check ends | `moto wait --proof-run ID --json` |
| Until a live event arrives | `moto wait --event TYPE --json` |
| Stream live activity | `moto events --json --limit 50` (`--type TYPE`, `--timeout S`) |

`wait` takes `--timeout SECONDS` (exit 1 when it expires). If the run stops
before the goal, `wait` exits 4 with the stop reason. `events` reconnects
automatically if the stream drops; events sent while disconnected are missed.

## Reading results

| Goal | Command |
|------|---------|
| Brainstorm topics | `moto autonomous brainstorms` / `brainstorm ID` |
| Papers | `moto autonomous papers` / `paper ID` / `progress` |
| Final answer (Tier 3) | `moto autonomous final-answer` |
| Steer the run | `moto autonomous force-paper-writing`, `force-tier3 --tier3-mode skip_incomplete` |
| Aggregator database | `moto aggregator results` |
| Paper being written | `moto compiler paper` / `outline` |
| Proof Solver | `moto leanoj proofs` / `master-proof` / `skip-brainstorm` |
| Proofs | `moto proofs list` / `show ID` / `certificate ID --lean -o f.lean` / `library` / `search TEXT` |

## Proof checks

`moto proofs check aggregator` (or `compiler`, `brainstorm TOPIC_ID`,
`paper PAPER_ID`) looks for new Lean proofs in that source; add `--loop` to keep
going until 10 rounds in a row find nothing. Then `moto proofs runs`,
`moto proofs run ID [--activity]`, `moto proofs stop ID`.

## Fixing problems

- Out of OpenRouter credits: after the user adds credits, `moto retry openrouter`.
- Blocked solution-path update: `moto retry solution-path`.
- Provider alerts (failed sign-ins, usage limits): `moto alerts --json`.
- Settings: `moto settings show`, `moto settings free-models --looping on`,
  `moto settings skills --session-history off` (stop the workflow first).

## Anything else

`moto routes TEXT --schema --json` lists matching REST routes with their request
shapes. `moto api METHOD PATH` calls any route: `--data JSON` (or `@file`),
`--query KEY=VALUE`, `--file FIELD=PATH` uploads, `-o FILE` saves downloads such
as PDFs. Example: `moto api GET /api/auto-research/paper-history`.

## Several instances

`moto instances --json` lists running instances. Choose one with
`--instance ID` or `--data-root PATH`.

## Managing MOTO itself

- `moto launch [--no-browser]` starts MOTO. `moto open` opens the app.
- `moto shutdown --json` stops the running workflow (progress is kept), then
  closes MOTO cleanly. `--force` closes immediately.
- `moto update check`; `moto update apply --yes` installs an update (restart
  MOTO afterwards with `moto shutdown` and `moto launch`).
- `moto <mode> clear --yes` permanently erases that workflow's progress. Only do
  this when the user asked for it.
