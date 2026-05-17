# HardOJ AWS Outline: User-Donated Lean Verification and Karma

## Purpose

HardOJ is an advanced Lean proof challenge platform where users can submit theorem templates, vote on templates, solve templates, and optionally donate their own compute to verify Lean proofs safely.

The core product goal is to create a public problem marketplace for hard formalization targets, including advanced theorem templates such as unsolved Lean 100-style problems, while keeping verification trustworthy, reproducible, and resistant to compute abuse.

## Core User Flows

### Template Submission

Users can submit Lean theorem templates as public challenge problems.

A template includes:

- A title and informal mathematical statement
- The Lean theorem statement
- Required imports
- A pinned Lean version and Mathlib revision
- Optional explanatory notes, references, and difficulty tags
- Optional helper lemmas or staged subproblem templates
- A starter file ending in `sorry`

Example template shape:

```lean
import Mathlib

/-!
Informal statement:
This section explains the intended mathematical theorem.

Rules:
- The final proof must not use `axiom`, `constant`, `opaque`, or unrelated imported shortcuts.
- The theorem statement must remain unchanged.
- The proof must compile under the pinned Lean and Mathlib versions.
-/

theorem hardoj_target_theorem
    (/* variables */)
    (/* hypotheses */) :
    /* formal claim */ := by
  sorry
```

Before a submitted template becomes public, HardOJ validates that:

- The Lean file parses and typechecks with `sorry` allowed
- Imports are on the allowlist or approved by moderation
- The theorem statement is not empty, trivial, or already solved in the submitted file
- The template does not introduce fake proof devices such as `axiom`, `constant`, `opaque`, or unsafe escape hatches
- The informal statement reasonably matches the formal Lean target

### Template Voting

Users can upvote or downvote templates.

Voting is used to surface important, interesting, difficult, and well-scoped theorem challenges. A template's visible score works like Reddit-style post score: upvotes increase the score, downvotes decrease it, and ranking can use time decay, controversy, and anti-abuse filtering.

Templates with high scores become more valuable to solve because successful solvers receive karma based on the template's current score.

### Proof Submission

Users submit a completed Lean proof for a public template.

A proof submission includes:

- The unchanged template theorem statement
- The user's proof replacing `sorry`
- A proof hash
- The Lean and Mathlib version used
- Local verifier logs if available
- Optional explanation of the proof strategy

HardOJ accepts the solution only if the proof compiles in the pinned environment and passes integrity checks.

### Karma Rewards

Karma works like Reddit-style karma for platform reputation.

Users gain karma when other users upvote their submitted templates, comments, explanations, and accepted solutions. Users lose karma from downvotes, subject to anti-abuse controls.

Special HardOJ rule:

- If a user solves a template, the solver receives solution karma equal to the template's current positive score at the time the solution is accepted.
- If someone else solves a user's template, the template creator receives a creator bonus equal to 25% of that same positive score snapshot.

Example:

- A template has score `87`
- A user submits the first accepted proof
- The solver receives `87` solution karma
- The template creator receives `21.75` creator-bonus karma, rounded according to the platform's configured karma precision

If a template has a score below zero, the solution reward should floor at zero unless the platform later chooses to allow negative challenge rewards.

## Compute Donation Model

HardOJ should start with a safe model: users donate compute to verify their own submissions locally. Public volunteer verification can be added later with stronger controls.

### Local User-Owned Verification

The user runs a HardOJ verifier agent on their own machine.

The verifier:

1. Downloads the selected template package
2. Uses the pinned Lean and Mathlib versions
3. Inserts the user's submitted proof
4. Runs Lean in a sandbox
5. Returns verification metadata to HardOJ

Returned metadata includes:

- Template ID
- Submission ID
- Proof hash
- Lean version
- Mathlib revision
- Success or failure
- Lean errors if verification fails
- Runtime, memory use, and timeout status

Local verification helps users avoid wasting server resources and gives fast feedback. However, local verification alone should not be the sole source of official acceptance because a malicious client can fake success.

### Official Acceptance

For official karma, leaderboard placement, and "solved" status, HardOJ should use one of these trust paths:

1. **Trusted AWS verification**: HardOJ re-verifies the final proof on AWS before accepting it.
2. **Volunteer quorum plus spot check**: Multiple independent volunteer verifiers return matching success for the same proof hash, with HardOJ spot-checking high-value solves.
3. **Attested verifier later**: Use stronger attestation if a reliable cross-platform proof-verifier attestation path becomes available.

The recommended MVP is local pre-verification plus AWS official verification.

## Anti-Exploitation Rules for Donated Compute

User-donated compute must not become a free arbitrary job runner.

Every donated verification job must be constrained:

- Verify only a specific HardOJ template ID
- Use only the pinned Lean and Mathlib revision
- Use an immutable template package
- Use a proof file whose hash is known before execution
- Disable network access during Lean execution
- Run in a container, VM, or OS sandbox
- Enforce wall-time, CPU, RAM, process, and file-size limits
- Use a read-only project root and a temporary write directory
- Reject unauthorized imports or local file access
- Never expose user secrets, API keys, cookies, or filesystem paths to the sandbox
- Allow the user to cancel jobs at any time

For MVP, donated compute should verify only the user's own proofs. Public volunteer verification should be opt-in and off by default.

## AWS Architecture

### Frontend

Use one of:

- S3 + CloudFront for a static web frontend
- AWS Amplify for managed frontend deployment
- A separate web app stack if HardOJ shares infrastructure with another website

Frontend responsibilities:

- Browse templates
- Submit templates
- Vote on templates
- Submit proofs
- Show verification status
- Show karma, leaderboards, and user profiles
- Connect to the local verifier agent when installed

### API Layer

Use API Gateway or an Application Load Balancer in front of backend services.

Backend services can run on:

- ECS Fargate for containerized API services
- EKS if Kubernetes is already used
- Lambda for small event-driven tasks

Primary backend responsibilities:

- Authentication and user sessions
- Template submission and moderation
- Voting and karma accounting
- Proof submission lifecycle
- Verification queue management
- Leaderboards and notifications

### Database

Use PostgreSQL on Amazon RDS for relational data.

Recommended tables:

- `users`
- `templates`
- `template_versions`
- `template_votes`
- `proof_submissions`
- `verification_jobs`
- `verification_results`
- `karma_events`
- `comments`
- `comment_votes`
- `moderation_events`

Use immutable event rows for karma changes so reputation can be audited and recomputed.

### Object Storage

Use S3 for immutable artifacts:

- Template packages
- Lean source files
- Submitted proofs
- Verification logs
- Generated problem bundles
- Public downloadable archives

Every stored artifact should include a content hash.

### Verification Queue

Use SQS for verification jobs.

Job payload:

- Job ID
- Template ID
- Template version hash
- Proof submission ID
- Proof hash
- Lean toolchain version
- Mathlib revision
- Resource limits

AWS official verifiers and optional volunteer verifiers both consume jobs, but they should use different queues and trust levels.

### Official AWS Verifier Workers

Run official verifier workers on ECS Fargate, AWS Batch, or EC2 autoscaling groups.

Each verifier worker:

- Pulls one job from SQS
- Downloads immutable artifacts from S3
- Builds or reuses the pinned Lean environment
- Runs Lean in a locked sandbox
- Uploads logs and result metadata
- Writes the result to the backend

For hard theorem templates, AWS Batch or EC2 workers may be more practical than Lambda because Lean and Mathlib verification can be CPU-heavy and long-running.

## Local Verifier Agent

The local verifier agent is a small desktop service or CLI installed by the user.

Responsibilities:

- Authenticate with HardOJ
- Receive only user-approved jobs
- Download immutable template packages
- Run Lean in a sandbox
- Stream local logs to the browser
- Submit signed result metadata

The browser can connect to the local verifier through:

- `localhost` HTTP/WebSocket with a one-time pairing token
- A CLI command that verifies a downloaded package
- A desktop app wrapper

The local verifier should never accept remote arbitrary commands. It should expose only a narrow API:

- `GET /status`
- `POST /verify-template-proof`
- `POST /cancel-job`

## Template Ranking

Template ranking can use Reddit-style concepts:

- Net score: upvotes minus downvotes
- Hot ranking: score adjusted by age
- Top ranking: highest score over a time window
- New ranking: recent submissions
- Controversial ranking: high activity with mixed votes

HardOJ-specific ranking signals:

- Number of failed serious attempts
- Whether the template has a verified solution
- Difficulty tag
- Formalization quality score
- Moderator approval level
- Number of staged helper templates

## Karma System

Karma should be event-sourced.

Each karma change is stored as a `karma_event`:

- User ID
- Event type
- Source object type
- Source object ID
- Delta
- Timestamp
- Reason

Event types:

- Template upvote received
- Template downvote received
- Comment upvote received
- Comment downvote received
- Solution accepted
- Template solved creator bonus
- Solution upvote received
- Moderator adjustment
- Abuse rollback

Template solve reward:

- On accepted solution, compute `reward = max(template_score_at_acceptance, 0)`
- Compute `creator_bonus = reward * 0.25` for the template author when the solver is not the same user as the template author
- Add a `solution_accepted` karma event for the solver
- Add a `template_solved_creator_bonus` karma event for the template creator
- Store the score snapshot used for the reward
- Do not retroactively change the solver's reward or creator bonus if the template later gains or loses votes, unless the platform later adds a bounty mechanism

This keeps rewards predictable and prevents old solves from constantly changing user karma.

## Abuse Controls

HardOJ needs anti-abuse protections because votes and karma create incentives.

Recommended controls:

- One vote per user per object
- Rate limits on voting, posting, and proof submissions
- New-account vote weighting or trust thresholds
- Bot and sockpuppet detection
- Vote-ring detection
- Moderator review for high-value solve rewards
- Karma rollback events for abuse
- Shadow filtering for suspicious votes until reviewed
- No unlimited verification retries on shared AWS compute

## MVP Scope

The first version should implement:

- User accounts
- Template submission
- Template upvotes/downvotes
- Reddit-style template score
- Proof submission
- Local user-owned verifier for pre-checking
- AWS official verifier for accepted solves
- Karma events
- Solver receives karma equal to template score at acceptance
- Template creator receives a 25% karma bonus when another user solves their template
- Basic leaderboards

Public donated verification for other users should wait until after the local verifier and AWS verifier are stable.

## Later Extensions

Possible future additions:

- Volunteer verifier pool for public proofs
- Quorum-based verification
- Template bounties separate from karma
- AI-generated Lean template proposals
- AI semantic review for informal/formal theorem match
- Staged theorem packs for major open formalization targets
- Team solving
- Private templates before publication
- Proof explanation rewards
- HardOJ API for external theorem-proving agents

## Critical Invariants

1. A template can be public only if it typechecks with `sorry` in the pinned Lean environment.
2. A proof can be officially accepted only after trusted verification or an approved trust policy.
3. User-donated compute must never run arbitrary unscoped jobs.
4. Verification jobs must be sandboxed, resource-limited, and network-isolated.
5. Karma must be auditable through immutable karma events.
6. Solver karma from a template solve is based on the template score snapshot at acceptance.
7. Template creator solve bonuses are 25% of the same acceptance-time score snapshot.
8. Lean acceptance is necessary but not always semantically sufficient; high-value templates need informal/formal review.
