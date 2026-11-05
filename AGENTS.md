# Working on SCNSim

Use `$github-flow` for ownership, branches, allowed paths, review, CI and
delivery; auto-merge squashes after CI and `cross-review-gate` pass.

## Maintain behavior and its documentation together

An authorized feature, fix, refactor or removal includes its directly related
documentation, tests and ordinary local checks. Retiring a feature includes
removing its obsolete dedicated tests. Diagnose failures; do not hide them by
skipping tests, weakening assertions or reducing coverage.

Keep Human-defined research conditions, numerical tolerances, acceptance
criteria and privacy boundaries. Resolve changes to those requirements with
the Human. Follow the task's assigned numerical and verification scope; a
documentation change does not restart simulations or paused work.

Reuse valid results when their relevant inputs and behavior are unchanged.
Check the affected surface after a change. Run a full numerical comparison
only at the milestone assigned by the Human, not after every commit.

## Documentation and Preview

Keep public API explanations, docstrings, tutorials and examples aligned with
the implementation. Introductory tutorials use in-memory storage; advanced
material explains Workspace. Change generated pages and catalogues through
their owning sources and generators.

## Preserve work and deliver coherent changes

Preserve Human edits, other writers' work, held overlays and unrelated staged
changes. Compose each candidate from its declared base and owned changes.
Historical results remain attached to the source and artifact that produced
them.

Keep the assigned branch and current PR scope. Merge, release, deployment,
dependency pins and external data transfer retain their own authorization.
Passing checks do not themselves grant Human acceptance.

<!-- Paste this section into the repository's root AGENTS.md. Codex's GitHub
review reads `## Code Review Rules`; Claude's review receives the same rules.
Keep it short: every rule here is checked on every PR. -->
## Code Review Rules

- Review against the Issue named on the PR's `Issue:` line; a PR with no
  readable Issue has unverified acceptance.
- Review in two stages. First look for counterexamples from the change's
  normal entry points, including combinations that break assumptions (two
  valid inputs paired wrongly, two calls writing one output target, a result
  that passes quality but exceeds a limit the caller set). Then weigh each
  against the Issue, the public docs and the callers you can see.
- Each finding states the condition that reproduces it, what the code does,
  the code or assumption that causes it, and a proposed disposition with its
  basis: `fix` (a supported use breaks a promised behavior), `document` (the
  formal flow guarantees a precondition nothing states) or `observe` (no
  known caller reaches it). "Callers outside this repository not checked" is
  a valid basis.
- Flag as P1: a supported use breaking an acceptance criterion, a public
  contract, data or security, or Human-confirmed UI; an out-of-scope input
  that silently produces a plausible but wrong result; a test double
  standing in for the result being claimed (a real solver, database,
  authorization or integration). A counterexample outside the supported use
  is not P1 by itself.
- Test doubles that isolate dependencies are fine; call counts are fine when
  the count is the contract. Do not ask for checks the Issue does not need.
- A CI change must say in the PR what runs when, before and after, the
  expected time change and what could break.
