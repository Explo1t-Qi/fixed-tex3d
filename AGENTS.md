# AGENTS.md

## Repository Purpose

This repository is the working baseline for Tex3D robustness experiments on Vision-Language-Action models.

Current development must preserve the original Tex3D behavior unless a task explicitly requires a correctness fix or a new opt-in experimental method.

## Development Rules

1. **Never modify `main` directly.**

   * Start each task from a clean working tree.
   * Create a dedicated branch before editing.
   * Record the base commit before development.

2. **Use Git for every code change.**

   * Check `git status` before and after work.
   * Keep changes scoped to the current task.
   * Commit completed logical changes with clear commit messages.
   * Do not mix unrelated refactors with scientific changes.

3. **Test every behavioral change.**

   * Add or update unit/regression tests for each bug fix or new behavior.
   * Run the relevant tests after each logical modification.
   * Run the broader test suite before declaring the task complete.
   * GPU smoke tests are required when the modified path depends on CUDA, OpenVLA, LIBERO, or nvdiffrast.

4. **Prefer minimal fixes.**

   * Reuse existing interfaces where possible.
   * Do not perform large refactors unless they are necessary for correctness or explicitly requested.
   * Do not silently introduce new research methods while fixing baseline bugs.

5. **Preserve scientific boundaries.**

   * Baseline correctness fixes must not change the attack objective unless explicitly requested.
   * New experimental objectives must be opt-in and must not overwrite the original baseline behavior.
   * Multi-model information must not enter formal single-surrogate texture optimization unless explicitly authorized.

6. **Use authoritative implementations as references.**

   * `/home/xmq/src/openvla` is the reference for OpenVLA model and preprocessing semantics.
   * `/home/xmq/src/modified-tex3d` may be used to understand historical bugs and validated fixes.
   * Do not copy historical implementations blindly; reproduce the bug in the current baseline first.

7. **Require evidence before claiming a fix.**
   For every bug:

   * reproduce or audit the current behavior;
   * implement the smallest justified fix;
   * add a regression test;
   * report the test or smoke evidence.

8. **Do not fabricate experimental results.**

   * If GPU, checkpoints, assets, or dependencies are unavailable, report what was not executed.
   * Never infer PASS from code inspection alone when runtime validation is required.

## Required Final Report

Every completed development task must report:

* branch name;
* base commit;
* final HEAD;
* `git status`;
* changed files and their responsibilities;
* tests executed and results;
* GPU smoke results when applicable;
* remaining known issues or unexecuted validation.
