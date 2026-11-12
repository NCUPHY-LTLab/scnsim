## T1. Readings of the same storage

Each Workspace Result is summarised four times: as returned, after `run.close()`, through `run.resolve` on a reopened Workspace, and through `run.resolve` in a fresh process. Each Memory Result is summarised as returned and after `run.close()`. `same` means every summarised public value, evidence tree included, is bit-identical.

| Case | Memory after close | Workspace after close | Workspace reopened | Workspace fresh process |
|---|---|---|---|---|
| coupling/single | same | same | same | same |
| coupling/single-at-point | same | same | same | same |
| coupling/single-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| coupling/single-fail-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| coupling/single-pole-branches | same | same | same | same |
| coupling/sweep | same | same | same | same |
| coupling/sweep-fail-point | same | same | same | same |
| diagonal_root/single | same | same | same | same |
| diagonal_root/single-at-point | same | same | same | same |
| diagonal_root/single-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/single-fail-hint | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/single-fail-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/single-search | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/single-search-at-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/single-search-fail-point | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/sweep | same | same | same | same |
| diagonal_root/sweep-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| diagonal_root/sweep-fail-point | same | same | same | same |
| direct/single | same | same | same | same |
| direct/single-fail | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| direct/single-fail-frequency | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| direct/single-fail-singular | n/a (call raised) | n/a (call raised) | raised: same failure | raised: same failure |
| direct/sweep | same | same | same | same |
| direct/sweep-fail | same | same | same | same |
| operator/single | same | same | same | same |
| operator/single-fail | same | same | same | same |
| operator/single-fail-singular | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| operator/sweep | same | same | same | same |
| operator/sweep-fail | same | same | same | same |
| operator/sweep-fail-singular | same | same | same | same |
| operator_element_root/single | same | same | same | same |
| operator_element_root/single-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| operator_element_root/single-fail-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| operator_element_root/sweep | same | same | same | same |
| operator_element_root/sweep-fail-point | same | same | same | same |
| optimization/coupling-cma | same | same | same | same |
| optimization/pole-cma | same | same | same | same |
| optimization/response-cma | same | same | same | same |
| optimization/root-cma | same | same | same | same |
| optimization/root-cma-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| optimization/root-cma-failing-trials | same | same | same | same |
| optimization/root-lbfgsb | same | same | same | same |
| optimization/zero-cma | same | same | same | same |
| pole/single | same | same | same | same |
| pole/single-at-point | same | same | same | same |
| pole/single-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| pole/single-fail-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| pole/sweep | same | same | same | same |
| pole/sweep-fail-point | same | same | same | same |
| response/single | same | same | same | same |
| response/single-fail | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| response/single-fail-singular | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| response/sweep | same | same | same | same |
| response/sweep-fail | same | same | same | same |
| sensitivity/coupling-jacobian | same | same | same | same |
| sensitivity/pole-jacobian | same | same | same | same |
| sensitivity/pole-jvp | same | same | same | same |
| sensitivity/pole-vjp | same | same | same | same |
| sensitivity/response-fail | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| sensitivity/response-jacobian | same | same | same | same |
| sensitivity/response-jvp | same | same | same | same |
| sensitivity/response-vjp | same | same | same | same |
| sensitivity/root-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| sensitivity/root-fail-point | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| sensitivity/root-jacobian | same | same | same | same |
| sensitivity/root-jvp | same | same | same | same |
| sensitivity/root-vjp | same | same | same | same |
| sensitivity/zero-jacobian | same | same | same | same |
| static/element | same | same | same | same |
| static/element-fail | same | same | same | same |
| static/element-fail-singular | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| static/matrix | same | same | same | same |
| static/matrix-fail | same | same | same | same |
| static/matrix-fail-singular | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| static/matrix-retained | same | same | same | same |
| static/sweep-refused | n/a (call raised) | n/a (call raised) | raised: same failure | raised: same failure |
| transfer_zero/single | same | same | same | same |
| transfer_zero/single-at-point | same | same | same | same |
| transfer_zero/single-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| transfer_zero/single-fail-point | same | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| transfer_zero/single-search-at-point | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| transfer_zero/single-search-fail-point | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| transfer_zero/sweep | same | same | same | same |
| transfer_zero/sweep-fail-baseline | n/a (call raised) | n/a (call raised) | raised: ResultUnavailableError (`result_unavailable`) | raised: ResultUnavailableError (`result_unavailable`) |
| transfer_zero/sweep-fail-point | same | same | same | same |

## T2. Public evidence of a successful call

`evidence` is `result.evidence` (for a sweep: of each succeeded point, listed once when all points agree). The Memory column also lists the other public attributes of the in-memory Result that carry numerical evidence. `values` compares every public number of the Result bit for bit.

| Case | Memory `evidence` | Memory `evaluation.evidence_bytes` | Memory `dependencies` | Workspace `evidence` | Values | `evidence` equal |
|---|---|---|---|---|---|---|
| coupling/single | `{}` | 17 keys, 12437 B JSON | 2 | 17 keys, 2668 B arrays | same | **no** |
| coupling/single-at-point | `{}` | 17 keys, 12437 B JSON | 2 | 17 keys, 2668 B arrays | same | **no** |
| coupling/single-pole-branches | `{}` | 17 keys, 7187 B JSON | 2 | 17 keys, 2400 B arrays | same | **no** |
| coupling/sweep | `{}` | 17 keys, 12437 B JSON | 2 | 18 keys, 2968 B arrays | same | **no** |
| diagonal_root/single | `{}` | 22 keys, 2872 B JSON | 0 | `{}` | same | yes |
| diagonal_root/single-at-point | `{}` | 22 keys, 2872 B JSON | 0 | `{}` | same | yes |
| diagonal_root/sweep | `{}` | 22 keys, 2872 B JSON | 0 | `{}` | same | yes |
| direct/single | `{}` | no `evaluation` | n/a | `{}` | same | yes |
| direct/sweep | `{}` | 4 keys, 11363 B JSON | 0 | `{}` | same | yes |
| operator/single | `{}` | 5 keys, 1644 B JSON | 0 | `{}` | same | yes |
| operator/sweep | `{}` | 5 keys, 2327 B JSON | 0 | `{}` | same | yes |
| operator_element_root/single | `{}` | 22 keys, 2960 B JSON | 0 | 25 keys, 719 B arrays | same | **no** |
| operator_element_root/sweep | `{}` | 22 keys, 2960 B JSON | 0 | 25 keys, 719 B arrays | same | **no** |
| pole/single | `{}` | 22 keys, 1921 B JSON | 0 | 24 keys, 585 B arrays | same | **no** |
| pole/single-at-point | `{}` | 22 keys, 1921 B JSON | 0 | 24 keys, 585 B arrays | same | **no** |
| pole/sweep | `{}` | 22 keys, 1921 B JSON | 0 | 24 keys, 585 B arrays | same | **no** |
| response/single | `{}` | 5 keys, 1463 B JSON | 0 | `{}` | same | yes |
| response/sweep | `{}` | 5 keys, 1463 B JSON | 0 | `{}` | same | yes |
| static/element | 11 keys, 30 B arrays | 11 keys, 1136 B JSON | 0 | 11 keys, 30 B arrays | same | yes |
| static/matrix | 11 keys, 0 B arrays | 11 keys, 554 B JSON | 0 | 11 keys, 0 B arrays | same | yes |
| static/matrix-retained | 11 keys, 30 B arrays | 11 keys, 1136 B JSON | 0 | 11 keys, 30 B arrays | same | yes |
| transfer_zero/single | `{}` | 27 keys, 6387 B JSON | 0 | 31 keys, 1113 B arrays | same | **no** |
| transfer_zero/single-at-point | `{}` | 27 keys, 6387 B JSON | 0 | 31 keys, 1113 B arrays | same | **no** |
| transfer_zero/sweep | `{}` | 27 keys, 6387 B JSON | 0 | 31 keys, 1113 B arrays | same | **no** |

## T3. Is the Workspace evidence the in-memory record plus extra keys?

For each successful root-type call: the keys of the Workspace `result.evidence` that the in-memory record (`result.evaluation.evidence_bytes`, decoded as JSON) does not have, the keys only the in-memory record has, and the common keys whose bits differ.

| Case | Workspace-only keys | Memory-only keys | Common keys that differ |
|---|---|---|---|
| coupling/single | `branches` | `equation_certificates` | none |
| coupling/single-at-point | `branches` | `equation_certificates` | none |
| coupling/single-pole-branches | `branches` | `equation_certificates` | none |
| coupling/sweep | `branches` | none | none |
| operator_element_root/single | `column`, `row`, `selected_state` | none | none |
| operator_element_root/sweep | `column`, `row`, `selected_state` | none | none |
| pole/single | `null_vector`, `selected_state` | none | none |
| pole/single-at-point | `null_vector`, `selected_state` | none | none |
| pole/sweep | `null_vector`, `selected_state` | none | none |
| static/element | none | none | none |
| static/matrix | none | none | none |
| static/matrix-retained | none | none | none |
| transfer_zero/single | `family`, `input`, `output`, `selected_state` | none | none |
| transfer_zero/single-at-point | `family`, `input`, `output`, `selected_state` | none | none |
| transfer_zero/sweep | `family`, `input`, `output`, `selected_state` | none | none |

## T4. Failing calls

| Case | Memory | Workspace | Workspace `resolve` after reopening | Same surface | Evidence difference |
|---|---|---|---|---|---|
| coupling/single-fail-baseline | RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `residue_coupling`, 16 evidence keys | RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `residue_coupling`, 16 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| coupling/single-fail-point | RETURNED failed Result (`status == "failure"`): NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `selected_state` |
| coupling/sweep-fail-point | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys, `point.result` raises NumericalResolutionUnresolved | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included, `point.result` raises NumericalResolutionUnresolved | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included, `point.result` raises NumericalResolutionUnresolved | yes | Workspace only: `selected_state` |
| diagonal_root/single-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 22 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `failed_site`, `realized_capabilities`; Memory only: `certificate`, `certificate_status`, `check_status`, `closure`, `derivative_equation`, `derivative_limit`, `determinant_derivative_exponent`, `determinant_derivative_mantissa`, `determinant_exponent`, `determinant_mantissa`, `failure_stage`, `left_vector`, `normalizer_exponent`, `normalizer_mantissa`, `omega`, `right_vector`, `row_shifts`, `singular_values`, `slope`, `steps`; common keys differ: `certificates`, `status`; failure identity differs (message: `root did not satisfy its existing local Newton/closure contract` vs `numerical operation did not satisfy its existing solve/root contract`) |
| diagonal_root/single-fail-hint | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `selected_state` |
| diagonal_root/single-fail-point | RETURNED failed Result (`status == "failure"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 22 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `selected_state` |
| diagonal_root/single-search-fail-point | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `root_search`, 7 evidence keys | RAISED SCNSimValidationError, kind `validation_error`, stage `canonical_identity`, 1 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `type`; Memory only: `authored_hint_hz`, `candidates`, `catalogue`, `declaration`, `resolved_imag_frequency_interval_hz`, `resolved_real_frequency_interval_hz`, `selected_source_index`; failure identity differs (type: `NumericalResolutionUnresolved` vs `SCNSimValidationError`; kind: `numerical_resolution_unresolved` vs `validation_error`; stage: `root_search` vs `canonical_identity`; message: `Declared root search returned no certified candidate inside its region` vs `canonical JSON received an unsupported value`) |
| diagonal_root/sweep-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 22 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `selected_state` |
| diagonal_root/sweep-fail-point | RETURNED sweep, points TTFT; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 22 evidence keys, `point.result` raises EliminatedBlockSolveFailure | RETURNED sweep, points TTFT; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 23 evidence keys, `selected_state` included, `point.result` raises EliminatedBlockSolveFailure | RETURNED sweep, points TTFT; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 23 evidence keys, `selected_state` included, `point.result` raises EliminatedBlockSolveFailure | yes | Workspace only: `selected_state` |
| direct/single-fail | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | common keys differ: `certificates` |
| direct/single-fail-frequency | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | common keys differ: `certificates` |
| direct/single-fail-singular | RAISED PortRealizabilityError, kind `port_realizability`, stage `preflight`, 3 evidence keys | RAISED PortRealizabilityError, kind `port_realizability`, stage `preflight`, 3 evidence keys | RAISED PortRealizabilityError, kind `port_realizability`, stage `preflight`, 3 evidence keys | yes | none |
| direct/sweep-fail | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 7 evidence keys, `point.result` raises DirectResponseFormationError | yes | none |
| operator/single-fail | no failure with this input | no failure with this input | - | yes | n/a |
| operator/single-fail-singular | RETURNED failed Result (`status == "failure"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`) |
| operator/sweep-fail | no failure with this input | no failure with this input | - | yes | n/a |
| operator/sweep-fail-singular | RETURNED sweep, points FF; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys, `point.result` raises EliminatedBlockSolveFailure | RETURNED sweep, points FF; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys, `point.result` raises EliminatedBlockSolveFailure | RETURNED sweep, points FF; failed point: EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys, `point.result` raises EliminatedBlockSolveFailure | yes | none |
| operator_element_root/single-fail-baseline | RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `newton`, 22 evidence keys | RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `newton`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `selected_state` |
| operator_element_root/single-fail-point | RETURNED failed Result (`status == "failure"`): NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `selected_state` |
| operator_element_root/sweep-fail-point | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys, `point.result` raises NumericalResolutionUnresolved | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included, `point.result` raises NumericalResolutionUnresolved | RETURNED sweep, points TTFT; failed point: NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included, `point.result` raises NumericalResolutionUnresolved | yes | Workspace only: `selected_state` |
| optimization/root-cma-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| optimization/root-cma-failing-trials | no failure with this input | no failure with this input | - | yes | n/a |
| pole/single-fail-baseline | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `rank_certificate`, 22 evidence keys | RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `rank_certificate`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `selected_state` |
| pole/single-fail-point | RETURNED failed Result (`status == "failure"`): RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 22 evidence keys | RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 23 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `selected_state` |
| pole/sweep-fail-point | RETURNED sweep, points TTFT; failed point: RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 22 evidence keys, `point.result` raises RootSlopeUnresolved | RETURNED sweep, points TTFT; failed point: RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 23 evidence keys, `selected_state` included, `point.result` raises RootSlopeUnresolved | RETURNED sweep, points TTFT; failed point: RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 23 evidence keys, `selected_state` included, `point.result` raises RootSlopeUnresolved | yes | Workspace only: `selected_state` |
| response/single-fail | RETURNED failed Result (`status == "failure"`): DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 7 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`) |
| response/single-fail-singular | RETURNED failed Result (`status == "failure"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`) |
| response/sweep-fail | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 7 evidence keys, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 7 evidence keys, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 7 evidence keys, `point.result` raises DirectResponseFormationError | yes | none |
| sensitivity/response-fail | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| sensitivity/root-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| sensitivity/root-fail-point | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| static/element-fail | no failure with this input | no failure with this input | - | yes | n/a |
| static/element-fail-singular | RETURNED failed Result (`status == "failure"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | none |
| static/matrix-fail | no failure with this input | no failure with this input | - | yes | n/a |
| static/matrix-fail-singular | RETURNED failed Result (`status == "failure"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | none |
| static/sweep-refused | RAISED RuntimePreparationError, kind `runtime_preparation`, stage `preflight`, 0 evidence keys | RAISED RuntimePreparationError, kind `runtime_preparation`, stage `preflight`, 0 evidence keys | RAISED RuntimePreparationError, kind `runtime_preparation`, stage `preflight`, 0 evidence keys | yes | none |
| transfer_zero/single-fail-baseline | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 27 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `dut_voltage_constraint`, 4 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `failed_site`, `realized_capabilities`; Memory only: `certificate`, `certificate_status`, `check_status`, `closure`, `denominator`, `denominator_certificate`, `derivative_equation`, `derivative_limit`, `failure_stage`, `iteration_exponent`, `iteration_mantissa`, `iteration_residual`, `iteration_slope_exponent`, `iteration_slope_mantissa`, `numerator_exponent`, `numerator_mantissa`, `numerator_slope`, `omega`, `row_shifts`, `singular_values`, `slope`, `slope_exponent`, `slope_mantissa`, `source_stage`, `steps`; common keys differ: `certificates`, `status`; failure identity differs (stage: `transfer_denominator` vs `dut_voltage_constraint`; message: `root did not satisfy its existing local Newton/closure contract` vs `numerical operation did not satisfy its existing solve/root contract`) |
| transfer_zero/single-fail-point | RETURNED failed Result (`status == "failure"`): DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 27 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 28 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | **no** | Workspace only: `selected_state` |
| transfer_zero/single-search-fail-point | RAISED RuntimePreparationError, kind `runtime_preparation`, stage `root_search_prepare`, 0 evidence keys | RAISED RuntimePreparationError, kind `runtime_preparation`, stage `root_search_prepare`, 0 evidence keys | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | none |
| transfer_zero/sweep-fail-baseline | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 27 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 28 evidence keys, `selected_state` included | RAISED ResultUnavailableError, kind `result_unavailable`, stage `resolve`, 2 evidence keys | yes | Workspace only: `selected_state` |
| transfer_zero/sweep-fail-point | RETURNED sweep, points TTFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 27 evidence keys, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TTFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 28 evidence keys, `selected_state` included, `point.result` raises DirectResponseFormationError | RETURNED sweep, points TTFT; failed point: DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 28 evidence keys, `selected_state` included, `point.result` raises DirectResponseFormationError | yes | Workspace only: `selected_state` |

## T5. Sensitivity

| Case | Memory | Workspace | Values and block failures equal | Memory-only public attributes |
|---|---|---|---|---|
| sensitivity/coupling-jacobian | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 12 records, 360 B arrays |
| sensitivity/pole-jacobian | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 1 records, 30 B arrays |
| sensitivity/pole-jvp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 1 records, 30 B arrays |
| sensitivity/pole-vjp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 1 records, 30 B arrays |
| sensitivity/response-fail | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys | RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys | n/a | n/a |
| sensitivity/response-jacobian | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 3 records, 90 B arrays |
| sensitivity/response-jvp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 3 records, 90 B arrays |
| sensitivity/response-vjp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 5 records, 150 B arrays |
| sensitivity/root-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | n/a | n/a |
| sensitivity/root-fail-point | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | n/a | n/a |
| sensitivity/root-jacobian | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 13 records, 390 B arrays |
| sensitivity/root-jvp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jvp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 13 records, 390 B arrays |
| sensitivity/root-vjp | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'vjp': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 13 records, 390 B arrays |
| sensitivity/zero-jacobian | RETURNED InMemorySensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | RETURNED SensitivityResult, status `complete`, primal ['complete'], operations {'jacobian': ('complete', ['complete'])} | yes | `benchmark`, `certificates`, `execution_id`, `observable_metadata`, `operation`, `output_layout`, `parameters`, `persistence`, `persistence_kind`, `plan_sha256`, `quantity_metadata`, `request_sha256`, `summary`; `certificates`: 21 records, 630 B arrays |

## T6. Optimization

| Case | Memory | Workspace | Memory: bodies kept | Workspace: evidence reachable from public history |
|---|---|---|---|---|
| optimization/coupling-cma | RETURNED InMemoryOptimizationResult, history SSSSS | RETURNED OptimizationResult, history SSSSS | 5 history rows, 15 bodies, 57420 B of `evidence_bytes`, 0 with `selected_state`, 50 certificate records | 5 history rows; 1 rows carry `dependencies` (3 bodies, 2732 B arrays, 2 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |
| optimization/pole-cma | RETURNED InMemoryOptimizationResult, history SSSSS | RETURNED OptimizationResult, history SSSSS | 5 history rows, 5 bodies, 9605 B of `evidence_bytes`, 0 with `selected_state`, 0 certificate records | 5 history rows; 1 rows carry `dependencies` (1 bodies, 617 B arrays, 1 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |
| optimization/response-cma | RETURNED InMemoryOptimizationResult, history SSSSS | RETURNED OptimizationResult, history SSSSS | 5 history rows, 5 bodies, 4285 B of `evidence_bytes`, 0 with `selected_state`, 5 certificate records | 5 history rows; 1 rows carry `dependencies` (1 bodies, 46 B arrays, 0 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |
| optimization/root-cma | RETURNED InMemoryOptimizationResult, history SSSSS | RETURNED OptimizationResult, history SSSSS | 5 history rows, 5 bodies, 14360 B of `evidence_bytes`, 0 with `selected_state`, 20 certificate records | 5 history rows; 1 rows carry `dependencies` (1 bodies, 643 B arrays, 1 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |
| optimization/root-cma-fail-baseline | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys | n/a | n/a |
| optimization/root-cma-failing-trials | RETURNED InMemoryOptimizationResult, history SFFFF | RETURNED OptimizationResult, history SFFFF | 5 history rows, 1 bodies, 2872 B of `evidence_bytes`, 0 with `selected_state`, 4 certificate records; failed trial `failure` keys: detail, evidence, kind, phase, stage (evidence keys: 4) | 5 history rows; 1 rows carry `dependencies` (1 bodies, 643 B arrays, 1 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0; failed row `failure` keys: detail, evidence_bytes, kind, phase, stage |
| optimization/root-lbfgsb | RETURNED InMemoryOptimizationResult, history SSSSSS | RETURNED OptimizationResult, history SSSSSS | 6 history rows, 6 bodies, 17232 B of `evidence_bytes`, 0 with `selected_state`, 69 certificate records | 6 history rows; 1 rows carry `dependencies` (1 bodies, 643 B arrays, 1 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |
| optimization/zero-cma | RETURNED InMemoryOptimizationResult, history SSSSS | RETURNED OptimizationResult, history SSSSS | 5 history rows, 5 bodies, 31935 B of `evidence_bytes`, 0 with `selected_state`, 40 certificate records | 5 history rows; 1 rows carry `dependencies` (1 bodies, 1177 B arrays, 1 with `selected_state`); 0 terms carry `evidence`; 0 terms carry `residue_coupling_evidence`; `result.ledger` length 0 |

## T7. Every Memory/Workspace difference found by this script

```json
[
 {
  "case": "coupling/single",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "17 keys, 2668 B arrays",
  "facts": {
   "workspace_only": [
    "branches"
   ],
   "memory_only": [
    "equation_certificates"
   ],
   "differing_common_keys": []
  }
 },
 {
  "case": "coupling/single-at-point",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "17 keys, 2668 B arrays",
  "facts": {
   "workspace_only": [
    "branches"
   ],
   "memory_only": [
    "equation_certificates"
   ],
   "differing_common_keys": []
  }
 },
 {
  "case": "coupling/single-pole-branches",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "17 keys, 2400 B arrays",
  "facts": {
   "workspace_only": [
    "branches"
   ],
   "memory_only": [
    "equation_certificates"
   ],
   "differing_common_keys": []
  }
 },
 {
  "case": "coupling/sweep",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "18 keys, 2968 B arrays",
  "facts": {
   "workspace_only": [
    "branches"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "operator_element_root/single",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "25 keys, 719 B arrays",
  "facts": {
   "workspace_only": [
    "column",
    "row",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "operator_element_root/sweep",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "25 keys, 719 B arrays",
  "facts": {
   "workspace_only": [
    "column",
    "row",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "pole/single",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "24 keys, 585 B arrays",
  "facts": {
   "workspace_only": [
    "null_vector",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "pole/single-at-point",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "24 keys, 585 B arrays",
  "facts": {
   "workspace_only": [
    "null_vector",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "pole/sweep",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "24 keys, 585 B arrays",
  "facts": {
   "workspace_only": [
    "null_vector",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "transfer_zero/single",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "31 keys, 1113 B arrays",
  "facts": {
   "workspace_only": [
    "family",
    "input",
    "output",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "transfer_zero/single-at-point",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "31 keys, 1113 B arrays",
  "facts": {
   "workspace_only": [
    "family",
    "input",
    "output",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "transfer_zero/sweep",
  "kind": "success_evidence",
  "memory": "`{}`",
  "workspace": "31 keys, 1113 B arrays",
  "facts": {
   "workspace_only": [
    "family",
    "input",
    "output",
    "selected_state"
   ],
   "memory_only": [],
   "differing_common_keys": []
  }
 },
 {
  "case": "coupling/single-fail-point",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys",
  "workspace": "RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included"
 },
 {
  "case": "coupling/single-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "coupling/sweep-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "diagonal_root/single-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `failed_site`, `realized_capabilities`; Memory only: `certificate`, `certificate_status`, `check_status`, `closure`, `derivative_equation`, `derivative_limit`, `determinant_derivative_exponent`, `determinant_derivative_mantissa`, `determinant_exponent`, `determinant_mantissa`, `failure_stage`, `left_vector`, `normalizer_exponent`, `normalizer_mantissa`, `omega`, `right_vector`, `row_shifts`, `singular_values`, `slope`, `steps`; common keys differ: `certificates`, `status`; failure identity differs (message: `root did not satisfy its existing local Newton/closure contract` vs `numerical operation did not satisfy its existing solve/root contract`)"
 },
 {
  "case": "diagonal_root/single-fail-hint",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "diagonal_root/single-fail-point",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 22 evidence keys",
  "workspace": "RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 23 evidence keys, `selected_state` included"
 },
 {
  "case": "diagonal_root/single-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "diagonal_root/single-search-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `type`; Memory only: `authored_hint_hz`, `candidates`, `catalogue`, `declaration`, `resolved_imag_frequency_interval_hz`, `resolved_real_frequency_interval_hz`, `selected_source_index`; failure identity differs (type: `NumericalResolutionUnresolved` vs `SCNSimValidationError`; kind: `numerical_resolution_unresolved` vs `validation_error`; stage: `root_search` vs `canonical_identity`; message: `Declared root search returned no certified candidate inside its region` vs `canonical JSON received an unsupported value`)"
 },
 {
  "case": "diagonal_root/sweep-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "diagonal_root/sweep-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "direct/single-fail",
  "kind": "failure_evidence",
  "detail": "common keys differ: `certificates`"
 },
 {
  "case": "direct/single-fail-frequency",
  "kind": "failure_evidence",
  "detail": "common keys differ: `certificates`"
 },
 {
  "case": "operator/single-fail-singular",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys",
  "workspace": "RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys"
 },
 {
  "case": "operator/single-fail-singular",
  "kind": "failure_evidence",
  "detail": "Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`)"
 },
 {
  "case": "operator_element_root/single-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "operator_element_root/single-fail-point",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 22 evidence keys",
  "workspace": "RAISED NumericalResolutionUnresolved, kind `numerical_resolution_unresolved`, stage `newton_certificate`, 23 evidence keys, `selected_state` included"
 },
 {
  "case": "operator_element_root/single-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "operator_element_root/sweep-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "pole/single-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "pole/single-fail-point",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 22 evidence keys",
  "workspace": "RAISED RootSlopeUnresolved, kind `root_slope_unresolved`, stage `determinant_scaling`, 23 evidence keys, `selected_state` included"
 },
 {
  "case": "pole/single-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "pole/sweep-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "response/single-fail",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 7 evidence keys",
  "workspace": "RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `y_to_z`, 4 evidence keys"
 },
 {
  "case": "response/single-fail",
  "kind": "failure_evidence",
  "detail": "Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`)"
 },
 {
  "case": "response/single-fail-singular",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 7 evidence keys",
  "workspace": "RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys"
 },
 {
  "case": "response/single-fail-singular",
  "kind": "failure_evidence",
  "detail": "Workspace only: `failed_site`; Memory only: `certificate_position`, `certificate_status`, `check_position`, `check_status`; common keys differ: `certificates`; failure identity differs (message: `numerical operation did not satisfy its existing solve contract` vs `numerical operation did not satisfy its existing solve/root contract`)"
 },
 {
  "case": "static/element-fail-singular",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys",
  "workspace": "RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys"
 },
 {
  "case": "static/matrix-fail-singular",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys",
  "workspace": "RAISED EliminatedBlockSolveFailure, kind `eliminated_block_solve_failure`, stage `eliminated_block`, 4 evidence keys"
 },
 {
  "case": "transfer_zero/single-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `failed_site`, `realized_capabilities`; Memory only: `certificate`, `certificate_status`, `check_status`, `closure`, `denominator`, `denominator_certificate`, `derivative_equation`, `derivative_limit`, `failure_stage`, `iteration_exponent`, `iteration_mantissa`, `iteration_residual`, `iteration_slope_exponent`, `iteration_slope_mantissa`, `numerator_exponent`, `numerator_mantissa`, `numerator_slope`, `omega`, `row_shifts`, `singular_values`, `slope`, `slope_exponent`, `slope_mantissa`, `source_stage`, `steps`; common keys differ: `certificates`, `status`; failure identity differs (stage: `transfer_denominator` vs `dut_voltage_constraint`; message: `root did not satisfy its existing local Newton/closure contract` vs `numerical operation did not satisfy its existing solve/root contract`)"
 },
 {
  "case": "transfer_zero/single-fail-point",
  "kind": "failure_surface",
  "memory": "RETURNED failed Result (`status == \"failure\"`): DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 27 evidence keys",
  "workspace": "RAISED DirectResponseFormationError, kind `direct_response_formation`, stage `transfer_denominator`, 28 evidence keys, `selected_state` included"
 },
 {
  "case": "transfer_zero/single-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "transfer_zero/sweep-fail-baseline",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 },
 {
  "case": "transfer_zero/sweep-fail-point",
  "kind": "failure_evidence",
  "detail": "Workspace only: `selected_state`"
 }
]
```
