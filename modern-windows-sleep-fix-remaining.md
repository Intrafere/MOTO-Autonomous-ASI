# Modern Windows Sleep Fix: Remaining External Validation

The implementation and automated tests are complete. One optional operating-system
visibility check requires an elevated Windows terminal and could not be performed from
the current non-elevated process.

## Elevated `powercfg` visibility check

Run from an Administrator PowerShell in the repository root:

```powershell
$process = Start-Process python -ArgumentList @(
  "-c",
  'import time; from backend.shared.sleep_inhibitor import SleepInhibitor; s=SleepInhibitor(); s.acquire("powercfg-validation"); time.sleep(30); s.release_all(); time.sleep(3)'
) -PassThru
Start-Sleep -Seconds 3
powercfg /requests
Wait-Process -Id $process.Id
powercfg /requests
```

Acceptance criteria:

- During the 30-second active interval, `powercfg /requests` lists the Python
  process under both `SYSTEM` and `EXECUTION`, with the MOTO active-work reason.
- After the process releases its final owner, neither request remains.

The non-elevated attempt returned Windows' expected “requires administrator
privileges” response. The real native integration test
`test_real_windows_power_request_activates_and_releases` passed and confirms that
Windows successfully creates, sets, clears, and closes both request types.

## Existing deep-suite blockers

The required deep command completed its 356 deterministic workflow tests successfully,
then failed in the opt-in real-adapter matrix for pre-existing test-overlay reasons
unrelated to the sleep-inhibitor change:

- The maturity-registry assertion omits two already-registered blocked scenarios:
  `real_pruning_overflow_commit_interleaving_unobservable` and
  `real_continuous_loop_stop_terminal_zero_policy`.
- Six cases cannot hash a locked file under the workspace mutable runtime roots and
  fail with `PermissionError`. The repository currently has active runtime processes
  and mutable runtime data.

Re-run after reconciling that existing registry assertion and closing processes that
hold runtime files:

```powershell
npm run test:deep
```

The ordinary complete suite passed: 1,306 backend tests passed with 118 skipped, and
all 171 frontend tests passed.
