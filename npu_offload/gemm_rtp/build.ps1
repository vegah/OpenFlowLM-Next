# Build every open_npue (BERT) design set, in order.
#
#   cd C:\dev\mlir-aie; . .\iron_env.ps1        # MUST be dot-sourced
#   cd <repo>; .\npu_offload\gemm_rtp\build.ps1
#
# THIS IS NOW A WRAPPER over `python tools/build_designs.py`, which builds the
# Qwen sets in open_kernels/ as well and holds ONE ~/.npu/cache lock across
# both. The loop that used to live here was Windows-only, and the header it
# carried claimed a guard that this tree does not have:
#
#   "export_gemm_rtp.py holds a lock now and refuses in under a second"
#
# It does upstream. The copy in this repository predates that change, so until
# the entry command took the lock, two builds through the shared cache deleted
# each other's work exactly as the comment said they would -- and the failure
# lands minutes later as a FileNotFoundError on a cache hash that names
# nothing. The other traps the old header recorded are still real and are why
# nobody should paste build commands by hand:
#
#   * `<dst>` -- PowerShell rejects `<` as a reserved operator during PARSING,
#     so it dies naming neither the placeholder nor what was forgotten.
#   * `$dst` -- PowerShell does NOT error on an undefined variable, it expands
#     it to nothing, so `--out $dst\BERT-...` resolves to the DRIVE ROOT.
#     Builds fine, three minutes a family, puts the set where nothing looks.
#
# THE FLAGS ARE IN families.json, NOT HERE and not in build_designs.py. One
# machine-readable source that the build reads and check_design_sets.py
# verifies against, so there is no second copy to drift.

[CmdletBinding()]
param(
    # Build one family instead of all five, by name (e.g. BERT-h1024-bfp16).
    [string] $Only = "",
    # Where the sets go. Defaults to <repo>/src/xclbins.
    [string] $Dst = "",
    # Rebuild even if the set is already there.
    [switch] $Force,
    # Break a stale ~/.npu/cache lock left by a crashed build.
    [switch] $ForceUnlock
)

$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot '..\..\tools\build_designs.py'
$entry = [IO.Path]::GetFullPath($entry)

# Without the dot-sourced environment there is no `python` at all, and the
# failure ("The term 'python' is not recognized") reads as a broken checkout
# rather than a shell that was never set up.
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "No 'python' on this shell's path." -ForegroundColor Red
    Write-Host ""
    Write-Host "    cd C:\dev\mlir-aie"
    Write-Host "    . .\iron_env.ps1        # MUST be dot-sourced"
    Write-Host ""
    Write-Host "Then re-run. (XILINX_XRT must stay UNSET -- it poisons Windows"
    Write-Host "builds. Use XRT_ROOT.) For everything else: python tools\build_designs.py doctor"
    exit 1
}

$argv = @('build', '--producer', 'gemm_rtp')
if ($Only)        { $argv += @('--only', $Only) }
if ($Dst)         { $argv += @('--xclbins', [IO.Path]::GetFullPath($Dst)) }
if ($Force)       { $argv += '--force' }
if ($ForceUnlock) { $argv += '--force-unlock' }

& python $entry @argv
exit $LASTEXITCODE
