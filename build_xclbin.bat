@echo off
setlocal
::
:: Build the NPU design sets (xclbins). Same command, same behaviour, on every
:: host -- this file and build_xclbin.sh are FORWARDERS, not two implementations.
::
::     build_xclbin.bat                 build whatever is not built yet
::     build_xclbin.bat doctor          can this shell build at all?
::     build_xclbin.bat list            what sets exist, and are they built?
::     build_xclbin.bat check           do the built sets match their spec?
::     build_xclbin.bat build --force   rebuild everything
::
:: Everything real is in tools\build_designs.py, which drives BOTH producers
:: (npu_offload\gemm_rtp\ and open_kernels\) and holds one lock across them.
:: Nothing about an xclbin is OS-specific -- it is a device artifact (AIE core
:: ELFs, CDOs, a PDI) with no host code in it.
::
:: .bat AND NOT .ps1, DELIBERATELY. The build.ps1 this replaced carried a list
:: of PowerShell footguns that had each cost somebody a session: `<` is a
:: RESERVED PARSE OPERATOR, so a pasted `<dst>` placeholder dies naming neither
:: the placeholder nor what was forgotten; and an undefined `$dst` does not
:: error, it expands to nothing, so `--out $dst\BERT-...` silently built to the
:: DRIVE ROOT, four times. A .bat that forwards its arguments has almost no
:: surface of its own, and it runs from cmd.exe, from PowerShell and from CI
:: without an execution policy in the way.
::
:: Needs the IRON toolchain on PATH:
::     cd C:\dev\mlir-aie
::     . .\iron_env.ps1        (MUST be dot-sourced)
:: `doctor` reports what is missing rather than failing four minutes in.

:: RUN python, do not merely locate it. Windows ships an "App Execution Alias"
:: stub at %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe which IS on PATH and
:: which `where python` happily finds -- and which then prints "Python was not
:: found; run without arguments to install from the Microsoft Store" and exits.
:: A `where` check passes on such a machine and the build fails one line later
:: with a message about a store listing. Asking for a version is the only test
:: that distinguishes an interpreter from a shortcut to a shop.
python -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo No working 'python' on this shell's path.
    echo ^(If `where python` finds one anyway, it is the Microsoft Store alias
    echo stub in %%LOCALAPPDATA%%\Microsoft\WindowsApps -- not an interpreter.^)
    echo.
    echo     cd C:\dev\mlir-aie
    echo     . .\iron_env.ps1        ^(MUST be dot-sourced^)
    echo.
    echo Then re-run. XILINX_XRT must stay UNSET -- it poisons Windows builds;
    echo use XRT_ROOT. See docs\design-sets.md.
    exit /b 1
)

:: No arguments means the thing you came here for.
if "%~1"=="" (
    python "%~dp0tools\build_designs.py" build
) else (
    python "%~dp0tools\build_designs.py" %*
)
exit /b %ERRORLEVEL%
