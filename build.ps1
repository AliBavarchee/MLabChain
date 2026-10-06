$ErrorActionPreference = "Stop"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is required."
}

cmake -S cpp -B build
cmake --build build --config Release

Write-Host "Built mera_core."
