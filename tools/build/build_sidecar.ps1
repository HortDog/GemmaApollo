# Freeze the app-tier sidecar for the Electron desktop app (stage 6).
#
# A DEDICATED venv without the s2l extra keeps torch/faster-whisper out of
# the bundle — the app tier has been torch-free since the ONNX VAD swap.
# Output: tools/build/dist/scribe-server/scribe-server.exe (+ _internal/),
# which desktop/package.json packs as an electron-builder extraResource.
#
# NOTE: native tools (uv, pyinstaller) write progress to stderr — do not
# set $ErrorActionPreference = "Stop" here (PS 5.1 turns that into fake
# failures); exit codes are checked explicitly instead.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

function Step($desc, $block) {
    Write-Host "==> $desc"
    & $block
    if ($LASTEXITCODE -ne 0) { Write-Error "$desc failed ($LASTEXITCODE)"; exit 1 }
}

$venv = "tools/build/.venv-freeze"
Step "create freeze venv"    { uv venv $venv --python 3.12 }
Step "install app tier"      { uv pip install --python "$venv/Scripts/python.exe" ".[audio]" pyinstaller }
Step "pyinstaller"           { & "$venv/Scripts/pyinstaller.exe" tools/build/scribe-server.spec `
                                   --noconfirm --distpath tools/build/dist --workpath tools/build/work }

Write-Host ""
Write-Host "frozen sidecar: tools/build/dist/scribe-server/scribe-server.exe"
