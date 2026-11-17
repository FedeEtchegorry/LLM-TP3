# Mide el prefiltro de pedidos vagos (gasta poco: 30 + 12 llamadas cortas del aclarador).
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-prefiltro.ps1
#
# 1. aclarar.py sugerir: clasifica las 30 consultas del set y escribe aclaraciones-v3.json.
# 2. aclarar.py validar: clasifica 12 pedidos fuera del set con nivel esperado (validacion-v3.json).
# 3. figuras_deck.py: arma prefiltro.png.

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$R   = "py -3.10 -m poetry run"
$LOG = "artifacts\analisis\prefiltro-log.txt"
$INICIO = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

function Paso($titulo, $cmd) {
    $t = Get-Date -Format "HH:mm:ss"
    "[$t] ...  $titulo" | Tee-Object -FilePath $LOG -Append | Out-Host
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $LOG -Append | Out-Host
    $t = Get-Date -Format "HH:mm:ss"
    if ($LASTEXITCODE -ne 0) { "[$t] FALLO $titulo (codigo $LASTEXITCODE)" | Tee-Object -FilePath $LOG -Append | Out-Host; return $false }
    "[$t] OK   $titulo" | Tee-Object -FilePath $LOG -Append | Out-Host
    return $true
}

[void](Paso "Prefiltro sobre las 30 consultas" "$R python analisis/aclarar.py sugerir")
[void](Paso "Prefiltro sobre los 12 pedidos de validacion" "$R python analisis/aclarar.py validar")
Push-Location analisis
[void](Paso "Figuras" "$R python figuras_deck.py")
Pop-Location
"[FIN] Log en $LOG" | Tee-Object -FilePath $LOG -Append | Out-Host
