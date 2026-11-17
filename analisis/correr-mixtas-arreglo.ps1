#  Vuelve a correr las consultas MIXTAS despues del arreglo del filtro por periodo/epoca.
#
#  Por que: `recuperador_textual.py` comparaba `year` (texto en el indice) contra enteros y
#  la rama de texto se caia con TypeError en 7 de las 10 mixtas (c01, c03, c04, c05, c08, c09,
#  c10). Esas corridas midieron "solo visual", no el filtro. Las 30 consultas no se vieron afectadas.
#
#  Correr DESPUES de que termine correr-pendientes.ps1:
#     powershell -ExecutionPolicy Bypass -File analisis\correr-mixtas-arreglo.ps1
#
#  Orden por prioridad; cada paso se saltea si no entra en el tope.
#  Estimado: C3 ~0,05 | C4 ~0,25 | v2 ~0,30 | luna ~0,10
param([double]$TopeUSD = 0.8)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$R   = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V"
$LOG = "artifacts\analisis\mixtas-arreglo-log.txt"
$RES = "artifacts\analisis\resumen-mixtas-arreglo.txt"
New-Item -ItemType Directory -Force -Path "artifacts\analisis" | Out-Null
$INICIO = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$script:resumen = New-Object System.Collections.ArrayList

try {
    Add-Type -Namespace Noche -Name Despierta -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
    [Noche.Despierta]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { }

function Anotar($estado, $titulo) {
    $t = Get-Date -Format "HH:mm:ss"
    [void]$script:resumen.Add("[$t] $estado  $titulo")
    "[$t] $estado  $titulo" | Tee-Object -FilePath $LOG -Append | Out-Host
}

function Paso($titulo, $cmd) {
    Anotar "..." $titulo
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $LOG -Append | Out-Host
    if ($LASTEXITCODE -ne 0) { Anotar "FALLO" "$titulo (codigo $LASTEXITCODE)"; return $false }
    Anotar "OK" $titulo
    return $true
}

function Numero($cmd) {
    $salida = (cmd /c "$cmd 2>NUL") | Select-Object -Last 1
    $n = 0.0
    if ([double]::TryParse(("$salida").Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$n)) { return $n }
    return 999.0
}
function GastoPropio { return Numero "$R python analisis/gasto.py --desde $INICIO --solo-numero" }
function UltimaCorrida { return Numero "$R python analisis/gasto.py --ultima --solo-numero" }
function FallasUltima { return Numero "$R python analisis/gasto.py --fallas-ultima --solo-numero" }

function Cabe($estimado, $titulo) {
    $g = GastoPropio
    if (($g + $estimado) -le $TopeUSD) { Anotar "GASTO" ("{0:N2} USD hasta ahora; '{1}' estima {2:N2}; tope {3:N2}" -f $g, $titulo, $estimado, $TopeUSD); return $true }
    Anotar "SALTEADO" ("{0} (gasto {1:N2} + estimado {2:N2} pasa el tope {3:N2})" -f $titulo, $g, $estimado, $TopeUSD)
    return $false
}

[void](Paso "Test del filtro" "$R pytest tests/test_filtro_de_metadata.py -q")
if (Cabe 0.05 "C3 mixtas") { [void](Paso "C3 + metadata mixtas" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set") }
if (Cabe 0.25 "C4 mixtas") { [void](Paso "C4 + metadata mixtas" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set") }
if (Cabe 0.30 "v2 mixtas") { [void](Paso "C4 prompt v2 mixtas" "$RESP --config config/analisis/c4-reranker-v2.toml --autoria $M set") }
if (Cabe 0.10 "luna mixtas") { [void](Paso "luna, mixtas" "$RESP --config config/analisis/c4-luna.toml --autoria $M set") }
[void](Paso "Planilla de mixtas" "$R python analisis/planilla_mixta.py")
Anotar "GASTO" ("total de este script: USD {0:N2}" -f (GastoPropio))
$script:resumen | Set-Content -Path $RES -Encoding UTF8
"[FIN] listo. Resumen en $RES" | Tee-Object -FilePath $LOG -Append | Out-Host
