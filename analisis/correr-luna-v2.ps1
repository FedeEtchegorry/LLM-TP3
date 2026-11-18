# Luna + prompt v2 (iteración final del Ej. 4): set de 30 y mixtas. Estimado ~USD 0,45 y ~1 h.
# Primero UNA consulta de prueba: sigue solo si el reranker puntuó todas las candidatas y no salió cara.
# Al final corre cascada.py y regenera las planillas de juicios con los pares nuevos.
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-luna-v2.ps1
#   ... -TopeUSD 1     # otro tope (cuenta SOLO lo que gasta este script)

param([double]$TopeUSD = 0.8)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$R   = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V"
$LOG = "artifacts\analisis\luna-v2-log.txt"
$RES = "artifacts\analisis\resumen-luna-v2.txt"
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

# Prueba de un modelo: UNA consulta. Pasa solo si el reranker puntuo TODAS las candidatas y no salio caro.
function ProbarModelo($nombre, $config, $topeDeLaPrueba) {
    if (-not (Paso "Prueba $nombre c01-conceptual" "$RESP --config $config set --consulta c01-conceptual")) { return $false }
    $fallas = FallasUltima
    $u = UltimaCorrida
    if ($fallas -gt 0) { Anotar "SALTEADO" ("{0}: {1} candidatas sin puntaje en la prueba (el modelo rechazo la llamada o no devolvio JSON); ver luna-v2-log.txt" -f $nombre, $fallas); return $false }
    if ($u -gt $topeDeLaPrueba) { Anotar "SALTEADO" ("{0}: la prueba costo USD {1:N4}, mas de {2:N2} (puede cobrar razonamiento)" -f $nombre, $u, $topeDeLaPrueba); return $false }
    Anotar "OK" ("prueba {0}: puntuo todo, USD {1:N4} por consulta" -f $nombre, $u)
    return $true
}

$CFG = "config/analisis/c4-luna-v2.toml"
if (Cabe 0.45 "luna + v2 (30 + mixtas)") {
    if (ProbarModelo "luna v2" $CFG 0.03) {
        [void](Paso "luna v2, 30 consultas" "$RESP --config $CFG set")
        [void](Paso "luna v2, mixtas" "$RESP --config $CFG --autoria $M set")
    }
}

# ---------------------------------------------------------------- cierre (no gasta)
[void](Paso "Analisis de cascada" "$R python analisis/cascada.py")
[void](Paso "Mixtas" "$R python analisis/mixtas.py")
[void](Paso "Planilla de juicios (set de 30)" "$R python analisis/planilla.py")
[void](Paso "Planilla de juicios (mixtas, pendientes)" "$R python analisis/planilla_mixta.py --pendientes")
Anotar "GASTO" ("total de este script: USD {0:N2}" -f (GastoPropio))
$script:resumen | Set-Content -Path $RES -Encoding UTF8
"[FIN] listo. Resumen en $RES, log completo en $LOG" | Tee-Object -FilePath $LOG -Append | Out-Host
