# Lo que falta despues de la noche del 15 al 16. Tope pensado para el piso de USD 15:
# el panel marca 9,13 gastados -> quedan ~5,9; con colchon de ~1 para juicios y lo no medido, tope 4,5.
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-pendientes.ps1
#   ... -TopeUSD 3     # otro tope (cuenta SOLO lo que gasta este script)
#
# Que cambio respecto de la noche anterior:
#  - Los modelos gpt-6 fallaban con "max_tokens not supported": el puerto LLM ahora se adapta
#    (max_completion_tokens, sin temperature/seed si no los acepta) y lo deja en la corrida.
#  - La prueba de cada modelo ahora exige DOS cosas: que no salga cara y que NO falle ningun
#    puntaje del reranker. Antes solo miraba el costo, y una corrida rota sale barata.
#  - El set mixto se autora con --derivar-otro (lo exige la clase 3 al haber otro set congelado).
#  - C3 es determinista (3 repeticiones dieron lo mismo): no se repite. C4 si.
#
# Estimado: mixtas ~0,7 | luna 30+mixtas ~0,35 | luna 40 candidatas ~0,6 | v2 mixtas ~0,3
#           | 40 candidatas mini mixtas ~0,55 | terra conceptuales+mixtas ~1,5 (sin verificar)

param([double]$TopeUSD = 4.5)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$R   = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V"
$LOG = "artifacts\analisis\pendientes-log.txt"
$RES = "artifacts\analisis\resumen-pendientes.txt"
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
    if ($fallas -gt 0) { Anotar "SALTEADO" ("{0}: {1} candidatas sin puntaje en la prueba (el modelo rechazo la llamada o no devolvio JSON); ver pendientes-log.txt" -f $nombre, $fallas); return $false }
    if ($u -gt $topeDeLaPrueba) { Anotar "SALTEADO" ("{0}: la prueba costo USD {1:N4}, mas de {2:N2} (puede cobrar razonamiento)" -f $nombre, $u, $topeDeLaPrueba); return $false }
    Anotar "OK" ("prueba {0}: puntuo todo, USD {1:N4} por consulta" -f $nombre, $u)
    return $true
}

# ---------------------------------------------------------------- 1. consultas mixtas (lo mas importante)
$mixtasOk = $false
if (Cabe 0.7 "mixtas C0-C4") {
    [void](Paso "Tests" "$R pytest tests/test_set_mixto.py tests/test_set_de_consultas.py tests/test_llm_ajustes.py -q")
    $mixtasOk = Paso "Autorar el set mixto" "$R tp3-consultas autorar --declaracion consultas-g2-mixto.toml --derivar-otro"
    if ($mixtasOk) {
        [void](Paso "C0 mixtas" "$RESP --config config/c0.toml --autoria $M set")
        [void](Paso "C1 mixtas" "$RESP --autoria $M set")
        [void](Paso "C2 mixtas" "$RESP --config config/c2.toml --autoria $M set")
        [void](Paso "C3 + metadata mixtas" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set")
        [void](Paso "C4 + metadata mixtas (1)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set")
        [void](Paso "C4 + metadata mixtas (2)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set")
    }
}

# ---------------------------------------------------------------- 2. modelos: luna
$lunaOk = $false
if (Cabe 0.35 "luna") {
    if (ProbarModelo "luna" "config/analisis/c4-luna.toml" 0.10) {
        $lunaOk = $true
        [void](Paso "luna, 30 consultas" "$RESP --config config/analisis/c4-luna.toml set")
        if ($mixtasOk) { [void](Paso "luna, mixtas" "$RESP --config config/analisis/c4-luna.toml --autoria $M set") }
    }
}

# ---------------------------------------------------------------- 3. prompt v2 sobre mixtas (Ej. 3 vs Ej. 4)
if ($mixtasOk -and (Cabe 0.3 "v2 mixtas")) {
    [void](Paso "C4 prompt v2 mixtas" "$RESP --config config/analisis/c4-reranker-v2.toml --autoria $M set")
}

# ---------------------------------------------------------------- 4. 40 candidatas
if ($lunaOk -and (Cabe 0.6 "luna con 40 candidatas")) {
    [void](Paso "luna 40 candidatas, 30 consultas" "$RESP --config config/analisis/c4-luna-rerank40.toml set")
    if ($mixtasOk) { [void](Paso "luna 40 candidatas, mixtas" "$RESP --config config/analisis/c4-luna-rerank40.toml --autoria $M set") }
}
if ($mixtasOk -and (Cabe 0.55 "mini con 40 candidatas, mixtas")) {
    [void](Paso "mini 40 candidatas, mixtas" "$RESP --config config/analisis/c4-metadata-rerank40.toml --autoria $M set")
}

# ---------------------------------------------------------------- 5. terra (el mas incierto y caro, al final)
if (Cabe 1.5 "terra (conceptuales + mixtas)") {
    if (ProbarModelo "terra" "config/analisis/c4-terra.toml" 0.25) {
        foreach ($i in 1..10) {
            $id = "c{0:00}-conceptual" -f $i
            [void](Paso "terra $id" "$RESP --config config/analisis/c4-terra.toml set --consulta $id")
        }
        if ($mixtasOk) { [void](Paso "terra, mixtas" "$RESP --config config/analisis/c4-terra.toml --autoria $M set") }
    }
}

# ---------------------------------------------------------------- cierre (no gasta)
[void](Paso "Analisis de cascada" "$R python analisis/cascada.py")
[void](Paso "Ruido entre repeticiones" "$R python analisis/ruido.py")
[void](Paso "Ablacion C0-C4" "$R python analisis/ablacion.py")
[void](Paso "Gasto medido" "$R tp3-metricas --variante-id $V costo")
Anotar "GASTO" ("total de este script: USD {0:N2}" -f (GastoPropio))
$script:resumen | Set-Content -Path $RES -Encoding UTF8
"[FIN] listo. Resumen en $RES, log completo en $LOG" | Tee-Object -FilePath $LOG -Append | Out-Host
