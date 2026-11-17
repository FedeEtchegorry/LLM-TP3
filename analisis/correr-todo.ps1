# Todas las corridas que faltan, en orden de prioridad, para dejar corriendo de noche.
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-todo.ps1                 # incluye la etapa 1
#   powershell -ExecutionPolicy Bypass -File analisis\correr-todo.ps1 -SinEtapa1      # si correr-etapa1.ps1 ya esta corriendo
#   ... -TopeUSD 6                                                                    # otro tope de gasto
#
# Reglas para que sea seguro desatendido:
#  - Un paso que falla NO corta el resto: queda como FALLO en el resumen y se sigue.
#  - Antes de cada bloque opcional se mide el gasto (analisis/gasto.py) y, si no entra en el
#    tope, el bloque se saltea (SALTEADO). El tope cuenta SOLO lo que gasta este script.
#  - Cada modelo nuevo (luna, terra) arranca con UNA consulta de prueba; si falla o sale mas
#    cara de lo esperado, se saltean las corridas completas de ese modelo.
#  - Mantiene la PC despierta mientras corre. No incluye los juicios de los pares nuevos.
#
# Estimado (gpt-4.1-mini medido; luna y terra extrapolados del precio de lista, SIN verificar):
#   etapa 1 ~2,8 | mixtas ~0,7 | luna ~0,35 | 40 candidatas mini ~2,3 | terra ~1,5 | luna 40 ~0,6
# Total ~8,3 si no se frena nada. Con el tope por defecto (7,5) se saltea lo ultimo que no entre.

param([switch]$SinEtapa1, [double]$TopeUSD = 7.5)

$ErrorActionPreference = "Continue"
# Python imprime caracteres no cp1252 (alfa, tildes de verificacion): forzar UTF-8 en los hijos y en la consola
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$R   = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V"
$LOG = "artifacts\analisis\noche-log.txt"
$RES = "artifacts\analisis\resumen-noche.txt"
New-Item -ItemType Directory -Force -Path "artifacts\analisis" | Out-Null
$INICIO = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
if ($SinEtapa1) { $TopeUSD = $TopeUSD - 2.6 }   # la etapa 1 ya corre aparte y gasta ~2,6
$script:resumen = New-Object System.Collections.ArrayList

# Mantener la PC despierta
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
    return 999.0   # si no se pudo medir, se asume que no entra
}

function GastoPropio { return Numero "$R python analisis/gasto.py --desde $INICIO --solo-numero" }
function UltimaCorrida { return Numero "$R python analisis/gasto.py --ultima --solo-numero" }

function Cabe($estimado, $titulo) {
    $g = GastoPropio
    if (($g + $estimado) -le $TopeUSD) { Anotar "GASTO" ("{0:N2} USD hasta ahora; '{1}' estima {2:N2}; tope {3:N2}" -f $g, $titulo, $estimado, $TopeUSD); return $true }
    Anotar "SALTEADO" ("{0} (gasto {1:N2} + estimado {2:N2} pasa el tope {3:N2})" -f $titulo, $g, $estimado, $TopeUSD)
    return $false
}

# ---------------------------------------------------------------- ETAPA 1: C3/C4 con el filtro actual
if ($SinEtapa1) {
    Anotar "..." "Esperando que termine correr-etapa1.ps1 (hasta 5 h, o 20 min sin escribir en su log)"
    $limite = (Get-Date).AddHours(5)
    while ((Get-Date) -lt $limite) {
        $l = "artifacts\analisis\etapa1-log.txt"
        if ((Test-Path $l) -and (Select-String -Path $l -Pattern "\[FIN\]" -Quiet)) { break }
        if ((Test-Path $l) -and (((Get-Date) - (Get-Item $l).LastWriteTime).TotalMinutes -gt 20)) { Anotar "AVISO" "etapa1 sin actividad hace 20 min: sigo igual"; break }
        Start-Sleep -Seconds 60
    }
    Anotar "OK" "etapa 1 terminada o abandonada"
} else {
    if (Paso "Prueba C3 c01-conceptual" "$RESP --config config/analisis/c3-filtro-metadata.toml set --consulta c01-conceptual") {
        [void](Paso "C3 + metadata, 30 consultas (1)" "$RESP --config config/analisis/c3-filtro-metadata.toml set")
        [void](Paso "C4 + metadata, 30 consultas (1)" "$RESP --config config/analisis/c4-filtro-metadata.toml set")
        [void](Paso "C3 + metadata, 30 consultas (2)" "$RESP --config config/analisis/c3-filtro-metadata.toml set")
        [void](Paso "C4 + metadata, 30 consultas (2)" "$RESP --config config/analisis/c4-filtro-metadata.toml set")
        [void](Paso "C4 prompt v2 + metadata, 30 consultas" "$RESP --config config/analisis/c4-reranker-v2.toml set")
    }
}

# ---------------------------------------------------------------- ETAPA 2: consultas mixtas
$mixtasOk = $false
if (Cabe 0.7 "mixtas C0-C4") {
    [void](Paso "Tests del set mixto" "$R pytest tests/test_set_mixto.py tests/test_set_de_consultas.py -q")
    $mixtasOk = Paso "Autorar el set mixto" "$R tp3-consultas autorar --declaracion consultas-g2-mixto.toml"
    if ($mixtasOk) {
        [void](Paso "C0 mixtas" "$RESP --config config/c0.toml --autoria $M set")
        [void](Paso "C1 mixtas" "$RESP --autoria $M set")
        [void](Paso "C2 mixtas" "$RESP --config config/c2.toml --autoria $M set")
        [void](Paso "C3 + metadata mixtas (1)" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set")
        [void](Paso "C4 + metadata mixtas (1)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set")
        [void](Paso "C3 + metadata mixtas (2)" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set")
        [void](Paso "C4 + metadata mixtas (2)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set")
    }
}

# ---------------------------------------------------------------- ETAPA 4: modelos (luna)
$lunaOk = $false
if (Cabe 0.35 "luna") {
    if (Paso "Prueba luna c01-conceptual" "$RESP --config config/analisis/c4-luna.toml set --consulta c01-conceptual") {
        $u = UltimaCorrida
        if ($u -le 0.10) {
            $lunaOk = $true
            Anotar "OK" ("prueba luna costo USD {0:N4} (esperado ~0,007)" -f $u)
            [void](Paso "luna, 30 consultas" "$RESP --config config/analisis/c4-luna.toml set")
            if ($mixtasOk) { [void](Paso "luna, mixtas" "$RESP --config config/analisis/c4-luna.toml --autoria $M set") }
        } else {
            Anotar "SALTEADO" ("luna: la prueba costo USD {0:N4}, mas de 0,10 (puede cobrar razonamiento)" -f $u)
        }
    }
}

# ---------------------------------------------------------------- ETAPA 5: 40 candidatas
if (Cabe 2.3 "40 candidatas con mini") {
    [void](Paso "C4 metadata 40 candidatas, 30 consultas" "$RESP --config config/analisis/c4-metadata-rerank40.toml set")
    if ($mixtasOk) { [void](Paso "C4 metadata 40 candidatas, mixtas" "$RESP --config config/analisis/c4-metadata-rerank40.toml --autoria $M set") }
}

# ---------------------------------------------------------------- ETAPA 4b: modelos (terra)
if (Cabe 1.5 "terra (conceptuales + mixtas)") {
    if (Paso "Prueba terra c01-conceptual" "$RESP --config config/analisis/c4-terra.toml set --consulta c01-conceptual") {
        $u = UltimaCorrida
        if ($u -le 0.25) {
            Anotar "OK" ("prueba terra costo USD {0:N4} (esperado ~0,07)" -f $u)
            foreach ($i in 1..10) {
                $id = "c{0:00}-conceptual" -f $i
                [void](Paso "terra $id" "$RESP --config config/analisis/c4-terra.toml set --consulta $id")
            }
            if ($mixtasOk) { [void](Paso "terra, mixtas" "$RESP --config config/analisis/c4-terra.toml --autoria $M set") }
        } else {
            Anotar "SALTEADO" ("terra: la prueba costo USD {0:N4}, mas de 0,25 (puede cobrar razonamiento)" -f $u)
        }
    }
}

# ---------------------------------------------------------------- ETAPA 5b: 40 candidatas con luna
if ($lunaOk -and (Cabe 0.6 "luna con 40 candidatas")) {
    [void](Paso "luna 40 candidatas, 30 consultas" "$RESP --config config/analisis/c4-luna-rerank40.toml set")
    if ($mixtasOk) { [void](Paso "luna 40 candidatas, mixtas" "$RESP --config config/analisis/c4-luna-rerank40.toml --autoria $M set") }
}

# ---------------------------------------------------------------- cierre (no gasta)
[void](Paso "Analisis de cascada" "$R python analisis/cascada.py")
[void](Paso "Ablacion C0-C4" "$R python analisis/ablacion.py")
[void](Paso "Gasto medido" "$R tp3-metricas costo")
Anotar "GASTO" ("total de este script: USD {0:N2}" -f (GastoPropio))
$script:resumen | Set-Content -Path $RES -Encoding UTF8
"[FIN] listo. Resumen en $RES, log completo en $LOG" | Tee-Object -FilePath $LOG -Append | Out-Host
