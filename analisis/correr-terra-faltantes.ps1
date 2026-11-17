# Terra como reranker en las consultas del set de 30 que todavia no lo tienen (visuales y ambiguas).
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-terra-faltantes.ps1
#
# Se puede cortar y volver a correr: cada vez pregunta a analisis/faltantes.py que consultas faltan
# y corre solo esas. Una corrida que el reranker no pudo puntuar (key vencida, sin saldo) no cuenta.
# Si fallan 2 consultas seguidas, frena: cambiar la key en .env (LLM_API_KEY) y volver a correr.
# Estimado: unos USD 0,06 y 2 minutos por consulta; 20 consultas, USD 1,2 y 40 minutos.

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V     = "c3274567371ff42b"
$R     = "py -3.10 -m poetry run"
$CFG   = "config/analisis/c4-terra.toml"
$ETIQ  = "C4.filtro100.metadata.rerank20.gpt-6\.1-terra"
$RESP  = "$R tp3-responder --variante-id $V --config $CFG"
$LOG   = "artifacts\analisis\terra-faltantes-log.txt"
$INICIO = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

try {
    Add-Type -Namespace Noche -Name Despierta -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
    [Noche.Despierta]::SetThreadExecutionState([uint32]2147483649) | Out-Null
} catch { }

function Anotar($texto) {
    $t = Get-Date -Format "HH:mm:ss"
    "[$t] $texto" | Tee-Object -FilePath $LOG -Append | Out-Host
}
function Numero($cmd) {
    $salida = (cmd /c "$cmd 2>NUL") | Select-Object -Last 1
    $n = 0.0
    if ([double]::TryParse(("$salida").Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$n)) { return $n }
    return 999.0
}

$faltan = @(cmd /c "$R python analisis/faltantes.py `"$ETIQ`" 2>NUL" | Where-Object { $_ -match '^c\d\d-' })
Anotar ("faltan {0} consultas: {1}" -f $faltan.Count, ($faltan -join " "))
if ($faltan.Count -eq 0) { Anotar "nada que correr" }

$seguidas = 0
$hechas = 0
foreach ($id in $faltan) {
    Anotar "...  terra $id"
    cmd /c "$RESP set --consulta $id 2>&1" | Tee-Object -FilePath $LOG -Append | Out-Host
    $codigo = $LASTEXITCODE
    $fallas = Numero "$R python analisis/gasto.py --fallas-ultima --solo-numero"
    if ($codigo -ne 0 -or $fallas -gt 0) {
        $seguidas++
        Anotar ("FALLO terra {0} (codigo {1}, candidatas sin puntaje {2})" -f $id, $codigo, $fallas)
        if ($seguidas -ge 2) {
            Anotar "FRENO: 2 consultas seguidas fallaron. Probable key vencida o sin saldo. Cambiar LLM_API_KEY en .env y volver a correr este script: retoma desde donde quedo."
            break
        }
    } else {
        $seguidas = 0
        $hechas++
        Anotar "OK   terra $id"
    }
}

$usd = Numero "$R python analisis/gasto.py --desde $INICIO --solo-numero"
Anotar ("consultas corridas bien: {0}; gasto de esta tanda: USD {1:N2}" -f $hechas, $usd)
$quedan = @(cmd /c "$R python analisis/faltantes.py `"$ETIQ`" 2>NUL" | Where-Object { $_ -match '^c\d\d-' })
if ($quedan.Count -gt 0) {
    Anotar ("quedan {0}: {1}. Volver a correr el script." -f $quedan.Count, ($quedan -join " "))
} else {
    Anotar "terra completo en las 30 consultas. Regenerando el analisis."
    cmd /c "$R python analisis/cascada.py 2>&1" | Tee-Object -FilePath $LOG -Append | Out-Host
    Push-Location analisis
    cmd /c "$R python figuras_deck.py 2>&1" | Tee-Object -FilePath "..\$LOG" -Append | Out-Host
    Pop-Location
}
"[FIN]" | Tee-Object -FilePath $LOG -Append | Out-Host
