# gpt-6.1-terra como reranker. NO correr hasta mirar en etapa2-log.txt cuantos tokens y USD
# gasto luna: si algun modelo gpt-6 cobra tokens de razonamiento, terra puede costar bastante mas
# que lo estimado (~USD 0,07 por consulta). Corre la prueba y despues SOLO las consultas conceptuales
# y las mixtas (20 consultas, ~USD 1,4 estimado), no las 40.
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-terra.ps1

$ErrorActionPreference = "Continue"
# Python imprime caracteres no cp1252 (alfa, tildes de verificacion): forzar UTF-8 en los hijos y en la consola
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$LOG = "artifacts\analisis\terra-log.txt"
function Paso($titulo, $cmd) {
    $t = Get-Date -Format "HH:mm:ss"
    "[$t] $titulo" | Tee-Object -FilePath $LOG -Append
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) { throw "Fallo: $titulo (codigo $LASTEXITCODE)" }
}
$R = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V --config config/analisis/c4-terra.toml"

Paso "Prueba terra c01-conceptual" "$RESP set --consulta c01-conceptual"
Read-Host "Mirar tokens y USD de la prueba. Enter para seguir, Ctrl+C para cortar"

foreach ($i in 1..10) {
    $id = "c{0:00}-conceptual" -f $i
    Paso "terra $id" "$RESP set --consulta $id"
}
Paso "terra, mixtas" "$RESP --autoria grupo-2-mixto set"
Paso "Analisis de cascada" "$R python analisis/cascada.py"
"[FIN] listo. Log en $LOG" | Tee-Object -FilePath $LOG -Append
