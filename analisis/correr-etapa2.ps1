# Etapa 2: set de consultas mixtas + comparacion de modelos (luna).
# Correr DESPUES de que termine correr-etapa1.ps1 (no deben solaparse: mismo indice y misma API).
# Costo estimado: ~USD 1,1 (mixtas ~0,7 + luna ~0,3 + pruebas). terra NO esta aca: ver correr-terra.ps1.
#
#   powershell -ExecutionPolicy Bypass -File analisis\correr-etapa2.ps1

$ErrorActionPreference = "Continue"
# Python imprime caracteres no cp1252 (alfa, tildes de verificacion): forzar UTF-8 en los hijos y en la consola
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$M   = "grupo-2-mixto"
$LOG = "artifacts\analisis\etapa2-log.txt"
New-Item -ItemType Directory -Force -Path "artifacts\analisis" | Out-Null

function Paso($titulo, $cmd) {
    $t = Get-Date -Format "HH:mm:ss"
    "[$t] $titulo" | Tee-Object -FilePath $LOG -Append
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) { throw "Fallo: $titulo (codigo $LASTEXITCODE)" }
}
$R = "py -3.10 -m poetry run"
$RESP = "$R tp3-responder --variante-id $V"

# 0. Tests del cambio de validacion (no gasta) y autoria del set mixto (no gasta)
Paso "Tests del set mixto" "$R pytest tests/test_set_mixto.py tests/test_set_de_consultas.py -q"
Paso "Autorar el set mixto" "$R tp3-consultas autorar --declaracion consultas-g2-mixto.toml"

# 1. Las 5 filas de la ablacion sobre las 10 mixtas (--autoria va ANTES de `set`)
Paso "C0 mixtas" "$RESP --config config/c0.toml --autoria $M set"
Paso "C1 mixtas" "$RESP --autoria $M set"
Paso "C2 mixtas" "$RESP --config config/c2.toml --autoria $M set"
Paso "C3 + metadata mixtas (1)" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set"
Paso "C4 + metadata mixtas (1)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set"

# 2. Repeticion para ruido sobre las mixtas
Paso "C3 + metadata mixtas (2)" "$RESP --config config/analisis/c3-filtro-metadata.toml --autoria $M set"
Paso "C4 + metadata mixtas (2)" "$RESP --config config/analisis/c4-filtro-metadata.toml --autoria $M set"

# 3. Modelo luna como reranker. PRUEBA con 2 consultas: si falla (identificador, temperature, seed) corta aca.
Paso "Prueba luna c01-conceptual" "$RESP --config config/analisis/c4-luna.toml set --consulta c01-conceptual"
Paso "Prueba luna c01-visual" "$RESP --config config/analisis/c4-luna.toml set --consulta c01-visual"
Paso "luna, 30 consultas" "$RESP --config config/analisis/c4-luna.toml set"
Paso "luna, mixtas" "$RESP --config config/analisis/c4-luna.toml --autoria $M set"

Paso "Analisis de cascada" "$R python analisis/cascada.py"
Paso "Gasto medido" "$R tp3-metricas costo"
"[FIN] listo. Log en $LOG. Revisar tokens y USD de luna antes de correr terra." | Tee-Object -FilePath $LOG -Append
