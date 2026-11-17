# Etapa 1 y 2 (parcial): C3 y C4 con filtro de metadata sobre las 30 consultas,
# con el codigo actual del filtro, DOS veces (la segunda es la repeticion para medir ruido),
# mas el prompt v2 del reranker una vez. Costo estimado: ~USD 2,8 (gpt-4.1-mini).
#
# Uso (desde la raiz del repo, en PowerShell):
#   powershell -ExecutionPolicy Bypass -File analisis\correr-etapa1.ps1
# Si se corta, volver a correrlo: cada corrida nueva suma un ordinal, no pisa las anteriores.

$ErrorActionPreference = "Continue"
# Python imprime caracteres no cp1252 (alfa, tildes de verificacion): forzar UTF-8 en los hijos y en la consola
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$V   = "c3274567371ff42b"
$LOG = "artifacts\analisis\etapa1-log.txt"
New-Item -ItemType Directory -Force -Path "artifacts\analisis" | Out-Null

function Paso($titulo, $cmd) {
    $t = Get-Date -Format "HH:mm:ss"
    "[$t] $titulo" | Tee-Object -FilePath $LOG -Append
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) { throw "Fallo: $titulo (codigo $LASTEXITCODE)" }
}

$R = "py -3.10 -m poetry run"

# 0. Prueba barata: 1 consulta conceptual con filtro (centavos). Si falla, corta aca.
Paso "Prueba C3 c01-conceptual" "$R tp3-responder --config config/analisis/c3-filtro-metadata.toml --variante-id $V set --consulta c01-conceptual"

# 1. Base con el codigo actual del filtro
Paso "C3 + metadata, 30 consultas (corrida 1)" "$R tp3-responder --config config/analisis/c3-filtro-metadata.toml --variante-id $V set"
Paso "C4 + metadata, 30 consultas (corrida 1)" "$R tp3-responder --config config/analisis/c4-filtro-metadata.toml --variante-id $V set"

# 2. Repeticion para medir ruido del reranker
Paso "C3 + metadata, 30 consultas (corrida 2)" "$R tp3-responder --config config/analisis/c3-filtro-metadata.toml --variante-id $V set"
Paso "C4 + metadata, 30 consultas (corrida 2)" "$R tp3-responder --config config/analisis/c4-filtro-metadata.toml --variante-id $V set"

# 6. Ejercicio 3 vs 4: prompt v2 del reranker
Paso "C4 v2 + metadata, 30 consultas" "$R tp3-responder --config config/analisis/c4-reranker-v2.toml --variante-id $V set"

# Resumen (no gasta)
Paso "Analisis de cascada" "$R python analisis/cascada.py"
Paso "Ablacion C0-C4" "$R python analisis/ablacion.py"
Paso "Gasto medido" "$R tp3-metricas costo"

"[FIN] listo. Log en $LOG" | Tee-Object -FilePath $LOG -Append
