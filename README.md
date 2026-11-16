<p align="center">
  <img src="assets/truthguard-banner.png" alt="TruthGuard AI — AI x Epistemics Research Sprint 2026" width="100%">
</p>

﻿# TruthGuard AI â€” Research Reproducibility Repository (PREPUBLICATION)

**Estado:** copia de preparacion; no publicar hasta revisar `_publication_audit/STATUS.txt`.

TruthGuard AI estudia cambios injustificados de respuesta ante presion directa (B)
y autoridad simulada (C), con control neutral (A), en preguntas numericas.

## Resultados exploratorios
- 7 modelos x 20 preguntas x 3 condiciones = 420 conversaciones completadas.
- B: 4/140 cambios injustificados; C: 0/140.
- McNemar exacto bilateral: p = 0.125 (no evidencia suficiente para concluir diferencia).
- Auditoria: 96 adjudicaciones asistidas, 42 revisiones de muestra adicional;
  permanecen clasificaciones automaticas no individualmente verificadas.
- No constituye un ranking general de robustez ni una prueba de actualizacion
  adecuada ante evidencia legitima (no se ejecuto condicion D).

## Empezar
- Notebook guiado: `notebooks/TruthGuard_AI_Research_Colab_V1_1_GUIADA.ipynb`.
- Dataset: `data/benchmark_v1.json`.
- Protocolo: `protocol/protocol_v1.json`.
- Resultados originales: `results/seven_models_v2_4096.jsonl`.
- Estadistica: `reports/scientific_v5_stats/statistical_report.md`.
- Para ejecutar pruebas locales: `python -m unittest discover -s tests -v`.

## Antes de publicar
1. Revisar alertas de secretos y contenido sensible en todo el repositorio.
2. Revisar reportes y hashes contra fuente original.
3. Confirmar que tests y notebook funcionan tras copiarse.
4. Agregar licencia elegida explicitamente por el autor y referencias cientificas.
5. Agregar informe final y versionar dependencias verificadas.

**No introducir API keys ni credenciales en el notebook, scripts o reportes.**
