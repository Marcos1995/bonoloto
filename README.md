<!-- managed-by-telegram-cursor-bot:agent-kit -->
# Bonoloto — Lotto Optimizer V3

Optimizador de combinaciones (cobertura / simulated annealing) para Bonoloto y loterías similares.

Código principal: `JamCat61.py`.

## Setup

```bash
pip install -r requirements.txt
python JamCat61.py
```

NumPy es opcional; sin él usa la implementación estándar.

## Navegador / nube

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Marcos1995/bonoloto/main?urlpath=proxy/8501/)

La primera carga en Binder puede tardar 1–2 minutos.

```bash
streamlit run app.py
```

## Docs

- `PROJECT.md` — context for agents
- `AGENTS.md` — lean workflow
- `.cursor/rules/context-lean.mdc` — token discipline
- `.cursor/skills/review` — on-demand `/review`
