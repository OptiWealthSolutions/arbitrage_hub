#!/bin/sh
# Compile le cours complet (3 passes + bibliographie)
set -e
cd "$(dirname "$0")"
N=arbitrage_hub_cours_complet
python3 make_formulaire.py
pdflatex -interaction=nonstopmode $N.tex > /dev/null || true
bibtex $N > /dev/null || true
pdflatex -interaction=nonstopmode $N.tex > /dev/null || true
pdflatex -interaction=nonstopmode $N.tex | grep -E "^!|Output written|undefined" || true
grep -E "^! " -A3 $N.log | head -20 || true
