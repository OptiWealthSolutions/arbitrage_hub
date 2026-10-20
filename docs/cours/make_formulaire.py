"""Génère chapitres/annexe_formulaire.tex à partir des fiches « tableau blanc » des chapitres.

Le formulaire reste ainsi toujours synchronisé avec le texte : modifier une fiche dans un
chapitre puis relancer build.sh met l'annexe à jour.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = [r"\chapter{Formulaire récapitulatif}", r"\label{ann:formulaire}", "",
       r"Cette annexe rassemble les fiches \og tableau blanc\fg{} de tous les chapitres. "
       r"Elle est générée automatiquement par \texttt{docs/cours/make\_formulaire.py}.", ""]
for i in range(1, 19):
    tex = (HERE / "chapitres" / f"ch{i:02d}.tex").read_text(encoding="utf8")
    title = re.search(r"\\chapter\{(.+?)\}\s*\n", tex).group(1)
    m = re.search(r"\\begin\{whiteboard\}(?:\[[^\]]*\])?\s*\\begin\{itemize\}\[[^\]]*\](.*?)\\end\{itemize\}\s*\\end\{whiteboard\}",
                  tex, re.S)
    if not m:
        continue
    out += [rf"\section*{{Chapitre {i} — {title}}}",
            rf"\addcontentsline{{toc}}{{section}}{{Chapitre {i} — {title}}}",
            r"\begin{itemize}[leftmargin=1.2em,itemsep=0.05em]" + m.group(1).rstrip() + "\n" + r"\end{itemize}", ""]
(HERE / "chapitres" / "annexe_formulaire.tex").write_text("\n".join(out), encoding="utf8")
print("formulaire :", sum(1 for l in out if l.startswith(r"\section*")), "chapitres")
