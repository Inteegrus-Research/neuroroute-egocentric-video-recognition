# LaTeX insertion guide

The generated figures are vector PDF plus 300-dpi PNG. The generated tables use conservative `booktabs` syntax and ordinary LaTeX constructs.

## Recommended insertion pattern

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig01_study_framework.pdf}
    \caption{Overview of the synchronized EgoBrain study and training-only neural supervision pathway.}
    \label{fig:framework}
\end{figure}
```

For tables, copy or `\input{}` the generated `.tex` files, e.g.

```latex
\input{tables/table02_visual_baselines.tex}
```

## Springer / LNEE note

Do **not** invent or override the Springer class, page dimensions, fonts, or margins in the manuscript. Springer Nature states that proceedings papers should use the publisher's proceedings templates and that the conference determines the maximum paper length. Use the exact IRIS author template / Springer proceedings package supplied by the conference for the final manuscript. The artifacts generated here are deliberately style-light so they can be inserted into that official template without changing its typography.
