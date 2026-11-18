# Research report

`main.tex` is the editable source. `market_states_report.pdf` is the compiled report. The LaTeX source uses the eight PNG figures already tracked in `../munik/analysis/notebook_paper_v2/`, so the report builds from a clone of the complete repository without duplicate images or absolute paths.

From this folder, rebuild with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
python -m pip install pypdf Pillow
python optimize_pdf.py build\main.pdf market_states_report.pdf 75
```

The original paper is cited in the report. The stock CSVs and large generated arrays are excluded from Git; rerunning the numerical analysis requires those inputs. The guided notebook is `../notebooks/market_states_paper.ipynb`.

The optional optimization keeps text and links in the PDF while reducing the size of embedded chart images for GitHub. The unoptimized PDF remains available as `build/main.pdf` after compilation.
