#!/bin/bash

echo "Run handin 2 Evelyn van der Kamp s2138085"


# First exercise
echo "Run the first script ..."
python3 NUR_handin2Q1.py

# Second exercise
echo "Run the second script ..."
python3 NUR_handin2Q2.py


echo "Generating the pdf"

pdflatex Handin2.tex
bibtex Handin2.aux
pdflatex Handin2.tex
pdflatex Handin2.tex
