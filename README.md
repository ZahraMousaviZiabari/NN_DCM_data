

##  Peer Review Version
Code and data for "Hybrid Neural Network Discrete Choice Models: Collective Insights and Two New Designs" – temporary version for peer review. Full dataset and final code will be released after publication.

This repository contains the code and data supporting the paper:

**Hybrid Neural Network Discrete Choice Models: Collective Insights and Two New Designs**  
Submitted to: Transportation Research Part B
> **Pre-publication notice:** This version is provided **only for peer review**.  
> The full dataset and final code will be made publicly available **after the paper is officially published**.

## Contents
- `code/` – Scripts used for analysis and figures  
- `data/` – Datasets for reproducibility

## Usage
- Reviewers may use this code to **reproduce the results reported in the manuscript**.
1. Install dependencies
2. Run experiment scripts (.py)
3. Generate figures

Examples:

python code/swiss_metro/multipleModels-boxplot.py

python code/synthetic_correlated/multiplicative/evaluation_one_run.py

python code/synthetic_correlated/multiplicative/evaluation_multiple_runs.py

- Redistribution or citation before official publication should be avoided.
- 
## Requirements
Python 3.10
PyTorch
NumPy
Pandas
Matplotlib
biogeme

## References
We included the **TasteNet-MNL** implementation provided by the authors of

Han, Y., Pereira, F.C., Ben-Akiva, M., Zegras, C., 2022. A neural-embedded discrete choice model: Learning taste representation with strengthened interpretability. Transportation Research Part B: Methodological 163, 166–186. doi:https://doi.org/10.1016/j.trb.2022.07.001.

as well as the Swiss metro data described in

Bierlaire, M., 2018. Swissmetro. URL: http://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf.

## License
This version is licensed for **peer review purposes only**. Full license will be applied after publication.
