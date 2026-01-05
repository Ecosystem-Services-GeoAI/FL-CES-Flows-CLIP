# FL-CES-FLOWS-CLIP  
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=plastic&logo=python&logoColor=white)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/conda-environment-43B02A?style=plastic&logo=anaconda&logoColor=white)](environment.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-4B6CB7?style=plastic)](LICENSE)
[![Project Website](https://img.shields.io/badge/View%20Project-GeoAI%20CES-00ACC1?style=plastic)](https://es-geoai.rc.ufl.edu/FL-CES-Flows-CLIP/)
[![Preprint](https://img.shields.io/badge/Preprint-EcoEvoRxiv-2E7D32?style=plastic)](https://doi.org/10.32942/X29S8C)






**Mapping Cultural Ecosystem Service Flows from Social Media Imagery with Vision–Language Models: A Zero-Shot CLIP Framework**

---

## Overview

This repository provides code and reproducible workflows for mapping **Cultural Ecosystem Services (CES)** from **Flickr images** using the open-source **Contrastive Language–Image Pretraining (CLIP) model** that integrates:

- **CLIP's visual embeddings-based Binary Random Forests (BRF) for open-set image filtering**
- **Zero-shot CLIP for CES image classification**
- **CES flow mapping as indicated by average annual Photo-User-Days (PUD) and average annual PUD/user at 1 km resolution**

The workflow leverages **Flickr imagery (2014–2019)** across the state of Florida's natural and working lands to support scalable, interpretable, and reproducible CES analysis.
<p align="center">
  <img src="assets/flowchart.tif" width="50%" alt="Pipeline chart">
</p>

---
## Interactive Web Application
<p align="center">
  <a href="https://es-geoai.rc.ufl.edu/FL-CES-Flows-CLIP/">
    https://es-geoai.rc.ufl.edu/FL-CES-Flows-CLIP/
  </a>
</p>

<p align="center">
  <img src="assets/platform.png" width="80%" alt="Web">
</p>

[![Project Website](https://img.shields.io/badge/View%20Project-GeoAI%20CES-00ACC1?style=plastic)](https://es-geoai.rc.ufl.edu/FL-CES-Flows-CLIP/)
---

## Installation & Quick Start

A fully reproducible **Conda environment** is provided via `environment.yml`.

### 1. Install Conda (Miniconda recommended)

- https://docs.conda.io/en/latest/miniconda.html

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate FL-CES-Flows-CLIP
```

### 4. Launch Jupyter Lab

```bash
jupyter lab
```

### 5. Open the main workflow notebook

```bash
notebooks/CLIP-BRF-ZS/Main.ipynb
```
## Repository Structure
```bash
FL-CES-FLOWS-CLIP/
├── notebooks/          CLIP-BRF-ZS and CES analysis workflows
├── model-outputs/      Aggregated CES and PUD outputs
├── model/              CLIP's visual embeddings-based Binary Random Forest models
├── figures/            Figures outputs
├── environment.yml     Conda environment specification
└── README.md
```
Large intermediate outputs (e.g., full-resolution CSV or GeoTIFF files) are generated during notebook execution and are not required for installation.
<!--
## Method Overview
- **Vision–Language Model** : MobileCLIP
- **Learning Paradigm**    : CLIP's visual embeddings-based Binary Random Forest + Zero-shot CLIP for CES classification
- **CES Flow Mapping**  : Average annual PUD and PUD/user at 1 km resolution
- **Study Area**           : Florida, USA
- **Time Period**          : 2014–2019
-->
## Citation

If you use this repository, please cite one or more of the following works.
**Citation files (RIS format)** — copy and import into Zotero / EndNote / Mendeley.

```bash
TY  - PREPRINT
TI  - Mapping Cultural Ecosystem Service Flows from Social Media Imagery with Vision–Language Models: A Zero-Shot CLIP Framework
AU  - Liao, Hao-Yu
AU  - Zhao, Chang
AU  - Koylu, Caglar
AU  - Cao, Haojie
AU  - Qiu, JiangXiao
AU  - Callaghan, Corey T.
AU  - Song, Jiayi
AU  - Shao, Wei
PY  - 2025
DO  - 10.32942/X29S8C
UR  - https://doi.org/10.32942/X29S8C
JO  - EcoEvoRxiv
ER  -
```
```bash
TY  - CONF
TI  - Mapping Cultural Ecosystem Services Using One-Shot In-Context Learning with Multimodal Large Language Models
AU  - Liao, Hao-Yu
AU  - Zhao, Chang
AU  - Song, Jiayi
AU  - Shao, Wei
PY  - 2025
DO  - 10.1145/3748636.3764178
UR  - https://doi.org/10.1145/3748636.3764178
JO  - Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems
VL  - 33
SP  - 1
EP  - 4
PB  - Association for Computing Machinery
CY  - Minneapolis, MN, USA
ER  -
```
<!--
* Alternative
> **Liao, H.-Y., Zhao, C.\*, Koylu, C., Cao, H.J., Qiu, J.X., Callaghan, C. T., Song, J.Y., & Shao, W. (2025).**  
> *Mapping Cultural Ecosystem Service Flows from Social Media Imagery with Vision–Language Models: A Zero-Shot CLIP Framework.*  
> EcoEvoRxiv. https://doi.org/10.32942/X29S8C

> **Liao, H.-Y., Zhao, C.\*, Song, J.Y., & Shao, W. (2025).**  
> *Mapping Cultural Ecosystem Services Using One-Shot In-Context Learning with Multimodal Large Language Models.*  
> In **Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’25)**,  
> November 3–6, 2025, Minneapolis, MN, USA. ACM, New York, NY, USA, 4 pages.  
> https://doi.org/10.1145/3748636.3764178
-->
