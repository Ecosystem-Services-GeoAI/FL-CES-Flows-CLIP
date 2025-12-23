---

## OVERVIEW

This folder contains the Jupyter notebooks used to compute **Photo–Use-Days (PUDs)** metrics and related urban land-cover statistics for **Cultural Ecosystem Services (CES)** analysis in Florida.

**Important:** The notebooks **must be run in the specified order** as each step produces intermediate outputs required by subsequent steps.

---

## ECOMMENDED EXECUTION ORDER

**Step 1 → Step 2 → Step 3 → Step 4**

---

## STEP 1: PUD Computation

**Notebook:** `PUD_Computation_Step1.ipynb`

**Purpose:**
- Compute raw Photo–Use-Days (PUDs) values
- Aggregate CES observations into spatial units (1 km resoultion)
- Generate intermediate PUD tables and shapefiles

**Outputs:**
- PUD counts by CES category
- Intermediate spatial layers for later analysis

**Note:** This notebook **must be completed first** before running any other steps.

---

## STEP 2: NLCD Urban Percentage Calculation

**Notebooks:**
1. `NLCD_Urban_Percentage_Step2-1.ipynb`
2. `NLCD_Urban_Percentage_Step2-2.ipynb`
3. `NLCD_Urban_Percentage_Step2-3.ipynb`

**Purpose:**
- Calculate urban land-cover percentages using NLCD data
- Associate NLCD urban classes with PUD spatial units
- Prepare urban context variables for statistical analysis

**Outputs:**
- Urban percentage tables
- Spatial joins between NLCD and PUDs layers

---

## STEP 3: Statistical Analysis

**Notebook:** `Statistical_Analysis_Step3.ipynb`

**Purpose:**
- Perform statistical analysis on PUD and urban variables
- Compute summary statistics by CES category
- Generate tables for interpretation and reporting

**Outputs:**
- Summary statistics tables
- Aggregated CES–urban relationships

---

## STEP 4: Heat Map Generation

**Notebook:** `HeatMap_Step4.ipynb`

**Purpose:**
- Generate spatial heat maps of CES activity
- Visualize PUD intensity across Florida
- Produce figures for analysis and publication

**Outputs:**
- CES heat map visualizations
- Final spatial figures

---

## Notes & Best Practices

- Run notebooks **in the specified order** only
- Do **not delete intermediate outputs** between steps
- Large intermediate files are expected
- Long execution time is normal for statewide data
- **GPU is NOT required**

### If errors occur, check:
- Previous steps completed successfully
- File paths are correct and unchanged
- Required input data exists in expected locations

---
