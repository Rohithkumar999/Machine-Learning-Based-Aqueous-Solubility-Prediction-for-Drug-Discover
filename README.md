# Molecular Solubility Prediction Using ESOL Dataset

## Project Overview
This project aims to predict the aqueous solubility (log mol/L) of small molecules using their chemical structures. Molecular solubility is a key property in drug discovery affecting absorption, distribution, and bioavailability. The Delaney ESOL dataset is used with machine learning techniques to build predictive models.

## Dataset
The Delaney ESOL dataset contains experimentally measured aqueous solubility values (log mol/L) and corresponding molecular SMILES strings representing chemical structures for a diverse set of compounds.

Key columns:
- `Compound ID`: Name/identifier of compound
- `measured log(solubility:mol/L)`: Experimental solubility
- `SMILES`: Molecular structure encoded as SMILES string
- `ESOL predicted log(solubility:mol/L)`: Original ESOL model predictions for comparison

## Methodology

1. **Feature Extraction**  
SMILES strings are converted into Morgan fingerprints (1024-bit vectors) to numerically represent molecular structures for machine learning.

2. **Model Training**  
A Random Forest Regressor is trained on 80% of the data to learn to predict solubility from fingerprints.

3. **Evaluation**  
Model performance is evaluated on the remaining 20% test set using Mean Squared Error (MSE) and R² score.

4. **Visualization**  
Plots of actual vs predicted solubility values and feature importances are generated to interpret model performance.

5. **Hyperparameter Tuning**  
Randomized search cross-validation is applied to tune Random Forest hyperparameters and improve model accuracy.

## Results
- The baseline Random Forest model achieves a test MSE of approximately 1.43 and an R² score of about 0.67.
- Tuned models slightly improve these metrics.
- Visualizations show good alignment of predictions with actual solubility values.
- Feature importance analysis helps identify molecular fingerprint bits most influential in predicting solubility.

## Installation and Usage

### Prerequisites
- Python 3.x  
- Libraries: `pandas`, `numpy`, `rdkit`, `scikit-learn`, `matplotlib`

Install dependencies using:
pip install pandas numpy scikit-learn matplotlib rdkit-pypi

### Usage
1. Clone this repository.  
2. Place the dataset CSV file (`delaney.csv`) in the project folder.  
3. Run the provided Python or Jupyter notebook script to execute data processing, model training, evaluation, and visualization.

Example:python molecular_solubility_prediction.py


## Future Work
- Explore alternative molecular representations such as graph neural networks.  
- Test other regression models and ensemble techniques.  
- Validate on external molecular datasets for robust generalization.  
- Further analyze important chemical features impacting solubility.

Author
[Rohith Kumar Reddipogula]
MSc Data Science Student, Berlin, Germany
Email: [rohith.reddipogula@outlook.com]
LinkedIn: [(https://www.linkedin.com/in/rohith-kumar-reddipogula-a6692030b/)]
GitHub: [https://github.com/Rohithkumar999]

---

## References
- Delaney, JS. “ESOL: estimating aqueous solubility directly from molecular structure.” Journal of Chemical Information and Computer Sciences (2004).
- RDKit: Open-source cheminformatics toolkit.
- Scikit-learn: Machine learning in Python.

