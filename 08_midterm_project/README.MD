# Heart Failure Prediction
`The goal is to predict heart failure rate of the patients based the available features.`

`Original data location:` [Heart failure data](https://www.openml.org/search?type=data&status=active&id=43682)

## Description about dataset:
Context Heart Disease Dataset (Most comprehensive) Content Heart disease is also known as Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year which is about 32 of all deaths globally. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other conditions. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. We have curated this dataset by combining different datasets already available independently but not combined before. W have combined them over 11 common features which makes it the largest heart disease dataset available for research purposes. The five datasets used for its curation are: Database: of instances:

Cleveland: 303 Hungarian: 294 Switzerland: 123 Long Beach VA: 200 Stalog (Heart) Data Set: 270

Total 1190 Acknowledgements The dataset is taken from three other research datasets used in different research papers. The Nature article listing heart disease database and names of popular datasets used in various heart disease research is shared below. https://www.nature.com/articles/s41597-019-0206-3 Inspiration Can you find interesting insight from the largest heart disease dataset available so far and build predictive model which can assist medical practitioners in detecting early-stage heart disease

### Attribute Information

| Variable | Description | Metrics   |
:---        |    :----:   |            ---:|
| Age | age of the patient | years |
| Sex | sex of the patient | M = Male,F = Female |
| ChestPainType | TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic | NA |
| RestingBP | resting blood pressure | mm Hg |
| Cholesterol | serum cholesterol | mm/dl |
| FastingBS | fasting blood sugar	`1 = if FastingBS > 120 mg/dl`, `0 = otherwise` | mg/dl |
|RestingECG | resting electrocardiogram results	Normal = Normal, ST = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH = showing probable or definite left ventricular hypertrophy by Estes' criteria|NA|
|MaxHR|	maximum heart rate achieved	Numeric value between 60 and 202
|ExerciseAngina|	exercise-induced angina	Y = Yes, N = No|NA|
|Oldpeak|	Oldpeak = ST	Numeric value measured in depression|NA|
|ST_Slope|	the slope of the peak exercise ST segment	Up = upsloping, Flat = flat, Down = downsloping|NA|

### Target variable

### `HeartDisease`	output class : 1 = heart disease, 0 = Normal

## Project Description

For this midterm project, a binary classification model was trained on the `heart.csv` dataset in order to predict the `HeartDisease` feature.

`5` models were trained which includes logistic regression, Decision Tree, Random Forest, Extra tress and a Gradient Boosting model.Evaluation metrics AUC score is considered as main criteria to choose the best performing model. Finally, Extra trees model was selected for the deployment based on its performance. 

The exploratory data analysis, Model building , parameter tuning and final model selection was done using Jupyter Notebook --> `notebook.ipynb`.

The model training script was exported as `train.ipynb`.

# Files

* `README.md`: Project description markdown file.
* `heart.csv`: the CSV file containing the dataset.
* `notebook.ipynb`: a Jupyter Notebook for model building and parameter tuning
* `train.ipynb`: Trained model file to export bentoml model with name tag.
* `service.py`:  File that used to call for the BentoML service.
* `bentofile.yaml`: file that used to build bentoml to get all the dependencies and creates one single deployable directory for us.





