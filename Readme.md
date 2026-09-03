# Run Instructions

## 1) Install catboost and scikit learn

The two packages should already be in `.venv`, but if for whatever reason they aren't install them.

```bash
pip install catboost scikit-learn
```

## 2) `dataCleaning.py`

`dataCleaning.py` contains the class with the data cleaning and data checking methods I created to check and clean the provided data. There are data checking that I have opted not to do due to the fact that I think it goes beyond the scope of the assessment (meaning it'd take considerable time by my estimation), such as checking the correlation between the distance and the straight absolute distance taken from the coordinates.

## 3) `training.py`

`training.py` is straigtforward, I applied data cleaning before inputting the dataset into the catboost model. If you want to train different models of different depths, just included the depths to the depths list. Hit run and it should be good to go, saving the model in the directory of the `training.py`.

## 4) `predict.py`

`predict.py` is also straigthforward, I did a similar cleaning before prediction. You would notice the commented data checking. I did this to check if there was anything amiss with the dataset for prediction. Hit run and it should generate three files: `validation.csv`, `validation-predictions.csv`, and `december-chart-inputs.csv`.

## 5) `TemplatesCSV`

I made sure to place the "template" of the three files in a subfolder called `TemplatesCSV`. I did that so I can save my own files without changing the template files (and potentially messing up the whole thing) into the current directory.

---

That's all I think. Also, I'd like to say I'm truly grateful for this opportunity, thank you for giving me this assessment.



# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
