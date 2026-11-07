# Case study: Estimating burnout from workplace and lifestyle data

## Summary

I built an end-to-end machine learning app that estimates a burnout score from 27 answers about someone's job and habits. The most important lesson was not about algorithms. My first model scored 0.94 R², and almost all of that came from columns that were measuring burnout instead of explaining it. After removing them, the honest model scores 0.72 R². I then deployed it as a FastAPI service with a web form.

**Stack:** Python, pandas, scikit-learn, FastAPI, Pydantic, Uvicorn, HTML/CSS/JavaScript.

## The problem

Burnout is usually discussed after it has happened. I wanted a model that takes everyday factors (hours, meetings, sleep, support, autonomy) and estimates burnout, so the result also shows which factors matter. The target was a numeric `burnout_score`.

## The data

An employee wellbeing dataset with [N rows] of tech workers across 10 countries, 12 job roles and 10 industries.It mixes numeric measures (hours, salary, 1 to 10 rating scores), ordered categories (company size, seniority) and nominal ones (country, job role, industry, work mode).

Data: source not documented. The dataset has [100000 rows] and 27 features plus the burnout score. its origin (real survey or synthetic) is unknown, so the results are a demonstration of the method, not evidence about real workers.

## Approach

**1. Preprocessing in one pipeline.** I grouped columns by type and gave each group its own steps inside a `ColumnTransformer`: median imputation, `log1p` and scaling for skewed columns, ordered encoding for ordinal ones and one-hot encoding for nominal ones. Keeping everything in one `Pipeline` means the API can pass raw values straight in with no separate preprocessing code.

**2. Catching target leakage.** My first feature list included `burnout_level`, which is the target cut into bins. I reviewed a correlation heatmap and ran a series of experiments, removing one group of columns at a time:

| Experiment | Test R² | Test MAE |
|---|---|---|
| All features including `burnout_level` | 0.941 | 0.514 |
| Without `burnout_level` | 0.880 | 0.742 |
| Also without `stress_score` | 0.820 | 0.911 |
| Also without `phq9_score`, `gad7_score` | 0.796 | 0.970 |
| Also without `phq9_category`, `gad7_category` | 0.723 | 1.130 |

The last step was a surprise. After removing the depression and anxiety scores, the model still scored 0.80, because their binned category columns still carried the same signal. Removing those too cost another 0.07. A lesson I took from this: dropping a column is not enough when a transformed copy of it is still in the data.

**3. Deciding what to keep.** `stress_score` correlated 0.85 with burnout, and the PHQ-9 and GAD-7 scores 0.71. They are symptoms that move with burnout, so they would make the model look better without explaining anything. I also dropped `age` because it correlated 0.97 with `years_experience`, and removing either one changed the score only in the sixth decimal place.

**4. Choosing the model.** I compared a random forest with the linear model. The untuned forest overfit badly (train 0.955, test 0.677). After tuning, the gap closed but it fell below the linear model (test 0.637). Five-fold cross-validation confirmed the result: Linear 0.723, Ridge 0.723, tuned forest 0.640, with small variation between folds (about 0.003). The relationships in this data look mostly linear, so I chose the simpler, more interpretable linear regression.

## What the data shows

The strongest relationships with burnout were work hours and meetings per day (both about +0.47), then sleep (-0.34) and vacation days taken (-0.29). More hours and meetings go with higher burnout, and more sleep and time off go with lower burnout. Support from managers, friends and family, and exercise showed milder negative relationships. These are correlations, not proven causes.

## Deployment

- **API:** a FastAPI service with a `POST /predict` endpoint. Pydantic validates all 27 inputs (allowed categories and numeric ranges), so a bad request is rejected with a clear 422 error before it reaches the model.
- **UI:** a single HTML page with grouped form sections, sliders for rating scales and a gauge that shows the estimate. It is served by the API at `/ui`.
- **Debugging along the way:** bugs I fixed included a function used where a class was needed, `=` instead of `:` in type annotations (which silently made fields optional), and column names that did not match the ones the model was trained on. The last one taught me to build the input row from the model's own `feature_names_in_` list.

## Limitations

- R² of 0.72 leaves about 28% of the variation unexplained, and the typical error is about one point.
- The model shows associations in this dataset, not causes, and may not generalize to people or workplaces unlike those in the data.
- It is an estimate, not a diagnosis, and should not be used to make decisions about individuals.
- The saved model only loads reliably on the scikit-learn version it was created with (1.9.1).

## What I would do next

- Show per-answer contributions from the model's coefficients, so the UI explains why a score is high or low.
- Check the model against data from a different source, to test how well it generalizes.
- Add a calibrated explanation of the score scale and bands.
- Containerize the app and add tests for the API.
