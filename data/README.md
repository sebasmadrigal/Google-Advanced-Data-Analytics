# Data dictionary and provenance

`raw/HR_capstone_dataset.csv` is the dataset supplied for the Google Advanced Data Analytics
capstone project. The source reference is
[HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).
Salifort Motors provides the educational business scenario. Collection dates, geography,
employer provenance, exit type and predictor measurement timing remain undocumented.

The dataset remains subject to its source terms. A dataset-specific license was absent
from the supplied materials; public redistribution requires checking the source's terms.

| Source column | Analysis name | Meaning / units |
|---|---|---|
| `satisfaction_level` | `satisfaction_level` | Reported satisfaction score, 0–1 |
| `last_evaluation` | `last_evaluation` | Last evaluation score, 0–1 |
| `number_project` | `number_project` | Number of projects |
| `average_montly_hours` | `average_monthly_hours` | Average monthly hours |
| `time_spend_company` | `tenure` | Years with company |
| `Work_accident` | `work_accident` | Work accident indicator, 0/1 |
| `left` | `left` | Departure indicator: 1 = left, 0 = stayed |
| `promotion_last_5years` | `promotion_last_5years` | Promotion in the past five years, 0/1 |
| `Department` | `department` | Department category |
| `salary` | `salary` | Ordered category: low, medium or high |

## Feature engineering

The reduced-input scenario excludes satisfaction and replaces monthly hours with
`overworked = average_monthly_hours > 175`. The 175-hour cutoff is a fixed analytical
assumption used to compare feature scenarios.

## Processing

- Source dimensions: 14,999 rows and 10 columns, with complete values.
- Remove 3,008 exact repeated rows before splitting, keeping the first occurrence.
- Analysis population: 11,991 rows, including 1,991 departures (16.6%).
- Deduplication treats exact repeats as redundant observations. Verifying identity requires
  employee identifiers.
- Preserve source row indices in `reports/tables/split_membership.csv`.
- Retain valid long-tenure and high-workload observations.
- Source SHA-256: `af8c4cede39f28b5a67c748a66aa850fe260f908cbeaa226694121e0a9a4e105`.

## Interpretation

The data support analysis of predictive associations. Structured clusters and limited
provenance constrain generalization. Prospective use requires verifying predictor availability,
defining a prediction horizon and validating performance on a later cohort.
