# Salifort Motors

## Employee retention | Executive summary

Sebastián Madrigal López · Google Advanced Data Analytics capstone project

**Selected model:** reduced-input random forest. **ROC AUC:** 0.968. **Precision:** 90.5%. **Recall:** 88.0%. **F1:** 0.892.

### Issue / Problem

HR seeks to understand employee departures and identify priorities for retention research. The analysis examines workload, tenure, satisfaction and evaluations in the Salifort Motors case study.

### Response

Analyzed 11,991 records after removing 3,008 exact repeats. Compared decision trees and random forests with full and reduced inputs. Selected the reduced-input forest by four-fold CV ROC AUC and evaluated it on 2,998 test rows.

### Impact

The selected model detected 438 of 498 departures, with 46 false positives. Project count, tenure and evaluation scores provide priorities for HR follow-up. Retention effects require prospective evaluation.

### Insights / Next Steps

Review project allocation at both workload extremes, career progression and evaluation practices. Confirm predictor timing and exit type, validate on a later cohort, and pilot a support intervention with measurable employee outcomes.

The reduced-input scenario removes satisfaction and replaces exact monthly hours with a >175-hour indicator. This assesses reliance on those measurements. The full-input forest achieves the highest test ROC AUC in the comparison.

Permutation importance identifies project count, tenure and last evaluation as the leading predictors. The findings describe predictive associations.

**Limitations:** internal evaluation on an educational dataset; undocumented collection dates and predictor timing; exact-repeat removal and workload-cutoff assumptions. Generalization requires validation on a later cohort.

**Source:** HR_capstone_dataset.csv, Google Advanced Data Analytics capstone. Detailed methods and results: [README](../README.md), [metrics](metrics.json), [notebooks](../notebooks/).
