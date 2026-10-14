"""Generate the README and executive summary from saved analysis outputs."""
from pathlib import Path
import json
import csv
import html

ROOT = Path(__file__).resolve().parents[1]
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

# Figure styling is local to report generation; notebooks define their own style.
TEAL, CORAL, INK, MUTED = '#167D8D', '#DB765F', '#16324A', '#627487'


def setup_style():
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 170,
                         'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.titleweight': 'bold', 'axes.titlepad': 15})


def build():
    m = json.loads((ROOT / 'reports/metrics.json').read_text())
    s, a = m['selected_test_scores'], m['audit']
    tn, fp = m['confusion_matrix'][0]
    fn, tp = m['confusion_matrix'][1]
    with (ROOT / 'reports/tables/test_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    table = '| Inputs | Model | ROC AUC | Avg. precision | Precision | Recall | F1 | Accuracy |\n'
    table += '|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        name = r['model'] + (' **(selected)**' if r['scenario'] == 'reduced' and r['model'] == m['selected_model'] else '')
        table += f"| {r['scenario'].capitalize()} | {name} | " + ' | '.join(f'{float(r[k]):.3f}' for k in ['roc_auc','average_precision','precision','recall','f1','accuracy']) + ' |\n'

    readme = f'''# Employee Attrition at Salifort Motors: Decision Trees and Random Forests

**Sebastián Madrigal López** · Google Advanced Data Analytics capstone project

## Project Overview

This project examines employee departures in the Salifort Motors case study and develops
a classifier to support retention research. The analysis combines data preparation,
exploratory analysis, feature engineering and a comparison of decision trees and random forests.

The selected reduced-input random forest achieved **ROC AUC {s['roc_auc']:.3f}, precision
{s['precision']:.1%}, recall {s['recall']:.1%} and F1 {s['f1']:.3f}** on {m['test_rows']:,} test records.
It detected **{tp} of {m['test_departures']} recorded departures**, with **{fp} false positives**.

[Executive summary (PDF)](reports/executive_summary.pdf) ·
[Executive summary (text)](reports/executive_summary.md) ·
[Data exploration](notebooks/01_data_exploration.ipynb) ·
[Modeling and evaluation](notebooks/02_modeling_and_evaluation.ipynb)

## Business Understanding

The HR team seeks to understand patterns associated with employee departures and identify
priorities for retention research. The analysis focuses on workload, tenure, satisfaction
and evaluations. The target, `left`, identifies recorded departures.

The project follows the **PACE framework**: Plan, Analyze, Construct and Execute.
Its outputs are an evaluated classifier and evidence-based questions for HR about project
allocation, career development and employee experience.

## Data Understanding

The supplied CSV contains **{a['raw_rows']:,} rows and {a['columns']} columns**, with complete values.
Removing **{a['duplicate_rows_removed']:,} exact repeated rows** produces an analysis population
of **{a['analysis_rows']:,} records**, including **{a['analysis_departures']:,} departures
({a['analysis_departure_rate']:.1%})**. The source departure rate is 23.8%.

Deduplication treats repeated records as redundant observations. This is an analytical
assumption because employee identifiers are unavailable. Salary is measured in three ordered
categories: low, medium and high. The [data dictionary](data/README.md) documents the variables,
processing decisions and source limitations.

Detailed exploration uses training rows. Departure rates are elevated at both extremes
of project count, with different patterns across monthly hours and satisfaction levels.

![Departure rates by project count in training data](reports/figures/departures_by_projects.png)

The dataset reference is
[HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction).
Salifort Motors provides the educational business scenario.

## Modeling and Evaluation

1. Remove exact repeats and create a **75/25 stratified split**, `random_state=0`:
   {m['train_rows']:,} training rows and {m['test_rows']:,} test rows.
2. Define two feature scenarios. **Full inputs** use all nine predictors. **Reduced inputs**
   remove satisfaction and replace exact monthly hours with `overworked = hours > 175`.
3. Fit salary and department encoders within each CV training fold using pipelines.
4. Compare decision trees and random forests using **four-fold shuffled stratified CV**.
   Select hyperparameters and the primary reduced-input model by mean CV ROC AUC.
5. Evaluate the fixed candidates on the test partition alongside a majority-class baseline.
   Classification uses the default **0.5 threshold**.

The reduced scenario assesses reliance on satisfaction and exact hours. Each scenario
evaluates 12 tree configurations and 8 forest configurations, with 200 trees per forest.

### Test results

{table}
The positive class is `left=1`. ROC AUC and average precision use predicted probabilities;
precision, recall, F1 and accuracy use class predictions. The baseline predicts the majority
class and assigns the training departure rate as a constant probability.

The selected forest achieved **CV ROC AUC {m['selected_cv_roc_auc']:.3f}**. On the test partition,
it missed **{fn} departures** and produced **{fp} false positives**. The full-input forest
achieved the highest test ROC AUC, indicating additional predictive information in satisfaction
and exact hours. Complete results and search parameters are available in [reports/tables](reports/tables/).

![Confusion matrix for the selected reduced-input model](reports/figures/confusion_matrix.png)

### Interpretation

Permutation importance identifies project count, tenure, last evaluation and the workload
indicator as the selected model's leading inputs. Importance is calculated on test rows
after selection and measures the decrease in ROC AUC when an input column is shuffled.
Error bars show standard deviations across five repeats. Correlated features can share
predictive information, affecting their individual importance estimates.

![Permutation importance of the selected model](reports/figures/feature_importance.png)

### Limitations

- Evaluation is internal to the supplied educational dataset. Generalization requires a later,
  independently collected cohort.
- Collection dates, predictor timing, exit type and employee identifiers remain undocumented.
- Exact-repeat removal and the 175-hour workload cutoff are analytical assumptions.
- Structured clusters and limited provenance constrain interpretation across workforces.
- Operational use requires probability calibration, an intervention cost model and validation
  of predictor availability at the intended prediction time.

## Conclusion

The reduced-input random forest classifies recorded departures with strong internal performance.
Workload, tenure and evaluation patterns provide priorities for retention research.

Recommended next steps:

- Review project allocation at both workload extremes.
- Investigate career progression and support around tenure groups with elevated departure rates.
- Examine how evaluation practices relate to workload and employee experience.
- Confirm predictor timing and exit type, then validate on a later cohort.
- Pilot a defined support intervention and measure its effect on retention and employee outcomes.

The findings describe predictive associations. Estimating the effects of interventions requires
prospective evaluation.

## Reproduce the Analysis

Tested with **Python {m['environment']['python']}** and the packages pinned in `requirements.txt`.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute_notebooks.py
python scripts/build_reports.py
```

On Windows, use the activation script in the `.venv/Scripts` folder.
The notebook runner executes all cells in a fresh namespace for each notebook and saves their
outputs. It supports the plain Python cells used in this project.

For interactive work, install JupyterLab, select the project environment and run each notebook
from top to bottom. The verified execution path uses the supplied Python runner.

The searches use two workers. Set `n_jobs=1` in each grid search and the permutation-importance
call to reduce resource use. Reports are generated from the saved tables and metrics.

## Repository Contents

| Location | Purpose |
|---|---|
| `notebooks/01_data_exploration.ipynb` | Data quality, preparation and exploratory analysis |
| `notebooks/02_modeling_and_evaluation.ipynb` | Feature scenarios, model selection, evaluation and interpretation |
| `data/raw/` | Supplied HR dataset |
| `data/README.md` | Data dictionary and provenance |
| `reports/` | Executive summary, figures, metrics and result tables |
| `scripts/` | Notebook execution and report generation |

## Acknowledgments and References

Completed as the capstone project for the **Google Advanced Data Analytics Professional
Certificate**, using the Salifort Motors business scenario and the supplied HR dataset.

- [HR Analytics and Job Prediction dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)
- [scikit-learn: ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [scikit-learn: pipelines and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

Third-party data and course materials remain subject to their respective terms.
'''
    (ROOT / 'README.md').write_text(readme)

    sections = {
        'Issue / Problem': (
            'HR seeks to understand employee departures and identify priorities for retention research. '
            'The analysis examines workload, tenure, satisfaction and evaluations in the Salifort Motors case study.'
        ),
        'Response': (
            f'Analyzed {a["analysis_rows"]:,} records after removing {a["duplicate_rows_removed"]:,} exact repeats. '
            'Compared decision trees and random forests with full and reduced inputs. '
            f'Selected the reduced-input forest by four-fold CV ROC AUC and evaluated it on {m["test_rows"]:,} test rows.'
        ),
        'Impact': (
            f'The selected model detected {tp} of {m["test_departures"]} departures, with {fp} false positives. '
            'Project count, tenure and evaluation scores provide priorities for HR follow-up. '
            'Retention effects require prospective evaluation.'
        ),
        'Insights / Next Steps': (
            'Review project allocation at both workload extremes, career progression and evaluation practices. '
            'Confirm predictor timing and exit type, validate on a later cohort, and pilot a support '
            'intervention with measurable employee outcomes.'
        ),
    }
    summary_md = '# Salifort Motors\n\n## Employee retention | Executive summary\n\n'
    summary_md += 'Sebastián Madrigal López · Google Advanced Data Analytics capstone project\n\n'
    summary_md += (
        f"**Selected model:** reduced-input random forest. **ROC AUC:** {s['roc_auc']:.3f}. "
        f"**Precision:** {s['precision']:.1%}. **Recall:** {s['recall']:.1%}. **F1:** {s['f1']:.3f}.\n\n"
    )
    for title, body in sections.items():
        summary_md += f'### {title}\n\n{body}\n\n'
    summary_md += (
        'The reduced-input scenario removes satisfaction and replaces exact monthly hours with a '
        '>175-hour indicator. This assesses reliance on those measurements. The full-input forest '
        'achieves the highest test ROC AUC in the comparison.\n\n'
        'Permutation importance identifies project count, tenure and last evaluation as the leading '
        'predictors. The findings describe predictive associations.\n\n'
        '**Limitations:** internal evaluation on an educational dataset; undocumented collection dates '
        'and predictor timing; exact-repeat removal and workload-cutoff assumptions. Generalization '
        'requires validation on a later cohort.\n\n'
        '**Source:** HR_capstone_dataset.csv, Google Advanced Data Analytics capstone. '
        'Detailed methods and results: [README](../README.md), [metrics](metrics.json), '
        '[notebooks](../notebooks/).\n'
    )
    (ROOT / 'reports/executive_summary.md').write_text(summary_md)
    make_pdf(m, sections)
    print('Built README.md, executive_summary.md and executive_summary.pdf.')


def make_pdf(m, sections):
    # Use compact charts with readable labels in the one-page summary.
    setup_style()
    with plt.rc_context({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8}):
        rates = pd.read_csv(ROOT/'reports/tables/departures_by_projects_train.csv')
        fig, ax = plt.subplots(figsize=(4.45,2.24), layout='constrained')
        bars = ax.bar(rates.number_project.astype(str), rates.departure_rate*100,
                      color=[CORAL if v==2 or v>=5 else TEAL for v in rates.number_project], width=.62)
        for bar, row in zip(bars, rates.itertuples()):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,
                    f'{row.departure_rate:.0%}\nn={row.employees:,}',ha='center',fontsize=6.8)
        ax.set(title='Departure rate by project count', xlabel='Number of projects',
               ylabel='Employees who left (%)', ylim=(0,123), yticks=[0,50,100])
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
        fig.savefig(ROOT/'reports/figures/summary_projects.png', dpi=220, bbox_inches='tight',facecolor='white')
        plt.close(fig)
        importance=pd.read_csv(ROOT/'reports/tables/permutation_importance.csv').head(4).iloc[::-1]
        labels={'number_project':'Project count','tenure':'Tenure',
                'last_evaluation':'Last evaluation','overworked':'>175 hours/month'}
        fig,ax=plt.subplots(figsize=(4.45,2.24), layout='constrained')
        ax.barh([labels.get(f,f) for f in importance.feature],importance.importance_mean,
                xerr=importance.importance_sd, color=TEAL,height=.55,
                error_kw={'ecolor':INK,'capsize':2})
        ax.set(title='Leading predictors in the selected model',xlabel='Drop in ROC AUC after shuffling')
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlim(left=0)
        ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
        fig.savefig(ROOT/'reports/figures/summary_importance.png',dpi=220,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fontdir = Path('/usr/share/fonts/truetype/dejavu')
    regular, bold = 'Helvetica', 'Helvetica-Bold'
    if (fontdir/'DejaVuSans.ttf').exists():
        pdfmetrics.registerFont(TTFont('Portfolio', str(fontdir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('PortfolioBold', str(fontdir/'DejaVuSans-Bold.ttf')))
        pdfmetrics.registerFontFamily('Portfolio', normal='Portfolio', bold='PortfolioBold')
        regular, bold = 'Portfolio', 'PortfolioBold'
    LIGHT = '#EEF5F6'
    c = canvas.Canvas(str(ROOT/'reports/executive_summary.pdf'), pagesize=(612,792))
    c.setTitle('Salifort Motors | Employee Retention Executive Summary')
    c.setAuthor('Sebastián Madrigal López')
    c.setSubject('Google Advanced Data Analytics capstone project')

    def text(txt, x, y, width, size=9, leading=13, color=INK, is_bold=False):
        style = ParagraphStyle('body', fontName=bold if is_bold else regular,
                               fontSize=size, leading=leading, textColor=HexColor(color))
        p = Paragraph(txt, style)
        _, height = p.wrap(width, 1000)
        p.drawOn(c,x,y-height)
        return y-height

    def heading(txt,x,y,width):
        return text(txt.upper(),x,y,width,size=9,leading=12,color=TEAL,is_bold=True)-7

    c.setFillColor(HexColor(TEAL)); c.rect(0,780,612,12,fill=1,stroke=0)
    text('SALIFORT MOTORS',36,755,540,size=25,leading=29,is_bold=True)
    text('Employee retention | Executive summary',36,719,540,size=12,leading=16,color=MUTED)
    text('Sebastián Madrigal López · Google Advanced Data Analytics capstone project',36,697,540,size=8.2,leading=12,color=MUTED)
    c.setFillColor(HexColor(LIGHT)); c.roundRect(36,604,540,69,7,fill=1,stroke=0)
    s=m['selected_test_scores']
    for x,value,label in [(50,f"{s['roc_auc']:.3f}",'ROC AUC'),(186,f"{s['precision']:.1%}",'PRECISION'),
                          (322,f"{s['recall']:.1%}",'RECALL'),(458,f"{m['test_rows']:,}",'TEST ROWS')]:
        text(value,x,659,119,size=21,leading=25,is_bold=True)
        text(label,x,625,119,size=7.4,leading=10,color=MUTED,is_bold=True)
    text('Reduced-input random forest · Positive class: left · Default threshold: 0.5',36,594,540,size=7.7,leading=11,color=MUTED)

    # Two-column layout: project narrative and supporting figures.
    y=566
    for title in ['Issue / Problem','Response','Impact']:
        y=heading(title,36,y,206)
        y=text(html.escape(sections[title]),36,y,206,size=8.8,leading=12.6)-17
    c.drawImage(str(ROOT/'reports/figures/summary_projects.png'),262,395,width=320,height=171,preserveAspectRatio=True,anchor='c')
    text('Training data. Labels show departure rates and group sizes.',272,393,300,size=7.4,leading=10,color=MUTED)
    c.drawImage(str(ROOT/'reports/figures/summary_importance.png'),262,195,width=320,height=171,preserveAspectRatio=True,anchor='c')
    text('Permutation importance on test rows. Error bars: standard deviation across 5 shuffles.',272,192,300,size=7.4,leading=10,color=MUTED)
    # The two middle columns must remain above the next-steps divider.
    if y < 188:
        raise RuntimeError(f'Executive-summary narrative overflows its column (y={y:.1f}).')

    c.setStrokeColor(HexColor('#D8E4E8')); c.line(36,160,576,160)
    y=heading('Insights / Next Steps',36,147,540)
    y=text(html.escape(sections['Insights / Next Steps']),36,y,540,size=8.7,leading=12.3)
    y=text('<b>Limits:</b> Internal evaluation on an educational dataset; undocumented dates and '
           'predictor timing; repeated rows treated as redundant. Generalization requires '
           'validation on a later cohort.',36,y-9,540,size=7.6,leading=10.5,color=MUTED)
    if y<38:
        raise RuntimeError(f'Executive-summary footer overflow (y={y:.1f}).')
    text('Source: supplied HR capstone data. Methods and complete results accompany this summary.',36,29,540,size=7,leading=9,color=MUTED)
    c.showPage();c.save()


if __name__ == '__main__':
    build()
