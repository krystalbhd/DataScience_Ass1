# Run from the folder containing data.csv.

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

DATA = Path('data.csv')
OUTPUT = Path('outputs')
OUTPUT.mkdir(exist_ok=True)
SEED = 401478
SAMPLE_SIZE = 36
ALPHA = 0.05
BENCHMARK = 50.0
raw = pd.read_csv(DATA)
print('Rows and columns:', raw.shape)
print('Missing values:\n', raw.isna().sum().to_string())
print('Exact duplicate rows:', raw.duplicated().sum())
print('Python:', sys.version.split()[0], 'pandas:', pd.__version__, 'SciPy:', scipy.__version__)

clean = raw.copy()
clean.columns = clean.columns.str.strip()
for column in clean.select_dtypes(include='object'):
    clean[column] = clean[column].str.strip()
clean = clean.drop_duplicates().copy()
clean['Team'] = clean['Squad'].str.replace(r'^[a-z]+\s+', '', regex=True)
numeric_columns = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
for column in numeric_columns:
    clean[column] = pd.to_numeric(clean[column], errors='coerce')
assert clean[numeric_columns].notna().all().all(), 'Inspect invalid numeric data'
assert clean['Team'].is_unique, 'Resolve repeated teams before analysis'
assert (clean['MP'] == clean['W'] + clean['D'] + clean['L']).all()
assert (clean['GD'] == clean['GF'] - clean['GA']).all()
assert (clean['Pts'] == 3 * clean['W'] + clean['D']).all()
assert (clean[['MP','W','D','L','GF','GA','Pts']] >= 0).all().all()
assert (clean[numeric_columns] % 1 == 0).all().all()
clean['Leading_scorer_goals'] = pd.to_numeric(
    clean['Top Team Scorer'].str.extract(r' - (\d+)\s*$', expand=False), errors='coerce')
excluded = clean.loc[clean['GF'] == 0, ['Team', 'GF', 'Top Team Scorer']]
eligible = clean.loc[clean['GF'] > 0].copy()
assert eligible['Leading_scorer_goals'].notna().all(), 'Resolve unparsed scorer records'
assert eligible['Leading_scorer_goals'].between(1, eligible['GF']).all()
eligible['Scorer_share_pct'] = 100 * eligible['Leading_scorer_goals'] / eligible['GF']
print('Excluded because GF = 0:\n', excluded.to_string(index=False))
print('Eligible teams:', len(eligible))
print('Internal checks passed. Total GF / GA:', clean.GF.sum(), '/', clean.GA.sum())
eligible.to_csv(OUTPUT / 'eligible_teams.csv', index=False)

sample = eligible.sample(n=SAMPLE_SIZE, replace=False, random_state=SEED).copy()
sample = sample.sort_values('Team').reset_index(drop=True)
x = sample['Scorer_share_pct'].to_numpy()
sample[['Team','GF','Leading_scorer_goals','Scorer_share_pct']].to_csv(
    OUTPUT / 'analysis_sample.csv', index=False)
print(sample[['Team','GF','Leading_scorer_goals','Scorer_share_pct']].round(3).to_string(index=False))
print('Sample size:', len(x), 'of', len(eligible))
print('Full-frame mean (known, descriptive):', round(eligible.Scorer_share_pct.mean(), 4), '%')

n = len(x)
mean = float(np.mean(x))
sd = float(np.std(x, ddof=1))
se = sd / np.sqrt(n)
q1, median, q3 = np.percentile(x, [25, 50, 75])
summary = pd.Series({'n':n, 'Mean (%)':mean, 'Median (%)':median,
                     'Sample SD (percentage points)':sd, 'Minimum (%)':x.min(),
                     'Q1 (%)':q1, 'Q3 (%)':q3, 'Maximum (%)':x.max(),
                     'IQR (percentage points)':q3-q1})
print(summary.round(4).to_string())
summary.to_csv(OUTPUT / 'descriptive_statistics.csv', header=['Value'])
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].hist(x, bins=np.arange(0, 111, 10), color='#211645', edgecolor='white')
axes[0].axvline(50, color='#A17C15', linestyle='--', label='50% benchmark')
axes[0].set(xlabel='Leading scorer share (%)', ylabel='Number of teams', title='Sample distribution', xlim=(0,105))
axes[0].legend()
axes[1].boxplot(x, vert=False, patch_artist=True, boxprops={'facecolor':'#D8CEE7'})
axes[1].scatter(x, np.ones(n), alpha=.5, color='#211645', s=25)
axes[1].axvline(50, color='#A17C15', linestyle='--')
axes[1].set(xlabel='Leading scorer share (%)', yticks=[], title='Spread and individual team values', xlim=(0,105))
fig.suptitle('Scoring concentration | Random sample of 36 teams', fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT / 'distribution.png', dpi=180, bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(6,4.5))
stats.probplot(x, dist='norm', plot=ax)
ax.set(title='Normal Q–Q plot: assess the t-test approximation', ylabel='Ordered scorer shares (%)')
fig.tight_layout()
fig.savefig(OUTPUT / 'qq_plot.png', dpi=180, bbox_inches='tight')
plt.show()
print('Sample skewness:', round(stats.skew(x, bias=False), 3))
print('Teams at 100%:', sample.loc[sample.Scorer_share_pct.eq(100), 'Team'].tolist())

df = n - 1
t_critical = stats.t.ppf(1-ALPHA/2, df)
ci_low, ci_high = stats.t.interval(1-ALPHA, df, loc=mean, scale=se)
print(f'Mean = {mean:.4f}%; SE = {se:.4f} percentage points')
print(f't critical = {t_critical:.4f}; df = {df}')
print(f'95% t confidence interval: [{ci_low:.4f}%, {ci_high:.4f}%]')
fig, ax = plt.subplots(figsize=(8, 3))
ax.errorbar(mean, 0, xerr=[[mean-ci_low],[ci_high-mean]], fmt='o', color='#211645', capsize=8, markersize=9)
ax.axvline(BENCHMARK, color='#A17C15', linestyle='--', label='50% benchmark')
ax.set(xlabel='Mean leading scorer share (%)', yticks=[], xlim=(30,60),
       title='Sample mean and 95% model-based t confidence interval')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(OUTPUT / 'confidence_interval.png', dpi=180, bbox_inches='tight')
plt.show()

test = stats.ttest_1samp(x, popmean=BENCHMARK, alternative='two-sided')
t_manual = (mean-BENCHMARK)/se
assert np.isclose(test.statistic, t_manual)
reject = test.pvalue < ALPHA
assert reject == (BENCHMARK < ci_low or BENCHMARK > ci_high)
results = {'n':n, 'mean_pct':mean, 'sd_pp':sd, 'se_pp':se,
           'ci_lower_pct':float(ci_low), 'ci_upper_pct':float(ci_high),
           'difference_from_50_pp':mean-BENCHMARK,
           't_statistic':float(test.statistic), 'df':df,
           'p_value_two_sided':float(test.pvalue), 'alpha':ALPHA,
           'reject_null':bool(reject), 'cohens_d':(mean-BENCHMARK)/sd}
pd.Series(results).to_csv(OUTPUT / 'test_results.csv', header=['Value'])
print(f't({df}) = {test.statistic:.4f}; two-sided p = {test.pvalue:.6f}')
print(f'Mean difference = {mean-BENCHMARK:.4f} percentage points')
print(f'One-sample Cohen d = {(mean-BENCHMARK)/sd:.4f}')
print('Decision:', 'Reject H0' if reject else 'Fail to reject H0')

N = len(eligible)
sampling_fraction = n / N
fpc = np.sqrt(1 - sampling_fraction)
finite_se = se * fpc
finite_ci = stats.t.interval(0.95, n-1, loc=mean, scale=finite_se)
print(f'Sampling fraction: {sampling_fraction:.4f}')
print(f'Finite-population correction: {fpc:.6f}')
print(f'Adjusted estimated SE: {finite_se:.4f} percentage points')
print(f'Approximate finite-frame t interval: {finite_ci[0]:.4f}% to {finite_ci[1]:.4f}%')
print(f'Known full-frame mean: {eligible.Scorer_share_pct.mean():.4f}%')
pd.Series({'N':N,'n':n,'sampling_fraction':sampling_fraction,'fpc':fpc,
           'adjusted_se_pp':finite_se,'approx_lower_pct':finite_ci[0],
           'approx_upper_pct':finite_ci[1],
           'known_frame_mean_pct':eligible.Scorer_share_pct.mean()}).to_csv(
    OUTPUT / 'finite_population_sensitivity.csv', header=['Value'])

PURPLE = '#211645'
GOLD = '#A17C15'
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
axes[0].boxplot(x, patch_artist=True, boxprops={'facecolor':'#D8CEE7'},
                medianprops={'color':GOLD,'linewidth':2})
axes[0].set(xticks=[1],xticklabels=['Sample (n = 36)'],ylabel='Leading scorer share (%)',ylim=(0,105),title='Spread of team contributions')
axes[0].axhline(50,color=GOLD,linestyle='--',linewidth=1)
axes[1].hist(x,bins=np.arange(0,111,10),color=PURPLE,edgecolor='white')
axes[1].axvline(50,color=GOLD,linestyle='--',linewidth=2,label='50% benchmark')
axes[1].set(xlim=(0,105),xlabel='Leading scorer share (%)',ylabel='Number of teams',title='Sample distribution')
axes[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT / 'slide_distribution.png',dpi=200,bbox_inches='tight'); plt.show()

fig,ax=plt.subplots(figsize=(10,2.7))
ax.errorbar(mean,0,xerr=[[mean-ci_low],[ci_high-mean]],fmt='o',color=PURPLE,elinewidth=5,capsize=10,markersize=10)
ax.axvline(50,color=GOLD,linestyle='--',linewidth=1.5)
ax.text(50,.67,'50% benchmark',ha='center',color=GOLD)
ax.text(ci_low,.3,f'{ci_low:.2f}%',ha='center',color=PURPLE,fontweight='bold')
ax.text(ci_high,.3,f'{ci_high:.2f}%',ha='center',color=PURPLE,fontweight='bold')
ax.text(mean,-.32,f'Mean {mean:.2f}%',ha='center',color=PURPLE,fontweight='bold')
ax.set(xlim=(30,60),ylim=(-.7,1),yticks=[],xlabel='Mean leading scorer share (%)')
ax.spines['left'].set_visible(False)
fig.tight_layout();fig.savefig(OUTPUT / 'slide_ci.png',dpi=200,bbox_inches='tight');plt.show()

fig,ax=plt.subplots(figsize=(7,3.5))
ordered=np.sort(eligible.Scorer_share_pct.to_numpy())
ax.scatter(np.arange(1,len(ordered)+1),ordered,color=PURPLE,s=30)
ax.axhline(50,color=GOLD,linestyle='--',label='50% benchmark')
ax.axhline(eligible.Scorer_share_pct.mean(),color='#9775B6',label='Known mean: 45.45%')
ax.set(xlabel='Eligible teams ordered by scorer share',ylabel='Leading scorer share (%)',ylim=(0,105),title='All 47 eligible teams in the supplied file')
ax.legend(fontsize=9,loc='upper left')
fig.tight_layout();fig.savefig(OUTPUT / 'slide_full_frame.png',dpi=200,bbox_inches='tight');plt.show()