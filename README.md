# kmsurvival


kmsurvial is an implementation of Kaplan-Meier (KM) survival estimation in Python. It's a practical program for comparing survial probabilities qualitatively among groups. And it's also small, fast, and easy to use.

The reason for writing a new KM estimator is that some features I want are not available or flexible in other implementations as of early 2016.

### Features

* Differentiate between the snapshot date and cutoff dates of a data set.
* Support hierarchical strata.
* Flexible combination of groups for comparisions.
* Support multiple data input formats.
* Users can easily get hazards and survival functions which can be piped into visualziaiton or further data processing.

### Installation

kmsurvival can be installed with the following command:

```
pip install kmsurvival
```

### Examples

```python
import pandas as pd
from kmsurvival import KMSurvival, plot_right_censor

df = pd.read_csv('censored_start_stop.txt', sep='\t', 
                 parse_dates=['start_date', 'stop_date'])
kms = KMSurvival(col_start_date='start_date',
                 col_stop_date='stop_date',
                 col_censored = 'censored')
cutoff = '2008-12-28'                 
group = ['market']
kms.fit(df, cutoff, group)
kms.plot(vlines=[365])                 
```

![alt](images/kms_market.png?raw=True)


kmsurvival includes an auxiliary function to plot right-censoring.

```python
snapshot_date = '2008-12-28'
cutoff_date = '2008-09-18'
n = 20
plot_right_censor(df[:n].copy(), snapshot_date, cutoff_date)
```

![alt](images/right_censoring.png?raw=True)

See [the post](https://geweiwang.github.io/2016/01/temporal-dynamics-of-data-within-a-time-frame.html) about dynamic right-censoring.

### Dependencies

kmsurvival has been tested under Python 3.5, Numpy 1.1, pandas 0.18, and matplotlib 1.5.
