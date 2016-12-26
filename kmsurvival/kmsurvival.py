"""This module includes a Kaplan-Meier survival estimation class.
Author email: gewei.wang.link@gmail.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import OrderedDict
from datetime import timedelta

class KMSurvival():
    """Kaplan-Meier estimation for right-censored data.

    Parameters
    ----------
    col_censored : str, default None
        column name for a customer's status (censored or uncensored)
    col_start_date : str, default None
        column name for date when a customer starts a service
    col_stop_date : str, default None
        column name for date when a customer stops a service
    col_tenure : str, default None
        column name for tenure, duration, frequency, or life span.
        If col_tenure is None, tenures are calculated using
        cutoff, col_start_date, col_stop_date, and col_censored.
    CENSORED : int, default 1
        an integer in 'col_censored' that means data is right-censored
    UNCENSORED : int, default 0
        an integer in 'col_censored' that means uncensored

    Attributes
    ----------
    estimates_ : pandas DataFrame
        Survival and hazard probabilities functions
    subgrps_ : OrderedDict
        Sub-groups in each group

    Input formats
    -------------
    Data are expected to be one of these formats:
    Format #1:
        censored | start_date | stop_date  | [other columns]
        ---------+------------+------------+----------------
          0      | 2001-01-01 | 2002-05-01 |
    Format #2:
        censored | tenure | [other columns]
        ---------+--------+----------------
          1      |  150   |
    Format #3:
        censored | start_date | stop_date  | tenure | [other columns]
        ---------+------------+------------+--------+----------------
            1    | 2001-01-01 | 2002-05-01 |  150   |
    For #3, the column 'tenure' will be ignored; tenures are calculated
    by 'censored', 'start_date', 'stop_date' and cutoff date.

    Examples
    --------
    >>> # Example 1
    >>> doctest_data =  '''
    ...  {"columns": ["start_date","stop_date", "censored"],
    ...   "data": [["2008-12-08", "", 1],
    ...            ["2008-12-08", "", 1],
    ...            ["2008-12-08", "2008-12-28", 0]]}
    ...  '''
    >>> df = pd.read_json(doctest_data, orient='split',
    ...                   convert_dates=['start_date', 'stop_date'])
    >>> kms = KMSurvival(col_censored='censored',
    ...                  col_start_date='start_date',
    ...                  col_stop_date='stop_date')
    >>> cutoff = '2008-12-28'
    >>> estimates = kms.fit(df, cutoff)
    >>> estimates['survival'].values
    array([ 0.66666667])
    >>> estimates['hazard'].values
    array([ 0.33333333])
    >>>
    >>> # Example 2
    >>> doctest_data = {"tenure": [20, 20, 20],
    ...                 "censored": [1, 1, 0]}
    >>> df = pd.DataFrame(doctest_data)
    >>> kms = KMSurvival(col_censored='censored',
    ...                  col_tenure='tenure')
    >>> estimates = kms.fit(df)
    >>> estimates['survival'].values
    array([ 0.66666667])
    >>> estimates['hazard'].values
    array([ 0.33333333])
    """

    H, S, T = 'hazard', 'survival', '__tenure'

    def __init__(self,
                 col_censored=None,
                 col_start_date=None,
                 col_stop_date=None,
                 col_tenure=None,
                 CENSORED=1, UNCENSORED=0):
        self.col_censored = col_censored
        self.col_start_date = col_start_date
        self.col_stop_date = col_stop_date
        self.col_tenure = col_tenure
        self.CENSORED = CENSORED
        self.UNCENSORED = UNCENSORED

    @property
    def cutoff(self):
        return self._cutoff.strftime("%Y-%m-%d")

    @staticmethod
    def _check_data(X, col_censored, group):
        """Verify the validity of data given"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("kmsurvival requires data to be a panads DataFrame")
        if not isinstance(group, list):
            raise TypeError("group is required to be a list")
        if X[col_censored].value_counts().size != 2:
            raise ValueError("Column '{}' should have two unique values"
                             .format(col_censored))
        return X

    def fit(self, X, cutoff=None, group=None):
        """Calculate hazard and survival probabilities

        Parameters
        ----------
        X : input, pandas DataFrame
        cutoff : str, default None
            cutoff date, format 'yyyy-mm-dd'
        group : list of strings, default None
            Column names in X for cross comparisons
        """
        if group is None:
            group = []
        X = self._check_data(X, self.col_censored, group)
        if cutoff:
            self._cutoff = pd.to_datetime(cutoff)
        else:
            self._cutoff = pd.to_datetime('today')

        if cutoff and self.col_start_date and self.col_stop_date:
            X = X.ix[X[self.col_start_date] <= self._cutoff,
                      [self.col_start_date, self.col_stop_date, \
                       self.col_censored] + group]

            # if stop_date > cutoff date, change censored status to CENSORED
            X.ix[X[self.col_stop_date] > self._cutoff, self.col_censored] \
                = self.CENSORED

            X[self.T] = (np.where(
                            X[self.col_censored] == self.CENSORED,
                            self._cutoff - X[self.col_start_date],
                            X[self.col_stop_date] - X[self.col_start_date])
                        / np.timedelta64(1, 'D')).astype(int)
        elif self.col_tenure:
            X[self.T] = X[self.col_tenure]

        RCC, RUCC  = 'reverse_censored_cumsum', 'reverse_uncensored_cumsum'
        CNT_C, CNT_UC = 'cnt_censored', 'cnt_uncensored'

        if group:
            # s_cnt is Series with a MultiIndex which is group + ['tenure']
            s_cnt = (X.groupby([self.col_censored] + group)[self.T] \
                    .value_counts()).sort_index()
            s_cnt.rename('cnt_tenure', inplace=True)

            df = s_cnt.unstack(level=0, fill_value=0)
            df.rename(columns={self.CENSORED: CNT_C, self.UNCENSORED: CNT_UC},
                      inplace=True)

            df.sortlevel(level=0, ascending=False, inplace=True)
            df[RCC] = df.groupby(level=group)[CNT_C].cumsum()
            df[RUCC] = df.groupby(level=group)[CNT_UC].cumsum()
            df[self.H] = df[CNT_UC] / (df[RCC] + df[RUCC])

            # The survival at tenure t is the product of one minus the hazards
            # for all tenures less than t.
            df.sortlevel(level=0, inplace=True)
            df[self.S] = (1 - df[self.H]).groupby(level=group).cumprod()

            # change index 'tenure' to column
            df.reset_index(level=[self.T], inplace=True)

            self.subgrps_ = OrderedDict()
            for g in group:
                self.subgrps_.update({g: tuple(X[g].unique().tolist())})
        else:
            s_cnt = (X.groupby([self.col_censored])[self.T] \
                    .value_counts()).sort_index()
            s_cnt.rename('cnt_tenure', inplace=True)

            df = s_cnt.unstack(level=0, fill_value=0)
            df.rename(columns={self.CENSORED: CNT_C, self.UNCENSORED: CNT_UC},
                      inplace=True)

            # descending tenure
            df.sort_index(ascending=False, inplace=True)
            df[RCC] = df[CNT_C].cumsum()
            df[RUCC] = df[CNT_UC].cumsum()
            df[self.H] = df[CNT_UC] / (df[RCC] + df[RUCC])

            # ascending tenure
            df.sort_index(inplace=True)
            df[self.S] = (1 - df[self.H]).cumprod()

        df.drop([RCC, RUCC], axis=1, inplace=True)
        df.columns.rename('estimates_', inplace=True)
        df.sort_index(axis=1, ascending=False, inplace=True)
        if cutoff:
            df['cutoff'] = cutoff
        self.estimates_ = df
        return self.estimates_

    def plot(self, curve='survival', strata=None, vlines=None,
             figsize=(9,6), style=None, xlabel='Tenure', title=None,
             **kwargs):
        """Plot survival or hazard curves of the whole, or all combinations
        of self.subgrps_ or any valid combinations of strata.

        Parameters
        ----------
        curve : str, default 'survival'
            'survival' -- plot survival function. Shortcut: 's'
            'hazard'   -- plot hazard function. Shortcut: 'h'
        strata : a list of lists of strings, default None
            Customized combinations of strata across groups to be compared
            instead of all in self.subgrps_
            Examples:
            strata=[['market-A], ['channel-1', 'channel-2']]
            strata=[['market-A', 'market-D']]
        style : str, default seaborn
            any style in plt.style.available
        vlines : list of int, default None
            A list of days at which vertical lines are drawn.
            Example:
            [180, 365] will draw vertical lines at tenure 180 and 365.
        kwargs : dict
            Options to pass to matplotlib plotting method
            Examples: line_width=1.5, lw=2, ls='--'
        """
        try:
            # prepare data for plotting
            val = self.estimates_
            if curve[0] == 'h':
                col_name = self.H   # hazard curve
            else:
                col_name = self.S   # survival curve
            max_prob = val[col_name].max() * 102.0

            if hasattr(self, 'subgrps_'):
                grp_size = len(self.subgrps_)
                if grp_size > 1:
                    # Cartesian product of self.subgrps_
                    valid_pairs = \
                        list(itertools.product(*list(self.subgrps_.values())))
                else:
                    valid_pairs = list(self.subgrps_.values())[0]
                max_tenure = val[self.T].max()
            else:
                valid_pairs = []
                grp_size = 0
                max_tenure = val.index.values.max()

            if strata:
                # use customized combinations to be compared
                grp_size = len(strata)
                if grp_size == 1:
                    pairs = strata[0]
                elif grp_size > 1:
                    pairs = list(itertools.product(*strata))
                for p in pairs:
                    if p not in valid_pairs:
                        raise KeyError(p)
            elif hasattr(self, 'subgrps_'):
                # use all combinations of self.subgroups_
                pairs = valid_pairs

            # begin plotting
            if style:
                plt.style.use(style)
            else:
                sns.set()  # use seaborn style

            fig = plt.figure(figsize=figsize)
            ax = plt.axes()

            if grp_size >= 1:
                if grp_size == 1:  label_link = ''
                elif grp_size > 1: label_link = ' - '
                for pair in pairs:
                    ax.plot(val.xs(pair)[self.T],
                            val.xs(pair)[col_name]*100.0,
                            label=label_link.join(pair), **kwargs)
            else:
                ax.plot(val[col_name]*100.0, **kwargs)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(col_name.capitalize())
            ax.set_xlim([0, max_tenure + 4])
            ax.set_ylim([0, max_prob + 0.2])
            if title:
                ax.set_title(title, size=14)
            else:
                ax.set_title(col_name.capitalize() + ' Curve\ncutoff date: {}'
                            .format(self.cutoff), size=14)
            ax.set_yticklabels(['{:.1f}%'.format(y) for y in ax.get_yticks()])
            if vlines:
                ax.vlines(vlines, 0, 100, colors='g', linestyles='dotted',
                          label='{}={}'.format(xlabel, vlines))
            ax.legend(loc='best')
            plt.show()
        except KeyError as e:
            print(e, 'Not in valid pairs:\n\n', valid_pairs)

if __name__ == "__main__":
    import doctest
    doctest.testmod()