"""This module includes an auxiliary fucntion for illustrating right-censoring.
Author email: gewei.wang.link@gmail.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def plot_right_censor(X, snapshot_date, cutoff_date=None,
                      censored='censored',
                      start='start_date',
                      stop='stop_date',
                      figsize=(10,6)):
    """Plot to illustrate the impact of cutoff on right-censoring.

    Parameters
    ----------
    snapshot_date : str, format: 'yyyy-mm-dd'
        The date when a snapshot of the data was taken.
    cutoff_date : str, default None, format: 'yyyy-mm-dd'
        Define the end of the time frame.
        If None, cutoff_date equals to snapshot_date.
        Cutoff_date should be earlier than snapshot_date.
    censored, start, stop : str
        The names of columns in DataFrame for censored, start_date,
        and stop_date.
    """
    snapshot = pd.to_datetime(snapshot_date)
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        assert cutoff < snapshot, "error: cutoff should < snapshot!"
    else:
        cutoff = snapshot
    X['cutoff'] = cutoff
    X.sort_values([censored, stop, start],
                  ascending=[False, False, True], inplace=True)
    X.index = range(X.shape[0])

    T, NEWC = 'tenure', censored + "_new"
    oneday = timedelta(days=1)
    first_day = X[start].min()
    last_day = X[stop].max()

    # Only keep rows whose start_dates <= cutoff date
    # May create holes in index
    X = X.ix[X[start] <= cutoff, [censored, start, stop]]
    # Add a new column for new censored status
    X[NEWC] = np.where(X[stop] > cutoff, 2, X[censored])
    # Calculate tenures
    X[T] = (np.where(X[NEWC] >= 1,
                    cutoff - X[start],
                    X[stop] - X[start])
            / np.timedelta64(1, 'D')).astype(int)

    # Begin plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    TXT_SIZE, X_MIN, Y_MIN = 13, -2, -2
    linestyles = ['-' if c else ':' for c in X[NEWC].tolist()]
    colors = ['gr'[c-1] if c else 'k' for c in X[NEWC].tolist()]
    color_text = {'k': ' ..........\n uncensored',
                  'g': ' _______\n censored',
                  'r': ' _______\n uncensored\n  --> censored'}
    text_y = {}  # y value of text
    for status, grp in X.groupby(NEWC):
        c_status = 'gr'[status-1] if status else 'k'
        text_y[c_status] = (grp.index.max() + grp.index.min()) // 2

    # subplot 1
    ax1.set_title("\nTimeline\n")
    ax1.set_xlim(first_day - oneday, last_day + oneday)
    ax1.set_ylim(Y_MIN, X.index.max()+1)
    ax1.hlines(X.index.values, X[start].values,
               X[stop].where(X[stop].notnull(), cutoff).values,
               colors=colors, linestyles=linestyles)
    ax1.axvline(snapshot, c='b', lw='0.7')
    ax1.text(snapshot, Y_MIN, " Snapshot\n {}".format(snapshot_date),
             size=TXT_SIZE)
    ax1.text(first_day, Y_MIN, "{}".format(first_day.strftime('%Y-%m-%d')),
             size=TXT_SIZE-2)
    for c in sorted(set(colors)):
        ax1.text(snapshot, text_y[c], color_text[c],
                 color=c, size=TXT_SIZE)
    if cutoff_date:
        ax1.axvline(cutoff, c='b', ls='--', lw='0.7')
        ax1.text(cutoff, Y_MIN, "Cutoff \n{} ".format(cutoff_date),
                 ha="right", size=TXT_SIZE-2)

    # subplot 2
    ax21 = ax2.twinx()
    ax21.set_title("\nTenure\n")
    ax21.set_xlim((last_day - first_day).days, X_MIN)
    ax21.set_ylim(Y_MIN, X.index.max()+1)
    ax21.axvline(0, c='k', lw=0.7)
    ax21.text(0, Y_MIN, " Tenure\n 0 (day)", size=TXT_SIZE)
    ax21.hlines(X.index.values, 0, X[T], colors=colors, linestyles=linestyles)
    # Write tenures above tenure lines
    for i in range(X.index.size):
        tval = X.ix[X.index[i], T]
        ax21.text(tval // 2, X.index[i], tval, va='bottom')

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax21.set_axis_off()
    plt.show()
    