import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import powerlaw
import random

np.random.seed(0)
random.seed(0)

FONTSIZE = 18

df = pd.read_csv('germany-consumption/all-users-daily-data.csv')

energy = 1 + df['energy'].values.astype(np.float64)


df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

dates  = list(sorted(df['date']))
date2idx = {}

counter = 0
for d in dates:
    if d not in date2idx:
        date2idx[d] = counter
        counter += 1

household2idx = dict([(x, i) for i, x in enumerate(set(df['userId']))])

cummulative_signals = np.zeros((len(household2idx), len(date2idx)))
signals = np.zeros((len(household2idx), len(date2idx)))

for _, x in df.iterrows():
    h = household2idx[x.userId]
    t = date2idx[x.date]
    cummulative_signals[h, t] = float(x.energy)

cummulative_signals /= 10**10

signals = np.gradient(cummulative_signals, axis=1)

log_signals = np.log(1 + signals)
nan_idx = np.isnan(log_signals)
log_signals_mean = log_signals[~nan_idx].mean()
log_signals_std = log_signals[~nan_idx].std()


for i in range(signals.shape[0]):
    for t in range(signals.shape[1]):
        log_signals[i, t] = np.random.normal(loc=log_signals_mean, scale=log_signals_std)

signals = np.exp(log_signals)



fig, ax = plt.subplots(figsize=(6, 6))

plt.title('Household Energy Consumption Distribution', fontsize=FONTSIZE)

fit = powerlaw.Fit(signals.flatten(), xmin=1)
fit_initial = powerlaw.Fit(signals[:, 0], xmin=1)

powerlaw.plot_pdf(signals.flatten(), color='b', ax=ax, label='Consumption Data (all measurements)')
powerlaw.plot_pdf(signals[:, 0], color='r', ax=ax, label='Consumption Data (Day 0 measurements)')
fit.lognormal.plot_pdf(color='b', linestyle='--', ax=ax, label=f'Log-Normal Fit ($\\mu = {fit.lognormal.mu:.2f}, \\sigma = {fit.lognormal.sigma:.2f}$)')
fit_initial.lognormal.plot_pdf(color='r', linestyle='--', ax=ax, label=f'Log-Normal Fit ($\\mu = {fit_initial.lognormal.mu:.2f}, \\sigma = {fit_initial.lognormal.sigma:.2f}$)')
plt.xlabel('Daily Consumption (kWh)', fontsize=FONTSIZE)
plt.ylabel('Frequency', fontsize=FONTSIZE)
plt.legend(fontsize=0.5*FONTSIZE)

plt.savefig('distribution_consumption.pdf')

np.savetxt('germany-consumption/signals.txt.gz', signals)
np.savetxt('germany-consumption/log_signals.txt.gz', log_signals)
