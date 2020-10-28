#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:15:04 2020

@author: runzhitian
"""

import numpy as np
import time
import csv
import math

np.random.seed(1)

######################################################################
# loading data
def read_file(filename, column_id, window_size, Add_EN=True):

    csvFile = open(filename, "r")
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        # omit the first line
        if reader.line_num == 1:
            continue

        data.append(item)

    csvFile.close()

    dataset = []

    for line in data:

        dataset.append(int(line[column_id]))

    dataset = np.array(dataset)
    num_data = len(dataset) - len(dataset) % window_size

    if Add_EN:
        dataset = dataset[:num_data].reshape(-1, window_size).sum(1).squeeze()

    else:
        dataset = (
            dataset[:num_data].reshape(-1, window_size)[:, window_size - 1].squeeze()
        )

    return dataset


# loading data y, d, r, I, C
# y, d ,r: infectious, death, recovered cases
# I: the cumulative number of active infectious
# C: the cumulative confirmed cases
# N: the total population

window_size = 1  # The memory length of the model.

filename_y = "../data/timeseries_canada/cases_timeseries_canada.csv"
column_id_y = 2

filename_d = "../data/timeseries_canada/mortality_timeseries_canada.csv"
column_id_d = 2

filename_r = "../data/timeseries_canada/recovered_timeseries_canada.csv"
column_id_r = 2

filename_I_C = "../data/timeseries_canada/active_timeseries_canada.csv"
column_id_I = 5
column_id_C = 2


y = read_file(filename_y, column_id_y, window_size, True)

d = read_file(filename_d, column_id_d, window_size, True)
d = np.pad(d, ((len(y) - len(d), 0)), "constant", constant_values=0)

r = read_file(filename_r, column_id_r, window_size, True)
r = np.pad(r, ((len(y) - len(r), 0)), "constant", constant_values=0)

r_d = (
    r + d
)  # considering a simpler setting: the recovered and the death cases are treated as one compartment

I = read_file(filename_I_C, column_id_I, window_size, False)
C = read_file(filename_I_C, column_id_C, window_size, False)


# ------------------------------------------------#
# construct the training dataset
num_data = len(y)
num_train = 50
# num_predict = 30
start = 10
end = start + num_train

y_train = y[start:end]
rd_train = r_d[start:end]

# y_train = y[30:170]
# d_train = d[30:170]
# r_train = r[30:170]
# rd_train = r_d[30:170]

N = 37786763 * np.ones(len(C))
Z = np.log((N - C) / N)

I_train = I[start - 1 : end - 1]
Z_train = Z[start - 1 : end - 1]
# I_train = I[29:169]
# Z_train = Z[29:169]


data_train = np.concatenate(
    (np.log(np.expand_dims(I_train, axis=1)), np.expand_dims(Z_train, axis=1)), axis=1
)
# data_train = np.concatenate((np.expand_dims(I_train, axis=1), np.expand_dims(Z_train, axis=1)), axis=1)


######################################################################
# fit GAM model using pygam

from pygam import PoissonGAM, l, s, f, intercept

model_y = PoissonGAM(s(0) + s(1)).fit(data_train, y_train)

model_rd = PoissonGAM(s(0)).fit(np.log(I_train), rd_train)


######################################################################
# prediction


def Predict(model_y, model_rd, I, Z, sample_EN=False):

    I = np.array(I).reshape(1, 1)
    Z = np.array(Z).reshape(1, 1)

    log_I = np.log(I)

    if sample_EN:

        mu_y = model_y.predict_mu(np.concatenate((log_I, Z), axis=1))
        mu_rd = model_rd.predict_mu(log_I)

        y = np.random.poisson(mu_y, 10).mean()
        rd = np.random.poisson(mu_rd, 10).mean()

    else:

        y = model_y.predict(np.concatenate((log_I, Z), axis=1))
        rd = model_rd.predict(log_I)

    y_interval = model_y.confidence_intervals(
        np.concatenate((log_I, Z), axis=1), width=0.95
    )

    return y, rd, y_interval


# Long term prediction: given current data, predict multidays data.


def Predict_multiDay(model_y, model_rd, I, C, length, sample_EN=False):

    y = 0
    rd = 0
    N = 37786763

    collect_I = []
    collect_y = []
    collect_y_interval = []

    for j in range(length):

        I = np.max((y + I - rd, 1))
        C = C + y
        Z = np.log((N - C) / N)

        y, rd, y_interval = Predict(model_y, model_rd, I, Z, sample_EN)

        collect_I.append(I)
        collect_y.append(y)
        collect_y_interval.append(y_interval)

    return collect_I, collect_y, collect_y_interval


def average(I, Z, rd, C, y, num_sample):

    y = y.reshape(-1, 1)
    I = I * np.ones((num_sample, 1))
    Z = Z * np.ones((num_sample, 1))
    rd = rd * np.ones((num_sample, 1))
    C = C * np.ones((num_sample, 1))
    N = 37786763 * np.ones((num_sample, 1))

    I = y + I - rd
    C = C + y
    Z = np.log((N - C) / N)
    log_I = np.log(I)

    mu_y = model_y.predict_mu(np.concatenate((log_I, Z), axis=1)).mean()
    rd = model_rd.predict(log_I).mean()

    return mu_y, rd


def Predict_sampling(model_y, model_rd, I, C, length, num_sample=1000):

    N = 37786763
    Z = np.log((N - C) / N)
    I_ = np.array(I).reshape(1, 1)
    Z_ = np.array(Z).reshape(1, 1)
    log_I = np.log(I_)

    mu_y = model_y.predict_mu(np.concatenate((log_I, Z_), axis=1))
    rd = model_rd.predict(log_I)

    collect_I = []
    collect_y = []

    for j in range(length):

        y = np.random.poisson(mu_y, num_sample)

        y_mean = y.mean()
        I = y_mean + I - rd
        C = C + y_mean
        Z = np.log((N - C) / N)

        mu_y, rd = average(I, Z, rd, C, y, num_sample)

        collect_I.append(I)
        collect_y.append(y_mean)

    return collect_I, collect_y


# Given today's data, predict the next day's data.


def Predict_Nextday(model_y, model_rd, I, Z):

    XX = np.concatenate(
        (np.log(np.expand_dims(I, axis=1)), np.expand_dims(Z, axis=1)), axis=1
    )
    y = model_y.predict(XX)
    y_interval = model_y.confidence_intervals(XX, width=0.95)

    return y, y_interval


predict_I, predict_y, interval_y = Predict_multiDay(
    model_y, model_rd, I[start], C[start], num_data - start, False
)
#
predict_I = np.array(predict_I).squeeze()
predict_y = np.array(predict_y).squeeze()
interval_y = np.array(interval_y).squeeze()


Nextday_y, Nextday_interval = Predict_Nextday(
    model_y, model_rd, I[start - 1 : num_data - 1], Z[start - 1 : num_data - 1]
)


######################################################################
# predict using multiple samples
pred_I, pred_y = Predict_sampling(
    model_y, model_rd, I[start - 1], C[start - 1], num_data - start
)


######################################################################
# plotting

import matplotlib.pyplot as plt

x = np.linspace(start, num_data, num_data - start)

plt.figure()
plt.plot(x, predict_I, color="red", linewidth=2.0, linestyle="-.", label="predict_I")
plt.plot(x, I[start:], color="blue", linewidth=2.0, linestyle=":", label="I")


plt.xlabel("Day", fontsize=15)
plt.ylabel("Active infectious", fontsize=15)
plt.legend(loc="upper left", fontsize=10)
plt.show()


plt.figure()
plt.plot(x, predict_y, color="red", linewidth=2.0, linestyle="-.", label="predict_y")
plt.plot(x, interval_y, color="grey", linewidth=2.0, linestyle="-.")
plt.plot(x, y[start:], color="blue", linewidth=2.0, linestyle=":", label="y")
plt.xlabel("Day", fontsize=15)
plt.ylabel("increased cases per day", fontsize=15)
plt.legend(loc="upper left", fontsize=10)
plt.show()


plt.figure()
plt.plot(
    x, Nextday_y, color="red", linewidth=2.0, linestyle="-.", label="predict_y_oneday"
)
# plt.plot(x,  Nextday_interval, color='grey', linewidth=2.0, linestyle='-.')
plt.plot(x, y[start:], color="blue", linewidth=2.0, linestyle=":", label="y")
plt.xlabel("Day", fontsize=15)
plt.ylabel("increased cases per day", fontsize=15)
plt.legend(loc="upper left", fontsize=10)
plt.show()


plt.figure()
plt.plot(x, pred_I, color="red", linewidth=2.0, linestyle="-.", label="predict_I")
plt.plot(x, I[start:], color="blue", linewidth=2.0, linestyle=":", label="I")


plt.xlabel("Day", fontsize=15)
plt.ylabel("Active infectious", fontsize=15)
plt.legend(loc="upper left", fontsize=10)
plt.show()


plt.figure()
plt.plot(x, pred_y, color="red", linewidth=2.0, linestyle="-.", label="predict_y")
# plt.plot(x, interval_y, color='grey', linewidth=2.0, linestyle='-.')
plt.plot(x, y[start:], color="blue", linewidth=2.0, linestyle=":", label="y")
plt.xlabel("Day", fontsize=15)
plt.ylabel("increased cases per day", fontsize=15)
plt.legend(loc="upper left", fontsize=10)
plt.show()
