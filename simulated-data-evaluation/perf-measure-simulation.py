# Code to simulate a small data set with two classes and find best decision
# boundaries when three different performance measures are used
#
# Original R code by David J Hand, converted from R to Python using:
# https://www.codeconvert.ai/r-to-python-converter
# followed by manual editing by Peter Christen, March 2024

import math

import numpy as np
import matplotlib.pyplot as plt

# Part 1 ----------------------------------------------------------------------

n = 100  # Number of data points

num_tries = 0

# Entire randomly generated data sets

curr_angle = 0

# Repeat until we have a random distribution that results in a large angle of
# close to 60 degrees between all three performance measures
#
while curr_angle < 55:
    num_tries += 1
    print('Try %d:' % num_tries)

    np.random.seed(374)  # For a successful run

    gmaxFM = 0  # Maximum F-measure (FM) obtained
    gmaxMCC = 0  # Maximum Matthew correlation coefficient (MCC) obtained
    gminER = 1  # Minimum error rate (ER) obtained

    # Generate initial data set, bivariate normal, mean 0 and identity CV mx
    # with n points in each of two classes, and adjust means to be 0
    #
    x1 = np.random.normal(size=n)
    y1 = np.random.normal(size=n)
    x1m = np.mean(x1)
    y1m = np.mean(y1)
    x1 = x1 - x1m  # Ensure mean of both dimensions is zero
    y1 = y1 - y1m

    x2 = np.random.normal(size=n)
    y2 = np.random.normal(size=n)
    x2m = np.mean(x2)
    y2m = np.mean(y2)
    x2 = x2 - x2m
    y2 = y2 - y2m

    # Arrays to be used for keeping prediction information
    #
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    AUC = np.zeros(360)
    FM = np.zeros(360)
    ER = np.zeros(360)
    MCC = np.zeros(360)
    theta = 2 * np.pi * np.arange(1, 361) / 360
    xd = np.cos(theta)
    yd = np.sin(theta)

    # Find direction d1 which maximises AUC, and find direction d2
    # which minimises ER
    #
    for i in range(360):  # For each direction find AUC and ER

        for j in range(n):  # Project points onto that direction
            d1[j] = x1[j] * xd[i] + y1[j] * yd[i]
            d2[j] = x2[j] * xd[i] + y2[j] * yd[i]

        # Calculate ER using 0 as threshold
        #
        ER[i] = (np.sum(d1 < 0) + np.sum(d2 >= 0)) / (2 * n)

        # Calculate F-measure using 0 as threshold
        #
        num_tp = np.sum(d1 >= 0)
        num_fn = np.sum(d1 < 0)
        num_fp = np.sum(d2 >= 0)
        num_tn = np.sum(d2 < 0)

        FM[i] = 2.0 * num_tp / (2.0 * num_tp + num_fp + num_fn)

        # Matthew correlation coefficient
        #
        MCC[i] = (num_tn * num_tp - num_fn * num_fp) / \
                 math.sqrt((num_tp + num_fp) * (num_tp + num_fn) * \
                           (num_tn + num_fp) * (num_tn + num_fn))
        assert (-1 <= MCC[i]) and (MCC[i] <= 1), MCC[i]

    maxFM = np.max(FM)
    maxMCC = np.max(MCC)
    minER = np.min(ER)

    if maxFM > gmaxFM:
        gmaxFM = maxFM
    if maxMCC > gmaxMCC:
        gmaxMCC = maxMCC
    if minER < gminER:
        gminER = minER

    locmaxFM = np.argmax(FM)
    locmaxMCC = np.argmax(MCC)
    locminER = np.argmin(ER)

    print('Angle of maximum F-measure:', locmaxFM)
    print('Angle of maximum MCC:', locmaxMCC)
    print('Angle of minimum ER: ', locminER)
    print('  Difference MCC and ER:', locmaxMCC - locminER)
    print('  Difference MCC and FM:', locmaxMCC - locmaxFM)
    print('  Difference ER and FM: ', locminER - locmaxFM)

    if locmaxMCC > 180:
        curr_ang_mcc = locmaxMCC - 180
    else:
        curr_ang_mcc = locmaxMCC

    if locmaxFM > 180:
        curr_ang_fm = locmaxFM - 180
    else:
        curr_ang_fm = locmaxFM

    if locminER > 180:
        curr_ang_er = locminER - 180
    else:
        curr_ang_er = locminER

    curr_angle_mcc_er = abs(curr_ang_mcc - curr_ang_er)
    if curr_angle_mcc_er > 90:
        curr_angle_mcc_er = 180 - curr_angle_mcc_er

    curr_angle_mcc_fm = abs(curr_ang_mcc - curr_ang_fm)
    if curr_angle_mcc_fm > 90:
        curr_angle_mcc_fm = 180 - curr_angle_mcc_fm

    curr_angle_er_fm = abs(curr_ang_er - curr_ang_fm)
    if curr_angle_er_fm > 90:
        curr_angle_er_fm = 180 - curr_angle_er_fm

    print('  Difference from nearest 90 degrees (MCC to ER):', curr_angle_mcc_er)
    print('  Difference from nearest 90 degrees (MCC to FM):', curr_angle_mcc_fm)
    print('  Difference from nearest 90 degrees (ER to FM): ', curr_angle_er_fm)
    print()

    # Keep the minimum of all three
    #
    curr_angle = min(curr_angle_mcc_er, curr_angle_mcc_fm, curr_angle_er_fm)

print('Found an example after %d tries' % num_tries)
print()
print('   *** set numpy random seed to: %d' % num_tries)
print()

# Angle between directions for maximum MCC, maximum FM, and minimum ER
#
score = np.abs(90 - np.abs(locmaxMCC - locminER))
print('Angle between maximum MCC and minimum ER:', score)
print()
score = np.abs(90 - np.abs(locmaxMCC - locmaxFM))
print('Angle between maximum MCC and maximum FM:', score)
print()
score = np.abs(90 - np.abs(locmaxFM - locminER))
print('Angle between maximum FM and minimum FM:', score)
print()

# Plot MCC, FM and ER against angle
#
print('all min:', min(MCC), min(FM), min(ER))
print('all max:', max(MCC), max(FM), max(ER))

ymax = max(max(MCC), max(FM), max(ER)) * 1.1  # Add some space at top and bottom
if min(MCC) >= 0:
    ymin = min(min(MCC), min(FM), min(ER)) * 0.9
else:
    ymin = min(min(MCC), min(FM), min(ER)) * 1.2
print('y min and max:', ymin, ymax)
print()

plt.plot(np.arange(1, 361), MCC, 'b-', linewidth=1, label='MCC')
plt.plot(np.arange(1, 361), ER, 'r-', linewidth=1, label='Error rate')
plt.plot(np.arange(1, 361), FM, 'g-', linewidth=1, label='F-measure')
plt.ylim(ymin, ymax)
plt.xlabel('Angle')
plt.ylabel('MCC / F-measure / Error rate')
plt.title('Differences between MCC, error rate, and F-measure')
plt.plot(locmaxMCC + 1, gmaxMCC, 'bo', markersize=6)
plt.plot(locmaxFM + 1, gmaxFM, 'go', markersize=6)
plt.plot(locminER + 1, gminER, 'ro', markersize=6)
plt.legend(loc="best")

# plt.show()
#
# input("Press Enter to continue...")

# plt.savefig('angle-between-directions-mcc-fm-er.pdf')
# plt.savefig('angle-between-directions-mcc-fm-er.eps')
plt.savefig('angle-between-directions-mcc-fm-er.svg')

# Generate a black/white version of the plot
#
plt.clf()

plt.plot(np.arange(1, 361), MCC, 'k-', linewidth=1, label='MCC')
plt.plot(np.arange(1, 361), ER, 'k--', linewidth=1, label='Error rate')
plt.plot(np.arange(1, 361), FM, 'k-.', linewidth=1, label='F-measure')
plt.ylim(ymin, ymax)
plt.xlabel('Angle')
plt.ylabel('MCC / F-measure / Error rate')
plt.title('Differences between MCC, error rate, and F-measure')
plt.plot(locmaxMCC + 1, gmaxMCC, 'ko', markersize=6)
plt.plot(locmaxFM + 1, gmaxFM, 'ko', markersize=6)
plt.plot(locminER + 1, gminER, 'ko', markersize=6)
plt.legend(loc="best")

# plt.show()
#
# input("Press Enter to continue...")

# plt.savefig('angle-between-directions-mcc-fm-er-bw.pdf')
# plt.savefig('angle-between-directions-mcc-fm-er-bw.eps')
plt.savefig('angle-between-directions-mcc-fm-er-bw.svg')

# Part 2 ----------------------------------------------------------------------

plt.clf()

# Plot data and directions
#
plt.plot(x1, y1, marker='o', linestyle='None', markersize=3)
plt.plot(x2, y2, marker='>', linestyle='None', markersize=3)

ang_mcc = locmaxMCC * 2 * np.pi / 360
xv = 3 * np.cos(ang_mcc)
yv = 3 * np.sin(ang_mcc)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color="blue", label='MCC')

ang_er = locminER * 2 * np.pi / 360
xv = 3 * np.cos(ang_er)
yv = 3 * np.sin(ang_er)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color='red', label='Error rate')

ang_fm = locmaxFM * 2 * np.pi / 360
xv = 3 * np.cos(ang_fm)
yv = 3 * np.sin(ang_fm)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color='green', label='F-measure')

plt.title('Directions of MCC, error rate, and F-measure')

print('Angle MCC:       ', ang_mcc)
print('Angle error rate:', ang_er)
print('Angle F-measure: ', ang_fm)
print('  Difference MCC to ER:', abs(ang_mcc - ang_er))
print('  Difference MCC to FM:', abs(ang_mcc - ang_fm))
print('  Difference ER to FM: ', abs(ang_er - ang_fm))
print()

plt.legend(loc="best")

# plt.show()
# input("Press Enter to continue...")

# plt.savefig('directions-mcc-fm-er.pdf')
# plt.savefig('directions-mcc-fm-er.eps')
plt.savefig('directions-mcc-fm-er.svg')

# Generate a black/white version of the plot
#
plt.clf()

# Plot data and directions
#
plt.plot(x1, y1, marker='o', color='k', linestyle='None', markersize=3)
plt.plot(x2, y2, marker='>', color='k', linestyle='None', markersize=3)

ang_mcc = locmaxMCC * 2 * np.pi / 360
xv = 3 * np.cos(ang_mcc)
yv = 3 * np.sin(ang_mcc)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color="k", linestyle='-',
         label='MCC')

ang_er = locminER * 2 * np.pi / 360
xv = 3 * np.cos(ang_er)
yv = 3 * np.sin(ang_er)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color='k', linestyle='--',
         label='Error rate')

ang_fm = locmaxFM * 2 * np.pi / 360
xv = 3 * np.cos(ang_fm)
yv = 3 * np.sin(ang_fm)
plt.plot([-xv, xv], [-yv, yv], linewidth=2, color='k', linestyle='-.',
         label='F-measure')

plt.title('Directions of MCC, error rate, and F-measure')

print('Angle MCC:       ', ang_mcc)
print('Angle error rate:', ang_er)
print('Angle F-measure: ', ang_fm)
print('  Difference MCC to ER:', abs(ang_mcc - ang_er))
print('  Difference MCC to FM:', abs(ang_mcc - ang_fm))
print('  Difference ER to FM: ', abs(ang_er - ang_fm))
print()

plt.legend(loc="best")

plt.savefig('directions-mcc-fm-er-bw.svg')
