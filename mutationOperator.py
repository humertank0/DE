import numpy as np
import random
# from DE import calFitness


# DE/rand/1
def mutation_rr1(XTemp, F, NP, size):  # r/r/1
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):

        r1, r2, r3 = np.random.choice(NP, 3, replace=False)

        # for j in range(size):
        # XMutationTmp[i, j] = XTemp[r1, j] + F * (XTemp[r2, j] - XTemp[r3, j])
        XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[r3])
        for j in range(size):
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0

    return XMutationTmp


# DE/rand/2
def mutation_rr2(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):

        r1, r2, r3, r4, r5 = np.random.choice(NP, 5, replace=False)

        # for j in range(size):
        # XMutationTmp[i, j] = XTemp[r1, j] + F * (XTemp[r2, j] - XTemp[r3, j])
        XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[r3]) + F * (XTemp[r4] - XTemp[r5])
        for j in range(size):
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0

    return XMutationTmp


# temp
# def mutation_temp(XTemp, F, NP, size):  # current 3 of best
#     XMutationTmp = np.zeros((NP, size))
#     for i in range(NP):
#
#         r1, r2, r3 = np.random.choice(NP, 3, replace=False)
#
#         # for j in range(size):
#         # XMutationTmp[i, j] = XTemp[r1, j] + F * (XTemp[r2, j] - XTemp[r3, j])
#         r1_calc = calFitness(XTemp, XTemp[r1])
#         r2_calc = calFitness(XTemp, XTemp[r2])
#         r3_calc = calFitness(XTemp, XTemp[r3])
#         if (r1_calc > r2_calc and r1_calc > r3_calc):
#             XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[r3])
#         elif (r2_calc > r1_calc and r2_calc > r3_calc):
#             XMutationTmp[i] = XTemp[r2] + F * (XTemp[r1] - XTemp[r3])
#         else:
#             XMutationTmp[i] = XTemp[r3] + F * (XTemp[r1] - XTemp[r2])
#         for j in range(size):
#             if (XMutationTmp[i, j] > 1):
#                 XMutationTmp[i, j] = 1
#             if (XMutationTmp[i, j] < 0):
#                 XMutationTmp[i, j] = 0
#
#     return XMutationTmp


# DE/CURRENT-TO-RAND/1
def mutation_ctr1(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):

        r1, r2, r3 = np.random.choice(NP, 3, replace=False)

        XMutationTmp[i] = XTemp[i] + F * (XTemp[r1] - XTemp[i]) + F * (XTemp[r2] - XTemp[r3])
        for j in range(size):
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0

    return XMutationTmp


def mutation_NSDE(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):
        r1, r2, r3 = np.random.choice(NP, 3, replace=False)
        difference = XTemp[r2] - XTemp[r3]
        if (random.random() < 0.5):
            XMutationTmp[i] = XTemp[r1] + difference * np.random.normal(loc=0.5, scale=0.5)
        else:
            XMutationTmp[i] = XTemp[r1] + difference * np.random.standard_cauchy()
        for j in range(size):
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0
    return XMutationTmp


# DE/rand/3
# 使用了自动生成的F
def mutation_rr3(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))

    for i in range(NP):

        r1, r2, r3, r4, r5, r6, r7 = np.random.choice(NP, 7, replace=False)

        F_adaptive = np.random.normal(loc=0.5, scale=0.15, size=7)

        F = F_adaptive[0] + np.random.normal(loc=0, scale=0.5) * (F_adaptive[1] - F_adaptive[2]) + \
            np.random.normal(loc=0, scale=0.5) * (F_adaptive[3] - F_adaptive[4]) + \
            np.random.normal(loc=0, scale=0.5) * (F_adaptive[5] - F_adaptive[6])

        XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[r3]) + F * (XTemp[r4] - XTemp[r5]) + \
                          F * (XTemp[r6] - XTemp[r7])

        for j in range(size):

            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0

    return XMutationTmp


# DE/RAND-TO-CURRENT/2
def mutation_rtc2(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):

        r1, r2, r3, r4 = np.random.choice(NP, 4, replace=False)
        F_adaptive = np.random.normal(loc=0.5, scale=0.15, size=5)

        F = F_adaptive[0] + np.random.normal(loc=0, scale=0.5) * (F_adaptive[1] - F_adaptive[2]) + \
            np.random.normal(loc=0, scale=0.5) * (F_adaptive[3] - F_adaptive[4])
        # for j in range(size):
        # XMutationTmp[i, j] = XTemp[r1, j] + F * (XTemp[r2, j] - XTemp[r3, j])
        XMutationTmp[i] = XTemp[r1] + F * (XTemp[r2] - XTemp[i] + XTemp[r3] - XTemp[r4])
        for j in range(size):
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0

    return XMutationTmp


# IMMSADE rr1的改进
def mutation_immsade(XTemp, F, NP, size):
    XMutationTmp = np.zeros((NP, size))
    for i in range(NP):

        r1, r2, r3 = np.random.choice(NP, 3, replace=False)  # False表示不能被多次选择

        XMutationTmp[i] = random.uniform(0.7, 1.0) * XTemp[r1] + random.uniform(0.1, 0.8) * (XTemp[r2] - XTemp[r3])

        for j in range(size):
            countZero = 0  # 处理选择特征数为0的特殊情况
            if (XMutationTmp[i, j] > 1):
                XMutationTmp[i, j] = 1
            if (XMutationTmp[i, j] < 0):
                XMutationTmp[i, j] = 0
                countZero = countZero + 1
        if countZero == size: # 若选择的特征数为0，则使用原先版本
            XMutationTmp[i] = XTemp[i]
    return XMutationTmp


# DE/CURRENT-TO-BEST/1
# def mutation_ctb1(XTemp, F, NP, size):
#     XMutationTmp = np.zeros((NP, size))
#     bestIndex = 0
#     bestFitness = 0
#     for t in range(NP):
#         if (bestFitness > calFitness(XTemp, XTemp[t])):
#             bestFitness = calFitness(XTemp, XTemp[t])
#             bestIndex = t
#     for i in range(NP):
#
#         r1, r2 = np.random.choice(NP, 2, replace=False)
#
#         XMutationTmp[i] = XTemp[i] + F * (XTemp[bestIndex] - XTemp[i] + XTemp[r1] - XTemp[r2])
#         for j in range(size):
#             if (XMutationTmp[i, j] > 1):
#                 XMutationTmp[i, j] = 1
#             if (XMutationTmp[i, j] < 0):
#                 XMutationTmp[i, j] = 0
#
#     return XMutationTmp
