""" Unstructrured Mesh Implementation VarDA Cost Function """ 

import numpy as np
import torch

def cost_fn_J(w, data, settings):
    """Computes VarDA cost function.
    """
    d = data.get("d")
    G_V = data.get("G_V")

    #Calculate J_o = 1/2 * [(HVw - d).T * R-1 * (HVw -d)]
    #Define Q = HVw - d
    Q = (G_V @ w - d)

    if settings.OBS_VARIANCE:
        # R is proportional to identity
        # as reasonable to assume no observation error correlations between independent sensors.
        J_o = (0.5 / settings.OBS_VARIANCE) * np.dot(Q.T, Q)

    wT = np.transpose(w)
    wTw = np.dot(wT, w)
    J_b = 0.5 * settings.ALPHA * wTw
    J = J_b + J_o

    if settings.DEBUG:
        print("J_b = {}, J_o = {}".format(J_b, J_o))
    return J


def grad_J(w, data, settings):
    d = data.get("d")
    G_V = data.get("G_V")

    Q = (G_V @ w - d)
    P = G_V.T
    if settings.OBS_VARIANCE:
        grad_o = (1.0 / settings.OBS_VARIANCE ) * np.dot(P, Q)

    grad_J = settings.ALPHA * w + grad_o

    return grad_J