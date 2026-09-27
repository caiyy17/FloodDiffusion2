"""Generic SMPL-22 skeleton constants and neutral-body FK.

Shared by every rotation-decoding representation (MEI-138, SOMA_relative-271,
MotionStreamer-272) and the mesh renderer. Nothing here is representation-
specific. Neutral rest joints / parents baked below (extracted from the SMPL-H
neutral body).
"""
import numpy as np

SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

SMPL_NEUTRAL_REST_JOINTS = np.array([[-1.79505953e-03, -2.23333446e-01,  2.82191255e-02],
 [ 6.77246757e-02, -3.14739671e-01,  2.14037877e-02],
 [-6.94655406e-02, -3.13855126e-01,  2.38993038e-02],
 [-4.32792313e-03, -1.14370215e-01,  1.52281192e-03],
 [ 1.02001221e-01, -6.89938274e-01,  1.69079858e-02],
 [-1.07755594e-01, -6.96424140e-01,  1.50492738e-02],
 [ 1.15910534e-03,  2.08102144e-02,  2.61528404e-03],
 [ 8.84055199e-02, -1.08789863e+00, -2.67853442e-02],
 [-9.19818258e-02, -1.09483879e+00, -2.72625243e-02],
 [ 2.61610388e-03,  7.37324481e-02,  2.80398521e-02],
 [ 1.14763659e-01, -1.14368952e+00,  9.25030544e-02],
 [-1.17353574e-01, -1.14298274e+00,  9.60854266e-02],
 [-1.62284535e-04,  2.87602804e-01, -1.48171829e-02],
 [ 8.14608431e-02,  1.95481750e-01, -6.04975478e-03],
 [-7.91430834e-02,  1.92565283e-01, -1.05754332e-02],
 [ 4.98955543e-03,  3.52572414e-01,  3.65317875e-02],
 [ 1.72437770e-01,  2.25950646e-01, -1.49179062e-02],
 [-1.75155461e-01,  2.25116450e-01, -1.97185045e-02],
 [ 4.32050017e-01,  2.13178586e-01, -4.23743412e-02],
 [-4.28897421e-01,  2.11787231e-01, -4.11194829e-02],
 [ 6.81283645e-01,  2.22164620e-01, -4.35452575e-02],
 [-6.84195501e-01,  2.19559526e-01, -4.66786778e-02]])



def yaw_rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    z, o = np.zeros_like(angle), np.ones_like(angle)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(*np.shape(angle), 3, 3)



def fk_neutral_22(root_R, body_R, transl):
    """FK the neutral SMPL-H body. transl = pelvis world position."""
    T = root_R.shape[0]
    rest = SMPL_NEUTRAL_REST_JOINTS
    G_R = np.empty((T, 22, 3, 3))
    G_p = np.empty((T, 22, 3))
    G_R[:, 0] = root_R
    G_p[:, 0] = transl
    for j in range(1, 22):
        p = SMPL_22_PARENTS[j]
        off = rest[j] - rest[p]
        Rj = body_R[:, j - 1]
        G_R[:, j] = np.einsum("tij,tjk->tik", G_R[:, p], Rj)
        G_p[:, j] = G_p[:, p] + np.einsum("tij,j->ti", G_R[:, p], off)
    return G_p


