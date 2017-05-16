from backend import *


def angular_distance_rad(y_target, y_predict):
    return angdist(compass2rad_tf(y_target), compass2rad_tf(y_predict))


def angular_distance_deg(y_target, y_predict):
    return 180 * angular_distance_rad(y_target, y_predict) / np.pi


def angular_distance_per(y_target, y_predict):
    return angdist(compass2rad(y_target), compass2rad(y_predict)) / np.pi


losses = {
    "adr": angular_distance_rad,
    "angular distance rad": angular_distance_rad,
    "add": angular_distance_deg,
    "angular distance degrees": angular_distance_deg,
    "adp": angular_distance_per,
    "angular distance percentage": angular_distance_per
}


def get_loss(name):
    assert name in losses.keys(), "Name of loss function does not exist."
    return losses[name]
