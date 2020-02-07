import numpy as np
import math

def heading_elevation_feat(horizon_views=12, tile=32):
    """ Get heading and elevation features relatively from the current
    heading and elevation """

    assert 360 % horizon_views == 0
    angle = 360 / horizon_views

    rel_heading = np.array(range(0, horizon_views))
    rel_sin_phi = [0] * horizon_views
    rel_cos_phi = [0] * horizon_views
    for i, x in enumerate(rel_heading):
        rel_heading[i] = x * angle
        if rel_heading[i] < 0:
            rel_heading[i] = rel_heading[i] + 360

        rel_sin_phi[i] = math.sin(rel_heading[i] / 180 * math.pi)
        rel_cos_phi[i] = math.cos(rel_heading[i] / 180 * math.pi)

    # duplicate the heading features for 3 levels
    rel_sin_phi = np.array(rel_sin_phi * 3)
    rel_cos_phi = np.array(rel_cos_phi * 3)

    rel_elevation = np.array([-1, 0, 1])
    rel_sin_theta = [0] * 3
    rel_cos_theta = [0] * 3
    for i, x in enumerate(rel_elevation):
        rel_elevation[i] = x * angle

        rel_sin_theta[i] = math.sin(rel_elevation[i] / 180 * math.pi)
        rel_cos_theta[i] = math.cos(rel_elevation[i] / 180 * math.pi)
    rel_sin_theta = np.repeat(rel_sin_theta, horizon_views)
    rel_cos_theta = np.repeat(rel_cos_theta, horizon_views)

    feat = np.stack([rel_sin_phi, rel_cos_phi, rel_sin_theta, rel_cos_theta], axis=0)
    feat = np.repeat(feat, tile, axis=0)

    return np.transpose(feat)

if __name__ == "__main__":
    print(heading_elevation_feat())
    print(heading_elevation_feat().shape)