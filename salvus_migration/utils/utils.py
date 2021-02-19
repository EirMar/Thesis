
import numpy as np
import xarray as xr
from scipy.signal import filtfilt


def get_layers(vp, rho, depth, x_max: float = 1000, nsmooth: int = 0):
    """Construct a dataset containing a lateral homogeneosus 1D seismic model
    VP, VS, coordinates

    :param vp: layer velocities, list
    :param rho: layer densities, list
    :param depth: layer depths, list
    :param x_max: Lateral extension in meters.
    :param nsmooth: Degree of smoothing in layers

    :return: xarray dataset
    """
    nx, ny = 1000, 1000
    x_min, x_max = 0, x_max
    y_min, y_max = 0, 2000

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    model = {"VP": vp,
             "RHO": rho,
             "DEPTH": depth}

    vp, rho = y.copy(), y.copy()
    for i in range(len(model["DEPTH"]) - 1):
        vp[(vp >= model["DEPTH"][i]) &
           (vp < model["DEPTH"][i+1])] = model["VP"][i+1]

        rho[(rho >= model["DEPTH"][i]) &
            (rho < model["DEPTH"][i+1])] = model["RHO"][i+1]

    if nsmooth > 0:
        smooth = np.ones(nsmooth) / nsmooth
        vp = filtfilt(smooth, 1, vp, axis=0)

    # Replicate modeles along x axis
    vp = np.tile(vp, (nx, 1))
    rho = np.tile(rho, (nx, 1))

    ds = xr.Dataset(data_vars={"vp": (["x", "y"], vp),
                               "rho": (["x", "y"], rho)},
                    coords={"x": x, "y": np.flip(y)})

    return ds


def get_gather(true_data):
    """Returs a numpy array of the salvus modelled data, shot gather.

    :param true_data: salvus synthetic data, EventDataCollection
            salvus.flow.collections.event_data_collection.EventDataCollection
    :return: shot gather, ndarray
    """
    ref = []
    for i, event in enumerate(true_data):
        shot_gather = []
        for j, rcv in enumerate(event.receiver_name_list):
            tr = event.get_receiver_data(receiver_name=rcv,
                                         receiver_field="phi_t")[0]
            shot_gather.append(tr.data)
        ref.append(shot_gather)

    R = np.asarray(ref).transpose()

    return R
