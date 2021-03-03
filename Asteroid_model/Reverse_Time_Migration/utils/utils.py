
import numpy as np
import xarray as xr
import h5py
import salvus

from scipy.signal import filtfilt

from typing import Union


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


def overwrite_receivers_h5(
    new_data: np.ndarray,
    events: Union[
        str,
        salvus.flow.collections.event_data_collection.EventDataCollection],
    field: str,
    sampling_rate_Hz: float,
    start_time_sec: float = 0.0,
    cmp: int = 0,

) -> None:
    """
    Overwrite the synthetic receiver data with a give np.ndarray. It access
    receivers.h5 created by Salvus and overwrites the data in the group
    /point/<fieldname>. This will keep all meta information about location,
    sampling rate, etc.

    Inspecting receivers.h5
        $ h5dump -H receivers.h5

    Use python methods dir() and vars() to inspect EventData object

    :param new_data: np.ndarray of shape [nt, n_rcv, n_src]
    :param events: EventDataCollection in salvus project
    :param field: Name of field to write (i.e. "phi", "phi_t", etc)
    :param sampling_rate_Hz: Sampling rate in Hz of new_data, along time axis
    :param start_time_sec: Starting time in seconds for new_data
    :param cmp: Index of the component to write, by default 0.
    """
    start_time_sec = np.array([start_time_sec])
    sampling_rate_Hz = np.array([sampling_rate_Hz])
    for i, event in enumerate(events):
        with h5py.File(event._data_object, mode="r+") as fh:
            # Get receivers ID
            receiver_id = fh["receiver_ids_ACOUSTIC_point"][...]
            # Overwrite point data
            fh["point/{}".format(field)][:, cmp, :] = \
                new_data[:, receiver_id, i].T
            fh["point"].attrs["sampling_rate_in_hertz"] = sampling_rate_Hz
            fh["point"].attrs["start_time_in_seconds"] = start_time_sec
