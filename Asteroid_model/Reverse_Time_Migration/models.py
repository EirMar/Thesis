import numpy as np
import xarray as xr


def my_model(vp, rho, max_x, max_y, acoustic=True):
    """
    """
    nx, ny = vp.shape
    x = np.linspace(-max_x, +max_x, nx)
    y = np.linspace(-max_y, +max_y, nx)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # xarray dataset containing vp, rho
    ds = xr.Dataset(
        data_vars={"vp": (["x", "y"], vp),
                   "rho": (["x", "y"], rho), },
        coords={"x": x, "y": y},)

    if acoustic:
        # Transform velocity to SI units (m/s).
        ds['vp'] *= 10000

    return ds
