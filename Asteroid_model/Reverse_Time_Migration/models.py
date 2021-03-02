import numpy as np
import xarray as xr


def my_model(vp, rho, nx, nz):
    """
    """

    x = np.linspace(-500, +500, nx)
    y = np.linspace(-500, +500, nx)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # xarray dataset containing vp, rho
    ds = xr.Dataset(data_vars={"vp": (["x", "y"], vp),  "rho": (
        ["x", "y"], rho), }, coords={"x": x, "y": y},)

    # Transform velocity to SI units (m/s).
    # ds['vp'] *= 10000

    return ds
