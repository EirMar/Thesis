#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[ ]:


import os
import numpy as np

import time

import xarray as xr

from pathlib import Path 
import pathlib

import salvus.namespace as sn
from salvus.flow import simple_config as config

import salvus.mesh.unstructured_mesh as um
import salvus.mesh.structured_grid_2D as sg2d

from salvus.flow import simple_config as config
from salvus.toolbox import toolbox


import salvus.namespace as sn

import matplotlib.pyplot as plt

SALVUS_FLOW_SITE_NAME=os.environ.get('eejit','eejit')


# In[ ]:


#Import the model - Relative Permittivity values

file = "vel1_copy.bin"

dt = np.dtype([('time', '<u2'),('time1', '<u2'),('time2', np.float32),('time3', np.float32)])

data = np.fromfile(file, dtype=np.float32, count=-1, sep='', offset=0)

array_eps_rel = data.reshape (3000,3000) 

# Density
array_rho = np.full((3000,3000),1000, dtype=int)

#Magnetic Permeability
array_mu = np.full((3000,3000),1, dtype=int)


# In[ ]:


import math

c = 3e8 #speed of light
mu = 1
#v_radar = c / math.sqrt(mu * array_eps_rel)

array_eps_rel_sqrt=np.sqrt(array_eps_rel)


v_radar = c / (mu * array_eps_rel_sqrt)


# In[ ]:


def my_model():
    nx, nz = 3000, 3000
    x = np.linspace(-500, +500, nx)
    y = np.linspace(-500, +500, nx)
    xx, yy = np.meshgrid(x, y, indexing = "ij")
    
    #put the array elements into the appropriate part of the model xarray structure
    ds = xr.Dataset ( data_vars= {"vp": (["x", "y"], v_radar),  "rho": (["x", "y"], array_rho),}, coords={"x": x, "y": y},)
    
    return ds
true_model = my_model()


# In[ ]:


wavelet=sn.simple_config.stf.Ricker(center_frequency=15.0e6)
mesh_frequency = wavelet.center_frequency

num_absorbing_layers = 10
absorbing_side_sets = ["x0", "x1", "y0", "y1"]

# Create a mesh from xarray Dataset
mesh = toolbox.mesh_from_xarray(
    model_order=4,
    data=true_model,
    slowest_velocity="vp",
    maximum_frequency=mesh_frequency,
    elements_per_wavelength=1.5,
    absorbing_boundaries=(absorbing_side_sets, num_absorbing_layers))


# In[ ]:


mesh.write_h5("true_model.h5")

import salvus.namespace as sn
from salvus.opt import smoothing

smoothing_config = sn.ModelDependentSmoothing(
 smoothing_lengths_in_wavelengths={
 "VP": 2.0,
 "RHO": 2.0
 },
 reference_frequency_in_hertz= 15.0,
 reference_model="true_model.h5",
 reference_velocities={"VP": "VP", "RHO": "RHO"},
 ).get_smoothing_config()

smooth_model = smoothing.run(
    site_name=SALVUS_FLOW_SITE_NAME,
    ranks_per_job=4,
    model=mesh,
    smoothing_config=smoothing_config,
    wall_time_in_seconds_per_job=1
)

 

smooth_model


# In[ ]:


# ------------------------------------------------------------------------------
# CREATE NEW SALVUS PROJECT
# ------------------------------------------------------------------------------
get_ipython().system('rm -rf salvus_project')
vm = sn.model.volume.cartesian.GenericModel(
    name="true_model", data=true_model)
p = sn.Project.from_volume_model(path="salvus_project_EM", volume_model=vm)


# In[ ]:


wavelet=sn.simple_config.stf.Ricker(center_frequency=15.0e6)
mesh_frequency = wavelet.center_frequency

#srcs = sn.simple_config.source.cartesian.collections.ScalarPoint2DRing(
#    x=0, y=0, radius=450, count=90, f=1.0
#)

srcs = sn.simple_config.source.cartesian.ScalarPoint2D( 
     source_time_function=wavelet, x=0.0, y=450.0, f=1)


recs = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
       x=0, y=0, radius=450, count=380, fields=["phi"]
)
    
p += sn.EventCollection.from_sources(sources=srcs, receivers=recs)


# In[ ]:


wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=9.5e-6)


# In[ ]:


ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=15.0e6),
)


# In[ ]:


p += sn.UnstructuredMeshSimulationConfiguration(
    name="smooth_model",
    unstructured_mesh=smooth_model,
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=15.0e6)))
  
p += sn.SimulationConfiguration(
    name="target_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz = 30.0,
    model_configuration=sn.ModelConfiguration(background_model=None, volume_models=["true_model"]),
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=15.0e6)))


# In[ ]:


sim = config.simulation.Waveform (mesh=mesh,sources=sources,receivers=receivers)


# In[ ]:


#sim.output.point_data.format = "hdf5"


# In[ ]:


# Save the volumetric wavefield for visualization purposes.
#sim.output.volume_data.format = "hdf5"
#sim.output.volume_data.filename = "output.h5"
#sim.output.volume_data.fields = ["phi"]
#sim.output.volume_data.sampling_interval_in_time_steps = 1000


# In[ ]:


# %%
# ------------------------------------------------------------------------------
# RUN FORWARD SIMULATION
# ------------------------------------------------------------------------------


for sim, store in zip(["smooth_model", "target_model"], [True, False]):
    p.simulations.launch(
        simulation_configuration=sim,
        events=p.events.get_all(),
        site_name=SALVUS_FLOW_SITE_NAME,
        ranks_per_job=4,
        verbosity=0,
        store_adjoint_checkpoints=True,
        wall_time_in_seconds_per_job=1,)

# %%
for sim in ["smooth_model", "target_model"]:
    p.simulations.query(
        simulation_configuration=sim,
        events=p.events.list(),
        verbosity=2,
        block=True,)


# In[ ]:


# %%
# ------------------------------------------------------------------------------
# Compute adjoint sources and gradients
# ------------------------------------------------------------------------------
def misfit_func(data_synthetic: np.ndarray,
                data_observed: np.ndarray,
                sampling_rate_in_hertz: float):
    adj_src = data_synthetic - data_observed
    return 1.0, adj_src


p += sn.MisfitConfiguration(
    name="migration",
    observed_data="target_model",
    misfit_function=misfit_func,
    receiver_field="phi")


# In[ ]:


# Computed misfits before running adjoint simulation
misfits = None
while not misfits:
    misfits = p.actions.inversion.compute_misfits(
        simulation_configuration="smooth_model",
        misfit_configuration="migration",
        store_checkpoints=False,
        events=p.events.list(),
        ranks_per_job=4,
        site_name="eejit",
        wall_time_in_seconds_per_job=1,
        verbosity=2,
    )
    time.sleep(5.0)

print(misfits)


# In[ ]:


# %%
# ------------------------------------------------------------------------------
# RUN ADJOINT SIMULATION
# ------------------------------------------------------------------------------
p.simulations.launch_adjoint(
    simulation_configuration="smooth_model",
    misfit_configuration="migration",
    events=p.events.list(),
    site_name="eejit",
    ranks_per_job=4,
    wall_time_in_seconds_per_job=1,
    verbosity=True,)


# In[ ]:


# %%
p.simulations.query(
    simulation_configuration="smooth_model",
    events=p.events.list(),
    misfit_configuration="migration",
    ping_interval_in_seconds=1.0,
    verbosity=2,
    block=True,)


# In[ ]:


# %%
gradient = p.actions.inversion.sum_gradients(
    simulation_configuration="smooth_model",
    misfit_configuration="migration",
    events=p.events.list(),)


# In[ ]:


# gradient.write_h5("tot_gradient/gradient.h5")
gradient
