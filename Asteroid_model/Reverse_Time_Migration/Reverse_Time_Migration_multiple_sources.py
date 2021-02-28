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

# Variable used in the notebook to determine which site
# is used to run the simulations.

#SALVUS_FLOW_SITE_NAME=os.environ.get('SITE_NAME','local')
SALVUS_FLOW_SITE_NAME=os.environ.get('eejit','eejit')


# In[ ]:


file = "vel1_copy.bin"

dt = np.dtype([('time', '<u2'),('time1', '<u2'),('time2', np.float32),('time3', np.float32)])

data = np.fromfile(file, dtype=np.float32, count=-1, sep='', offset=0)


# In[ ]:


my_array_rel_perm=data.reshape(3000,3000) 

#Make an array that has the same size as velocity
#Density constant ~ 1000 kg/m**3

my_array_rho=np.full((3000,3000),1000, dtype=int)
#print(my_array_rho)


# In[ ]:


def my_model():
    nx, nz = 3000, 3000
    x = np.linspace(-4000, +4000, nx)
    y = np.linspace(-4000, +4000, nx)
    xx, yy = np.meshgrid(x, y, indexing = "ij")
    
    #put the array elements into the appropriate part of the model xarray structure
    ds = xr.Dataset ( data_vars= {"vp": (["x", "y"], my_array_rel_perm), 
                                  "rho": (["x", "y"], my_array_rho),},
                     coords={"x": x, "y": y},)
    
    #Transform velocity to SI units (m/s).
    ds['vp'] *=10000

    return ds
true_model = my_model()


# In[ ]:


wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)
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


# Smooth the model

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
p = sn.Project.from_volume_model(path="salvus_project", volume_model=vm)


# In[ ]:


wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)
mesh_frequency =  wavelet.center_frequency


sources = sn.simple_config.source.cartesian.collections.ScalarPoint2DRing(
    source_time_function= wavelet, x=0, y=0, radius=3500, count=90, f=1.0
)

# Sources
#sources = sn.simple_config.source.cartesian.ScalarPoint2D( 
#     source_time_function= wavelet, x=-100.0, y=3500.0, f=1)



# Receivers
receivers = sn.simple_config.receiver.cartesian.collections.RingPoint2D(
       x=0, y=0, radius=3500, count=380, fields=["phi"]
)


    
p += sn.EventCollection.from_sources(sources=sources, receivers=receivers)


# In[ ]:


# Waveform Simulation Configuration
wsc = sn.WaveformSimulationConfiguration(end_time_in_seconds=6.0)


# In[ ]:


# Event configuration
ec = sn.EventConfiguration(
    waveform_simulation_configuration=wsc,
    wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0))


# In[ ]:


p += sn.UnstructuredMeshSimulationConfiguration(
    name="smooth_model",
    unstructured_mesh=smooth_model,
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)))
  
p += sn.SimulationConfiguration(
    name="target_model",
    elements_per_wavelength=2,
    tensor_order=2,
    max_frequency_in_hertz = 30.0,
    model_configuration=sn.ModelConfiguration(background_model=None, volume_models=["true_model"]),
    event_configuration=sn.EventConfiguration(
        waveform_simulation_configuration=wsc,
        wavelet=sn.simple_config.stf.Ricker(center_frequency=10.0)))


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


# In[ ]:





# In[ ]:





# In[ ]:


# %%
# ------------------------------------------------------------------------------
# COLLECT WAVEFORM DATA
# ------------------------------------------------------------------------------
true_data = p.waveforms.get(
    data_name="smooth_model", events=p.events.get_all())

direct_wave = p.waveforms.get(
    data_name="target_model", events=p.events.get_all())

Rt = ut.get_gather(true_data).reshape
Rd = ut.get_gather(direct_wave).reshape
R = Rt - Rd


# In[ ]:





# In[ ]:





# In[ ]:


p += sn.InverseProblemConfiguration(
    name="my_inversion",
    prior_model="smooth_model",
    events=[e.event_name for e in p.events.get_all()],
    mapping=sn.Mapping(scaling="absolute", inversion_parameters=["VP", "RHO"]), 
    #preconditioner=sn.ConstantSmoothing({"VP": 0.01, "RHO": 0.01}),
    method=sn.TrustRegion(initial_trust_region_linf=10.0),
    misfit_configuration="migration",
    job_submission=sn.SiteConfig(
        site_name="eejit",wall_time_in_seconds_per_job=1, ranks_per_job=4
    ),
)


# In[ ]:


p.inversions.add_iteration(inverse_problem_configuration="my_inversion")


# In[ ]:


p.inversions.resume(
    inverse_problem_configuration="my_inversion",
)


# In[ ]:


p.viz.nb.iteration(
    inverse_problem_configuration="my_inversion", iteration_id=0
)


# In[ ]:


p.inversions.iterate(
    inverse_problem_configuration="my_inversion",
    timeout_in_seconds=380,
    ping_interval_in_seconds=10,
)

p.viz.nb.inversion(inverse_problem_configuration="my_inversion")


# In[ ]:




