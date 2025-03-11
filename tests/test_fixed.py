## Importing these packages is specific for this simulation case
import os

import numpy as np
import pandas as pd
from pathlib import Path
import gc

import hmp
from hmp import simulations
from hmp.models import FixedEventModel
from hmp.models.base import EventProperties
from hmp.trialdata import TrialData


def test_fixed():
    sfreq = 100
    n_events = 3
    events = []
    raws = []
    event_id = {'stimulus':1}#trigger 1 = stimulus
    resp_id = {'response':5}
    
    for root, dirs, files in os.walk('tests/gen_data/'):
        if 'dataset' in root:
            events.append(np.load([x for x in files if 'events.npy' in x][0]))
            raws.append([x for x in files if 'raw.fif' in x][0])
            
    event_a = events[0]
    # Data reading
    epoch_data = hmp.utils.read_mne_data(raws, event_id=event_id, resp_id=resp_id, sfreq=sfreq,
            events_provided=events, verbose=True, reference='average', high_pass=1, low_pass=45,
                subj_idx=['a','b'],pick_channels='eeg', lower_limit_rt=0.01, upper_limit_rt=2 ) 
    epoch_data = epoch_data.assign_coords({'condition': ('participant', epoch_data.participant.data)})
    positions = simulations.simulation_positions()

    hmp_data = hmp.utils.transform_data(epoch_data, n_comp=2,)

        
   # Comparing to simulated data, asserting that results are the one simulated
    data_a = hmp.utils.participant_selection(hmp_data, 'a')
    event_properties = EventProperties.create_expected(sfreq=data_a.sfreq)
    trial_data = TrialData.from_standard_data(data=data_a, template=event_properties.template)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=False,)
    sim_source_times, true_pars, true_magnitudes, _ = \
        simulations.simulated_times_and_parameters(event_a, init_sim, trial_data)
    true_loglikelihood, true_estimates = init_sim.fit_transform(trial_data, parameters = np.array([true_pars]), magnitudes=np.array([true_magnitudes]),  verbose=True)
    true_topos = hmp.utils.event_topo(epoch_data, true_estimates.squeeze(), mean=True)
    init_sim = FixedEventModel(event_properties, n_events=n_events, maximization=True,)
    likelihood, estimates = init_sim.fit_transform(trial_data, verbose=True)
    test_topos = hmp.utils.event_topo(epoch_data, estimates.squeeze(), mean=True)
    assert (np.array(simulations.classification_true(true_topos,test_topos)) == np.array(([0,1,2],[0,1,2]))).all()
    
    assert np.isclose(np.sum(np.abs(true_topos.data - test_topos.data)), 0, atol=1e-4, rtol=0)
    assert np.isclose(likelihood, np.array(1.4300866), atol=1e-4, rtol=0)
