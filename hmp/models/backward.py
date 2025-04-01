"""Models to estimate event probabilities."""

import gc
import itertools
import multiprocessing as mp
from itertools import cycle, product
from warnings import resetwarnings, warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import MultiIndex
from scipy.signal import correlate
from scipy.stats import norm as norm_pval

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

default_colors = ["cornflowerblue", "indianred", "orange", "darkblue", "darkgreen", "gold", "brown"]
from hmp.models.base import BaseModel
from hmp.models.fixedn import FixedEventModel

class BackwardEstimationModel(BaseModel):
    def __init__(self, *args, max_events=None, min_events=0, max_starting_points=1,
                 tolerance=1e-4, max_iteration=1e3, **kwargs):
        self.max_events = max_events
        self.min_events = min_events
        self.max_starting_points = max_starting_points
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.submodels = {}
        super().__init__(*args, **kwargs)

    def fit(
        self,
        trial_data,
        max_events=None,
        min_events=0,
        base_fit=None,
        max_starting_points=1,
        tolerance=1e-4,
        maximization=True,
        max_iteration=1e3,
        cpus=1,
    ):
        """Perform the backward estimation.

        First read or estimate max_event solution then estimate max_event - 1 solution by
        iteratively removing one of the event and pick the one with the highest
        loglikelihood

        Parameters
        ----------
        max_events : int
            Maximum number of events to be estimated, by default the output of
            hmp.models.hmp.compute_max_events()
        min_events : int
            The minimum number of events to be estimated
        base_fit : xarray
            To avoid re-estimating the model with maximum number of events it can be provided
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal
            number of events
        tolerance: float
            Tolerance applied to the expectation maximization in the EM() function
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        max_iteration: int
            Maximum number of iteration for the expectation maximization in the EM() function
        """
        if max_events is None and base_fit is None:
            max_events = self.compute_max_events(trial_data)
        print("MAX_EVENTS", max_events)
        if not base_fit:
            if max_starting_points > 0:
                print(
                    f"Estimating all solutions for maximal number of events ({max_events}) with 1 "
                    "pre-defined starting point and {max_starting_points - 1} starting points"
                )
            fixed_n_model = self.get_fixed_model(n_events=max_events)
            event_loo_results = fixed_n_model.fit_transform(trial_data, verbose=False)
        else:
            event_loo_results = [base_fit]
        max_events = event_loo_results[0].event.max().values + 1

        for n_events in np.arange(max_events - 1, min_events, -1):
            fixed_n_model = self.get_fixed_model(n_events)
            # only take previous model forward when it's actually fitting ok
            if event_loo_results[-1].loglikelihood.values != -np.inf:
                print(f"Estimating all solutions for {n_events} events")

                pars_prev = event_loo_results[-1].dropna("stage").parameters.values
                mags_prev = event_loo_results[-1].dropna("event").magnitudes.values

                events_temp, pars_temp = [], []

                for event in np.arange(n_events + 1):  # creating all possible solutions
                    events_temp.append(mags_prev[np.arange(n_events + 1) != event,])

                    temp_pars = np.copy(pars_prev)
                    temp_pars[event, 1] = (
                        temp_pars[event, 1] + temp_pars[event + 1, 1]
                    )  # combine two stages into one
                    temp_pars = np.delete(temp_pars, event + 1, axis=0)
                    pars_temp.append(temp_pars)

                if cpus == 1:
                    for i in range(len(events_temp)):
                            fixed_n_model.fit(
                                trial_data,
                                magnitudes=events_temp[i],
                                parameters=pars_temp[i],
                                tolerance=tolerance,
                                max_iteration=max_iteration,
                                verbose=False,
                            )
                else:
                    inputs = zip(
                        itertools.repeat(trial_data),
                        events_temp,  # magnitudes
                        pars_temp,  # parameters
                        itertools.repeat(None),  # parameters_to_fix
                        itertools.repeat(None),  # magnitudes_to_fix
                        itertools.repeat(False),  # verbose
                        itertools.repeat(1),  # cpus
                    )
                    with mp.Pool(processes=cpus) as pool:
                        pool.starmap(fixed_n_model.fit, inputs)

                # lkhs = [x.loglikelihood.values for x in event_loo_likelihood_temp]
                # event_loo_results.append(event_loo_likelihood_temp[np.nanargmax(lkhs)])

                # remove event_loo_likelihood
                # del event_loo_likelihood_temp
                # Force garbage collection
                gc.collect()

            else:
                print(
                    f"Previous model did not fit well. Estimating a neutral {n_events} event model."
                )
            self.submodels[n_events] = fixed_n_model
                # event_loo_results.append(
                #     self.get_fixed_model(n_events)
                # )
    
        # event_loo_results = xr.concat(event_loo_results, dim="n_events", fill_value=np.nan)
        # event_loo_results = event_loo_results.assign_coords(
        #     {"n_events": np.arange(max_events, min_events, -1)}
        # )
        # event_loo_results = event_loo_results.assign_attrs(method="backward")
        # if "sp_parameters" in event_loo_results.attrs:
        #     del event_loo_results.attrs["sp_parameters"]
        #     del event_loo_results.attrs["sp_magnitudes"]
        #     del event_loo_results.attrs["maximization"]
        # return event_loo_results
    def transform(self, trial_data):
        if len(self.submodels) == 0:
            raise ValueError("Model has not been (succesfully) fitted yet, no fixed models.")
        return xr.concat([m.transform(trial_data) for m in self.submodels],
                         dim="n_events", coords=list(self.submodels))


    def get_fixed_model(self, n_events):
        return FixedEventModel(
            self.events, self.distribution, n_events=n_events,
            starting_points=self.max_starting_points,
            tolerance=self.tolerance,
            max_iteration=self.max_iteration)
