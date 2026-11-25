# imports 
import pandas as pd
import numpy as np


# 1. set parameters (beta, gamma, etc.)
# epiEngage uses a dictionary to store values
# we'll maybe have to use matrices for our different groups 

# epiEngage code: 
# (need to edit, e.g. we're not using a separate vaccinated compartment)
class MeaslesParameters(TypedDict):
    R0: float
    incubation_period: float
    infectious_period: float
    incubation_period_vaccinated: float
    infectious_period_vaccinated: float
    relative_infectiousness_vaccinated: float
    vaccine_efficacy: float
    school_contacts: float
    other_contacts: float
    population: list[int]
    I0: list[int]
    vax_prop: list[float]
    threshold_values: list[int]
    sim_duration_days: int
    time_step_days: float
    is_stochastic: bool
    simulation_seed: int


# https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html
# Incubation: 11.5 days, so first symptoms 11.5 days since t0
# Rash starts: 3 days after first symptoms, so 14.5 since t0
# Infectious: 4 days before rash (so 10.5 days since t0), 4 days after (18.5 days since t0)
DEFAULT_MSP_PARAMS: MeaslesParameters = \
    {
        'R0': 15.0,  # transmission rate
        'incubation_period': 10.5,
        'infectious_period': 5,
        'incubation_period_vaccinated': 10.5,
        'infectious_period_vaccinated': 5,
        'relative_infectiousness_vaccinated': 0.05,  # 0 means no infection from vaccinated
        'vaccine_efficacy': 0.997,  # 1.0 means perfect protection
        'school_contacts': 5.63424,
        'other_contacts': 2.2823,
        'population': [500],
        'I0': [1],
        'vax_prop': [0.85],  # number between 0 and 1
        'threshold_values': [10],  # outbreak threshold, only first value used for now
        'sim_duration_days': 250,
        'time_step_days': 0.25,
        'is_stochastic': True,  # False for deterministic,
        "simulation_seed": 147125098488,
        "save_all_transitions": False
    }


# 2. create the simulator 

# epiEngage code:
class MetapopSEIR:

    def __init__(self,
                 params: MeaslesParameters):
        """
        Initialize metapopulation SEPIR model

        Parameters:
        params: MeaslesParameters instance with disease parameters and movement rates
        N: array of population sizes
        I0: array of initial infectious cases
        T: total simulation time
        dt: time step size
        """

        self.save_all_transitions = params["save_all_transitions"]

        self.n_pop = len(params['population'])  # number of populations
        self.N = np.array(params['population'])  # population size
        self.steps_per_day = int(1.0 / params['time_step_days'])
        self.dt = params['time_step_days']
        self.n_steps = 1 + int(
            params['sim_duration_days'] / self.dt)  # number of time steps

        # Stochastic simulations
        self.is_stochastic = params['is_stochastic']
        if self.is_stochastic:
            # self.RNG = np.random.Generator(PCG64(params["simulation_seed"]))
            self.RNG = np.random.Generator(np.random.MT19937(seed=params["simulation_seed"]))
        else:
            self.RNG = None

        # Number of contacts
        self.school_contacts = params['school_contacts']
        self.other_contacts_base = params['other_contacts']
        self.total_contacts = self.school_contacts + self.other_contacts_base

        # Disease parameters
        beta = params['R0'] / (params["infectious_period"] * self.total_contacts)

        self.beta = beta  # transmission rate
        self.sigma = 1.0 / params["incubation_period"]  # rate of progression from E to I
        self.gamma = 1.0 / params["infectious_period"]  # recovery rate
        self.sigma_vax = 1.0 / params["incubation_period_vaccinated"]  # rate of progression from E to I for vaccinated
        self.gamma_vax = 1.0 / params["infectious_period_vaccinated"]  # recovery rate for vaccinated

        self.beta_vax = params['relative_infectiousness_vaccinated'] * params['R0'] / \
                        (params["infectious_period_vaccinated"] * self.total_contacts)
        self.vaccine_efficacy = params['vaccine_efficacy']

        # Time array
        self.t = np.linspace(0, params['sim_duration_days'], self.n_steps)

        self.params = params

        # Initialize compartments as 2D arrays [time_step, population]
        self.S_unvax = np.zeros((self.n_steps, self.n_pop))
        self.E_unvax = np.zeros((self.n_steps, self.n_pop))
        self.I_unvax = np.zeros((self.n_steps, self.n_pop))
        self.R_unvax = np.zeros((self.n_steps, self.n_pop))

        self.S_vax = np.zeros((self.n_steps, self.n_pop))  # vaccinated
        self.E_vax = np.zeros((self.n_steps, self.n_pop))  # vaccinated breakthrough infections
        self.I_vax = np.zeros((self.n_steps, self.n_pop))  # vaccinated breakthrough infections
        self.R_vax = np.zeros((self.n_steps, self.n_pop))  # vaccinated breakthrough infections

        self.S_unvax_to_E_unvax = np.zeros((self.n_steps, self.n_pop))
        self.S_vax_to_E_vax = np.zeros((self.n_steps, self.n_pop))

        # Add storage of transition variables
        if self.save_all_transitions:

            self.E_unvax_to_I_unvax = np.zeros((self.n_steps, self.n_pop))
            self.I_unvax_to_R_unvax = np.zeros((self.n_steps, self.n_pop))

            self.E_vax_to_I_vax = np.zeros((self.n_steps, self.n_pop))
            self.I_vax_to_R_vax = np.zeros((self.n_steps, self.n_pop))

        params = self.params
        N = self.N

        self.I_unvax[0] = np.array(params['I0'])
        self.S_vax[0] = [
            int(self.N[i_pop] * params['vax_prop'][i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S_unvax[0] = [
            min(self.S_unvax[0, i_pop], self.N[i_pop] - self.I_unvax[0, i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S_unvax[0] = self.N - self.I_unvax[0] - self.S_vax[0]

        # Current step tracker
        self.current_step = 0

    def get_compartment_transition(self,
                                   rate: float,
                                   compartment_count: int) -> float:
        """
        For a given compartment and a given rate, this calculates the number
        of individuals transitioning out of the compartment.
        The calculations is either deterministic or stochastic.

        Parameters
        ----------
        rate : double
            Force of infection for new infected, rate out of compartment
            otherwise.
        compartment_count : int
            Number of individuals in compartment.

        Returns
        -------
        Number of individuals leaving compartment.

        """

        total_rate = rate * compartment_count

        if total_rate > 0:
            if self.is_stochastic:
                delta = self.RNG.poisson(total_rate)
            else:
                delta = total_rate
            delta = min(delta, compartment_count)
        else:
            delta = 0

        return delta

    def calculate_compartment_updates(self,
                                      ix_pop: int) -> dict:
        """
        Calculates changes in compartments for population ix_pop.

        Parameters
        ----------
        ix_pop : int
            Index of population.

        Returns
        -------
        dict:
            Keys are: "dS, dE, dI, dR, dS_unvax_out, dE_unvax_out, dI_unvax_out",
            Values are the quantities at time `self.current_step`.

        """
        t = self.current_step
        dt = self.dt

        force_of_infection = self.total_contacts * dt * \
                             (self.beta * self.I_unvax[t, ix_pop] + self.beta_vax * self.I_vax[t, ix_pop]) / self.N[ix_pop]
        force_of_infection_vaccinated = \
            force_of_infection * (1 - self.vaccine_efficacy)

        dS_unvax_out = self.get_compartment_transition(force_of_infection, self.S_unvax[t, ix_pop])
        dE_unvax_out = self.get_compartment_transition(self.sigma * dt, self.E_unvax[t, ix_pop])
        dI_unvax_out = self.get_compartment_transition(self.gamma * dt, self.I_unvax[t, ix_pop])
        dS_vax_out = self.get_compartment_transition(force_of_infection_vaccinated, self.S_vax[t, ix_pop])
        dE_vax_out = self.get_compartment_transition(self.sigma_vax * dt, self.E_vax[t, ix_pop])
        dI_vax_out = self.get_compartment_transition(self.gamma_vax * dt, self.I_vax[t, ix_pop])

        dS = -dS_unvax_out
        dE = dS_unvax_out - dE_unvax_out
        dI = dE_unvax_out - dI_unvax_out
        dR = dI_unvax_out
        dS_vax = -dS_vax_out
        dE_vax = dS_vax_out - dE_vax_out
        dI_vax = dE_vax_out - dI_vax_out
        dR_vax = dI_vax_out

        updates_dict = {
            "dS_unvax": dS, "dE_unvax": dE, "dI_unvax": dI, "dR_unvax": dR,
            "dS_vax": dS_vax, "dE_vax": dE_vax, "dI_vax": dI_vax, "dR_vax": dR_vax,
            "dS_unvax_out": dS_unvax_out, "dE_unvax_out": dE_unvax_out, "dI_unvax_out": dI_unvax_out,
            "dS_vax_out": dS_vax_out, "dE_vax_out": dE_vax_out, "dI_vax_out": dI_vax_out,
        }

        return updates_dict

    def step(self):
        """Calculate one time step using Euler's method"""
        if self.current_step >= self.n_steps - 1:
            return False

        # Loop through populations
        for ix_pop in range(self.n_pop):
            updates = self.calculate_compartment_updates(ix_pop)
            current_step = self.current_step

            # Creates view of same array in memory, NOT a copy
            S_unvax, E_unvax, I_unvax, R_unvax, S_vax, E_vax, I_vax, R_vax = self.S_unvax, self.E_unvax, self.I_unvax, self.R_unvax, \
                                             self.S_vax, self.E_vax, self.I_vax, self.R_vax
            S_unvax_to_E_unvax, S_vax_to_E_vax = self.S_unvax_to_E_unvax, self.S_vax_to_E_vax
            if self.save_all_transitions:
                E_unvax_to_I_unvax, I_unvax_to_R_unvax, E_vax_to_I_vax, I_vax_to_R_vax = \
                    self.E_unvax_to_I_unvax, self.I_unvax_to_R_unvax, self.E_vax_to_I_vax, self.I_vax_to_R_vax

            # Update compartments using Euler's method
            S_unvax[current_step + 1, ix_pop] = S_unvax[current_step, ix_pop] + updates["dS_unvax"]
            E_unvax[current_step + 1, ix_pop] = E_unvax[current_step, ix_pop] + updates["dE_unvax"]
            I_unvax[current_step + 1, ix_pop] = I_unvax[current_step, ix_pop] + updates["dI_unvax"]
            R_unvax[current_step + 1, ix_pop] = R_unvax[current_step, ix_pop] + updates["dR_unvax"]
            S_vax[current_step + 1, ix_pop] = S_vax[self.current_step, ix_pop] + updates["dS_vax"]
            E_vax[current_step + 1, ix_pop] = E_vax[current_step, ix_pop] + updates["dE_vax"]
            I_vax[current_step + 1, ix_pop] = I_vax[current_step, ix_pop] + updates["dI_vax"]
            R_vax[current_step + 1, ix_pop] = R_vax[current_step, ix_pop] + updates["dR_vax"]

            S_unvax_to_E_unvax[current_step + 1, ix_pop] = updates["dS_unvax_out"]
            S_vax_to_E_vax[current_step + 1, ix_pop] = updates["dS_vax_out"]

            # Also update transition variables history
            if self.save_all_transitions:
                E_unvax_to_I_unvax[current_step + 1, ix_pop] = updates["dE_unvax_out"]
                I_unvax_to_R_unvax[current_step + 1, ix_pop] = updates["dI_unvax_out"]
                E_vax_to_I_vax[current_step + 1, ix_pop] = updates["dE_vax_out"]
                I_vax_to_R_vax[current_step + 1, ix_pop] = updates["dI_vax_out"]

        self.current_step += 1
        return True

    def simulate(self):
        """Run simulation for all time steps"""
        while self.step():
            pass

    def clear(self):

        self.S_unvax = np.zeros((self.n_steps, self.n_pop))
        self.E_unvax = np.zeros((self.n_steps, self.n_pop))
        self.I_unvax = np.zeros((self.n_steps, self.n_pop))
        self.R_unvax = np.zeros((self.n_steps, self.n_pop))
        self.S_vax = np.zeros((self.n_steps, self.n_pop))
        self.E_vax = np.zeros((self.n_steps, self.n_pop))
        self.I_vax = np.zeros((self.n_steps, self.n_pop))
        self.R_vax = np.zeros((self.n_steps, self.n_pop))

        self.S_unvax_to_E_unvax = np.zeros((self.n_steps, self.n_pop))
        self.S_vax_to_E_vax = np.zeros((self.n_steps, self.n_pop))

        if self.save_all_transitions:
            self.E_unvax_to_I_unvax = np.zeros((self.n_steps, self.n_pop))
            self.I_unvax_to_R_unvax = np.zeros((self.n_steps, self.n_pop))
            self.E_vax_to_I_vax = np.zeros((self.n_steps, self.n_pop))
            self.I_vax_to_R_vax = np.zeros((self.n_steps, self.n_pop))

        params = self.params
        N = self.N

        self.I_unvax[0] = np.array(params['I0'])
        self.S_vax[0] = [
            int(self.N[i_pop] * params['vax_prop'][i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S_unvax[0] = [
            min(self.S_unvax[0, i_pop], self.N[i_pop] - self.I_unvax[0, i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S_unvax[0] = self.N - self.I_unvax[0] - self.S_vax[0]

        self.current_step = 0

    def plot_results(self):
        """Plot the results for all populations"""
        fig, axes = plt.subplots(self.n_pop, 1, figsize=(10, 6 * self.n_pop))
        if self.n_pop == 1:
            axes = [axes]

        for pop in range(self.n_pop):
            ax = axes[pop]
            vaccinated_linestyle = 'dashed'
            ax.plot(self.t, self.S[:, pop], label='Susceptible', color='blue')
            ax.plot(self.t, self.E[:, pop], label='Exposed', color='orange')
            ax.plot(self.t, self.I[:, pop], label='Infectious', color='red')
            ax.plot(self.t, self.R[:, pop], label='Recovered', color='green')
            ax.plot(self.t, self.S_vax[:, pop], label='Vaccinated', color='blue', linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.E_vax[:, pop], label='Vaccinated Exposed', color='orange',
                    linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.I_vax[:, pop], label='Vaccinated Infectious', color='red',
                    linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.R_vax[:, pop], label='Vaccinated Recovered', color='green',
                    linestyle=vaccinated_linestyle)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Number of individuals')
            ax.set_title(f'Population {pop + 1} SEPIR Dynamics')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

