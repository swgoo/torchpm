from dataclasses import dataclass
import enum
import functools
import os
import pickle
import shelve
import time
from typing import List, Dict, Tuple, Union
import typing
import torch as tc
import sympy as sym
import sympytorch as spt
from torch import nn
import json
import asyncio

@enum.unique
class CompartmentDistributionMatrix(enum.Enum) :
    ONE_COMP_DIST   = True  # type: ignore
    TWO_COMP_DIST   = ((True, True), (True, False))  # type: ignore
    THREE_COMP_DIST = ((True, True, True), (True, False, False), (True, False, False))  # type: ignore

@dataclass(frozen=True, eq=True)
class ModelConfig:
    distribution_matrix: Union[Tuple[Tuple[bool]], bool] 
    has_depot: bool = False
    transit: int = 0
    observed_compartment_num : int = 0
    administrated_compartment_num : int = 0
    is_infusion: bool = False
    is_delay_time: bool = False

DB : Dict[ModelConfig, typing.Type[nn.Module]] = {}  # type: ignore
def __init__() :
    twoCompartmentInfusionKey = ModelConfig(
                    CompartmentDistributionMatrix.TWO_COMP_DIST.value,  # type: ignore
                    is_infusion = True)
    DB[twoCompartmentInfusionKey] = TwoCompartmentInfusion

class CompartmentModel(nn.Module) :
    DB_FILE_NAME = "ode.db"

    def __init__(self, 
            model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

        self.distribution_matrix = []
        if isinstance(model_config.distribution_matrix, bool) :
            self.distribution_matrix = [[model_config.distribution_matrix]]
        elif isinstance(model_config.distribution_matrix, Tuple) :
            for row in list(model_config.distribution_matrix) :
                self.distribution_matrix.append(list(row))

        self.is_infusion = model_config.is_infusion
        self.observed_compartment_num = model_config.observed_compartment_num
        self.administrated_compartment_num = model_config.administrated_compartment_num
        self.depot_compartment_num = model_config.administrated_compartment_num
        self.has_depot = model_config.has_depot
        self.transit = model_config.transit
        self.is_delay_time = model_config.is_delay_time

        self._make_distribution_matrix()

    def _make_distribution_matrix(self) -> None:
        if self.has_depot :
            self.distribution_matrix.append([False] * len(self.distribution_matrix))
            for row in self.distribution_matrix :
                row.append(False)
            self.distribution_matrix[-1][self.administrated_compartment_num] = True
        
            self.depot_compartment_num = len(self.distribution_matrix) - 1

        self.transit_compartment_nums = []
        if self.has_depot and self.transit > 0 :
            self.distribution_matrix[-1][self.administrated_compartment_num] = False
            length = len(self.distribution_matrix)
            for row in range(self.transit) :
                self.distribution_matrix.append([False] * length)
            
            for row in self.distribution_matrix :
                for i in range(self.transit):
                    row.append(False)
            
            #depot to transit 0
            
            self.distribution_matrix[self.depot_compartment_num][self.depot_compartment_num+1] = True
            self.transit_compartment_nums.append(self.depot_compartment_num+1)
            transit_start = self.depot_compartment_num + 1 
            for i in range(transit_start, transit_start + self.transit - 1):
                self.distribution_matrix[i][i+1] = True
                self.transit_compartment_nums.append(i)
            self.distribution_matrix[-1][self.administrated_compartment_num] = True
    
    def get_distribution_matrix(self) -> List[List[bool]]:
        if self.distribution_matrix == [] :
            self._make_distribution_matrix()
        return self.distribution_matrix

class NumericCompartmentModel(CompartmentModel) :
    
    def forward(self, y, t, **variables : tc.Tensor) :

        comps_num = len(self.distribution_matrix)

        elimination_rate = tc.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = variables['k_' + str(i) + str(i)] if self.distribution_matrix[i][i] else 0

        distribution_rate = tc.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_matrix[i][j]:
                    continue
                distribution_rate[i,j] = variables['k_'+str(i)+str(j)]
        
        dcdt_matrix = distribution_rate.t()  \
                    - tc.diag(distribution_rate @ tc.ones(comps_num, 1)) \
                    - elimination_rate

        if self.is_delay_time and variables['delay_time'] > t :
            return tc.zeros_like(y)
        else :
            return dcdt_matrix @ y

class SymbolicCompartmentModel(CompartmentModel) :
    
    def __init__(self, model_config : ModelConfig, timeout = 30) -> None:
        
        super().__init__(model_config=model_config)

        if self.is_infusion :
            t_sym = sym.symbols('t', real=True, negative = False, finite = True)
            r_sym, dose_sym = sym.symbols('RATE, AMT', real=True, positive = True, finite = True)
            initial_states_infusion = self._get_initial_states(is_infusion=self.is_infusion)
            cs_infusion = self._solve_linode(initial_states_infusion, is_infusion=True, timeout=timeout)

            funcs = self._get_functions()
            initial_states = self._get_initial_states(is_infusion=self.is_infusion)
            for i, func in enumerate(funcs) :
                initial_states[func(0)] = cs_infusion[i].subs({t_sym: dose_sym/r_sym})
            cs = self._solve_linode(initial_states, is_infusion=False, timeout=timeout)
            
            for i in range(len(cs)) :
                cs[i] = cs[i].subs({t_sym: t_sym - dose_sym/r_sym})

            self.infusion_model = spt.SymPyModule(expressions=cs_infusion)
        
        else:
            initial_states = self._get_initial_states(is_infusion=self.is_infusion) 
            cs = self._solve_linode(initial_states, is_infusion=self.is_infusion, timeout=timeout)
        
        self.model = spt.SymPyModule(expressions=cs)
    
    def _check_square_matrix(self, m : list[list[bool]], error_massage) :
        length = len(m)
        for row in m :
            if len(row) != length :
                raise RuntimeError(error_massage)

    def _get_functions(self) :
        comps_num = len(self.distribution_matrix)
        return [sym.symbols("c_"+ str(i), cls=sym.Function, negative = False, real = True, finite = True) for i in range(comps_num)]  # type: ignore

    def _get_initial_states(self, is_infusion : bool = False) :
        """
        Returns:
            ics: initial states of compartments
        """
        d = sym.symbols('AMT', real = True, Positve = True, finite = True) #dose
        result = {}

        funcs = self._get_functions()  
        funcs = [func(0) for func in funcs]
        
        for i, func in enumerate(funcs) :
            if not is_infusion and i == self.depot_compartment_num  :
                result[func] = d
            else :
                result[func] = 0

        return result

    def _solve_linode(self, initial_states, is_infusion : bool = False, timeout : int = 30) -> List[sym.Eq]:
        """
        Returns:
            eqs: differential equations of compartments
        """
        self._check_square_matrix(self.distribution_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_matrix)

        t = sym.symbols('t', negative = False, real=True, finite = True) #time
        r = sym.symbols('RATE', positive = True, real=True, finite = True) #infusion rate
        
        funcs = [func(t) for func in self._get_functions()]  # type: ignore
        funcs_ = [func.diff(t) for func in funcs]

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i), positive = True, real=True, finite = True) if self.distribution_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j), positive = True, real=True, finite = True)
        
        comps_matrix = sym.Matrix([funcs[i] for i in range(comps_num)])
        if is_infusion :
            infusion = sym.Matrix([r if i == self.depot_compartment_num else 0. for i in range(comps_num) ])
        else :
            infusion = sym.Matrix([0. for i in range(comps_num)])
        
        dcdt_eqs = (distribution_rate.T  \
                    - sym.diag(*(distribution_rate * sym.ones(comps_num, 1))) \
                    - elimination_rate) * comps_matrix + infusion
        dcdt_eqs = dcdt_eqs[:]

        dcdt_eqs = [sym.Eq(func_, dcdt_eq) for func_, dcdt_eq in zip(funcs_, dcdt_eqs)]

        eqs = self._get_eqs_from_db(dcdt_eqs, initial_states)

        if len(dcdt_eqs) ==1 :
                dcdt_eqs = dcdt_eqs[0]
                funcs = funcs[0]
        
        if eqs is None :
            try :
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                        None, 
                        functools.partial(sym.solvers.ode.dsolve, dcdt_eqs, funcs, hint = '1st_linear', ics = initial_states))
                future = asyncio.wait_for(future, timeout, loop=loop)
                eqs = loop.run_until_complete(future)
            except asyncio.TimeoutError :
                raise TimeoutError()

            if isinstance(eqs, sym.Eq) :
                eqs = [eqs]
            eqs = [eq.rhs for eq in eqs]
            self._put_eqs_to_db(dcdt_eqs, initial_states, eqs)
        return eqs
    
    def _get_eqs_from_db(self, dcdt_eqs, ics) :
        package_dir, _ = os.path.split(__file__)
        db = shelve.open(package_dir + '\\' + self.DB_FILE_NAME)
        key = json.dumps({'dcdt_eqs': str(dcdt_eqs), 'ics': str(ics)}, sort_keys=True)
        try :
            eqs_serialized = db[key]
            return pickle.loads(eqs_serialized)  # type: ignore
        except KeyError :
            return None
        finally :
            db.close()
    
    def _put_eqs_to_db(self, dcdt_eqs, ics, eqs) :
        package_dir, _ = os.path.split(__file__)
        db = shelve.open(package_dir + '\\' + self.DB_FILE_NAME)
        try :
            key = json.dumps({'dcdt_eqs': str(dcdt_eqs), 'ics': str(ics)}, sort_keys=True)
            db[key] = pickle.dumps(eqs)
        finally :
            db.close()

    def forward(self, t, **variables):
        if self.is_infusion :
            infusion_end_time = variables['AMT'] / variables['RATE']
            
            infusion_t = tc.masked_select(t, t <= infusion_end_time)
            # elimination_t = tc.masked_select(t, t > infusion_end_time)
            variables['t'] = t
            # variables['t'] = infusion_t
            infusion_amt = self.infusion_model(**variables).t()

            # variables['t'] = elimination_t
            amt = self.model(**variables).t()

            

            amts = tc.concat([infusion_amt[:,:infusion_t.size()[0]], amt[:,infusion_t.size()[0]:]], dim = -1)
        elif self.is_delay_time and self.has_depot :
            delay_time = variables['delay_time']
            variables['t'] = (t - delay_time).clamp(min=0)
            amts = self.model(**variables).t()
        else :
            variables['t'] = t
            amts = self.model(**variables).t()
        return amts


class PreparedCompartmentModel(CompartmentModel) :
    
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        try :
            self.module = DB[self.model_config]()
        except KeyError :
            raise KeyError('Model not found in DB')
    def forward(self, **variables):
        return self.module(**variables)
    
class TwoCompartmentInfusion(nn.Module) :
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, t, k_00, k_01, k_10, RATE, AMT, **kwargs) -> tc.Tensor:
        beta = 1/2 * (k_01 + k_10 + k_00 - tc.sqrt((k_01 + k_10 + k_00)**2 - 4*k_10*k_00)) 
        alpha = k_10 * k_00 / beta
        a = (alpha - k_10) / (alpha - beta)
        b = (beta - k_10) / (beta - alpha)
        infusion_time = AMT/RATE

        infusion_t = tc.masked_select(t, t <= infusion_time)
        # elimination_t = tc.masked_select(t, t > infusion_time)


        amt_infusion = AMT/infusion_time * (a/alpha * (1 - tc.exp(-alpha*(t))) + b/beta * (1 - tc.exp(-beta*(t))))
        amt_elimination = AMT/infusion_time * (a/alpha * (1 - tc.exp(-alpha*(infusion_time)))*tc.exp(-alpha*(t-infusion_time)) 
                + b/beta * (1 - tc.exp(-beta*(infusion_time))))*tc.exp(-beta*(t-infusion_time))
        


        return tc.concat([amt_infusion[:infusion_t.size()[0]], amt_elimination[infusion_t.size()[0]:]], dim = -1).unsqueeze(0)

__init__()

