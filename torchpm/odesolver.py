from copy import deepcopy
from dataclasses import dataclass
import enum
from functools import cache
import os
import pickle
from re import X
import shelve
from typing import ClassVar, List, Optional, Dict, Iterable, Tuple, Union
import typing
from attr import asdict
import torch as tc
import numpy as np
import sympy as sym
import sympytorch as spt
from torch import nn
import torch
import json

@enum.unique
class CompartmentDistributionMatrix(enum.Enum) :
    ONE_COMP_DIST : Tuple[Tuple[bool]] = ((True))  # type: ignore
    TWO_COMP_DIST : Tuple[Tuple[bool]] = ((True, True), (True, False))  # type: ignore
    THREE_COMP_DIST : Tuple[Tuple[bool]] = ((True, True, True), (True, False, False), (True, False, False))  # type: ignore
twoCompartmentInfusionKey = ModelConfig(
                CompartmentDistributionMatrix.TWO_COMP_DIST.value,
                is_infusion = True)

DB[json.dumps(asdict(twoCompartmentInfusionKey), sort_keys=True)] = TwoCompartmentInfusion

class CompartmentModelGenerator(nn.Module) :
    DB_FILE_NAME = "ode.db"

    def _make_distribution_matrix(self) -> None:
        if self.has_depot :
            self.distribution_matrix.append([False] * len(self.distribution_matrix))
            for row in self.distribution_matrix :
                row.append(False)
            self.distribution_matrix[-1][self.administrated_compartment_num] = True
        
            self.depot_compartment_num = len(self.distribution_matrix) - 1

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
            transit_start = self.depot_compartment_num + 1 
            for i in range(transit_start, transit_start + self.transit - 1):
                self.distribution_matrix[i][i+1] = True
            self.distribution_matrix[-1][self.administrated_compartment_num] = True
    
    def get_distribution_matrix(self) -> List[List[bool]]:
        if self.distribution_matrix == [] :
            self._make_distribution_matrix()
        return self.distribution_matrix
    

    def __init__(self, 
            distribution_bool_matrix: Union[Tuple[Tuple[bool]], bool], 
            has_depot : bool = False, 
            transit : int = 0, 
            observed_compartment_num = 0, 
            administrated_compartment_num = 0, 
            is_infusion : bool= False) -> None:
        super().__init__()

        self.distribution_matrix = []
        if isinstance(distribution_bool_matrix, bool) :
            self.distribution_matrix = [[distribution_bool_matrix]]
        elif isinstance(distribution_bool_matrix, Tuple) :
            for row in list(distribution_bool_matrix) :
                self.distribution_matrix.append(list(row))

        self.is_infusion = is_infusion
        self.observed_compartment_num = observed_compartment_num
        self.administrated_compartment_num = administrated_compartment_num
        self.depot_compartment_num = administrated_compartment_num
        self.has_depot = has_depot
        self.transit = transit

        self._make_distribution_matrix()

class NumericalCompartmentModelGenerator(CompartmentModelGenerator) :
    
    def forward(self, y, **k : tc.Tensor) :

        comps_num = len(self.distribution_matrix)

        elimination_rate = tc.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = k['k_' + str(i) + str(i)] if self.distribution_matrix[i][i] else 0

        distribution_rate = tc.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_matrix[i][j]:
                    continue
                distribution_rate[i,j] = k['k_'+str(i)+str(j)]
        
        dcdt_matrix = distribution_rate.t()  \
                    - tc.diag(distribution_rate @ tc.ones(comps_num, 1)) \
                    - elimination_rate

        return dcdt_matrix @ y


class SymbolicCompartmentModelGenerator(CompartmentModelGenerator) :
    
    def __init__(self, 
            distribution_matrix: Union[Tuple[Tuple[bool]], bool], 
            has_depot : bool = False, 
            transit : int = 0, 
            observed_compartment_num = 0, 
            administrated_compartment_num = 0, 
            is_infusion : bool= False) -> None:
        
        super().__init__(distribution_matrix, has_depot, transit, observed_compartment_num, administrated_compartment_num, is_infusion)

        if self.is_infusion :
            t_sym = sym.symbols('t', real=True, negative = False, finite = True)
            r_sym, dose_sym = sym.symbols('r, d', real=True, positive = True, finite = True)
            initial_states_infusion = self._get_initial_states(is_infusion=self.is_infusion)
            cs_infusion = self._solve_linode(initial_states_infusion, is_infusion=True)

            funcs = self._get_functions()
            initial_states = self._get_initial_states(is_infusion=self.is_infusion)
            for i, func in enumerate(funcs) :
                initial_states[func(0)] = cs_infusion[i].subs({t_sym: dose_sym/r_sym})
            cs = self._solve_linode(initial_states, is_infusion=False)
            
            for i in range(len(cs)) :
                cs[i] = cs[i].subs({t_sym: t_sym - dose_sym/r_sym})

            self.infusion_model = spt.SymPyModule(expressions=cs_infusion)
        
        else:
            initial_states = self._get_initial_states(is_infusion=self.is_infusion) 
            cs = self._solve_linode(initial_states, is_infusion=self.is_infusion)
        
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
        d = sym.symbols('d', real = True, Positve = True, finite = True) #dose
        result = {}

        funcs = self._get_functions()  
        funcs = [func(0) for func in funcs]
        
        for i, func in enumerate(funcs) :
            if not is_infusion and i == self.depot_compartment_num  :
                result[func] = d
            else :
                result[func] = 0

        return result

    def _solve_linode(self, initial_states, is_infusion : bool = False) -> List[sym.Eq]:
        """
        Returns:
            eqs: differential equations of compartments
        """
        self._check_square_matrix(self.distribution_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_matrix)

        t = sym.symbols('t', negative = False, real=True, finite = True) #time
        r = sym.symbols('r', positive = True, real=True, finite = True) #infusion rate
        
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
            eqs = sym.solvers.ode.dsolve(dcdt_eqs, funcs, hint='1st_linear', ics = initial_states)
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
            return pickle.loads(eqs_serialized)
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
            infusion_end_time = variables['d'] / variables['r']

            
            infusion_t = tc.masked_select(t, t <= infusion_end_time)
            elimination_t = tc.masked_select(t, t > infusion_end_time)

            variables['t'] = infusion_t
            infusion_amt = self.infusion_model(**variables).t()

            variables['t'] = elimination_t
            amt = self.model(**variables).t()

            variables['t'] = t
            return tc.concat([infusion_amt, amt], dim = -1)
        else :
            variables['t'] = t
            return self.model(**variables).t()

@dataclass
class ModelConfig:
    distribution_matrix: Union[Tuple[Tuple[bool]], bool] 
    has_depot: bool = False
    transit: int = 0
    observed_compartment_num : int = 0
    administrated_compartment_num : int = 0
    is_infusion: bool = False

class PreparedCompartmentModel :
    DB : Dict[str, typing.Type[nn.Module]] = {}
    
    def __init__(self, 
            distribution_matrix: Union[Tuple[Tuple[bool]], bool], 
            has_depot: bool = False, 
            transit: int = 0, 
            observed_compartment_num=0, 
            administrated_compartment_num=0, 
            is_infusion: bool = False) -> None:
        self.distribution_matrix = distribution_matrix
        self.has_depot = has_depot
        self.transit = transit
        self.observed_compartment_num = observed_compartment_num
        self.administrated_compartment_num = administrated_compartment_num
        self.is_infusion = is_infusion

        self.model_config = ModelConfig(
                distribution_matrix, 
                has_depot, transit, 
                observed_compartment_num, 
                administrated_compartment_num, 
                is_infusion)

        twoCompartmentInfusionKey = ModelConfig(
                CompartmentDistributionMatrix.TWO_COMP_DIST.value,
                has_depot, 
                transit, 
                observed_compartment_num, 
                administrated_compartment_num, 
                is_infusion)
        self.DB[json.dumps(asdict(twoCompartmentInfusionKey), sort_keys=True)] = TwoCompartmentInfusion


    def get_eqs_from_db(self) :
        db = self.DB
        model_cofig = ModelConfig(self.distribution_matrix, self.has_depot, self.transit, self.observed_compartment_num, self.administrated_compartment_num, self.is_infusion)
        key = json.dumps(asdict(model_cofig), sort_keys=True)
        try :
            return db[key]
        except KeyError :
            return None
    
class TwoCompartmentInfusion(nn.Module) :
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, t, k_00, k_01, k_10, r, d) :
        beta = 1/2 * (k_01 + k_10 + k_00 - tc.sqrt((k_01 + k_10 + k_00)**2 - 4*k_10*k_00)) 
        alpha = k_10 * k_00 / beta
        a = (alpha - k_10) / (alpha - beta)
        b = (beta - k_10) / (beta - alpha)
        infusion_time = d/r

        infusion_t = tc.masked_select(t, t <= infusion_time)
        elimination_t = tc.masked_select(t, t > infusion_time)


        amt_infusion = d/infusion_time * (a/alpha * (1 - tc.exp(-alpha*(infusion_t))) + b/beta * (1 - tc.exp(-beta*(infusion_t))))
        amt_elimination = d/infusion_time * (a/alpha * (1 - tc.exp(-alpha*(infusion_time)))*tc.exp(-alpha*(elimination_t-infusion_time)) 
                + b/beta * (1 - tc.exp(-beta*(infusion_time))))*tc.exp(-beta*(elimination_t-infusion_time))

        return tc.concat([amt_infusion, amt_elimination])