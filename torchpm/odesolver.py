from copy import deepcopy
from functools import cache
import pickle
from re import X
from typing import ClassVar, List, Optional, Dict, Iterable, Tuple, Union
import torch as tc
import numpy as np
import sympy as sym
import sympytorch as spt
from torch import nn
import torch
import shelve

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class CompartmentModelGenerator(nn.Module) :
    def __init__(self, distribution_bool_matrix: Union[Tuple[Tuple[bool]], bool], has_depot : bool = False, transit : int = 0, observed_compartment_num = 0, administrated_compartment_num = 0, is_infusion : bool= False) -> None:
        super().__init__()

        

        self.distribution_bool_matrix = []
        if isinstance(distribution_bool_matrix, bool) :
            self.distribution_bool_matrix = [[distribution_bool_matrix]]
        elif isinstance(distribution_bool_matrix, Tuple) :
            for row in list(distribution_bool_matrix) :
                self.distribution_bool_matrix.append(list(row))

        self.is_infusion = is_infusion
        self.observed_compartment_num = observed_compartment_num
        self.administrated_compartment_num = administrated_compartment_num
        self.depot_compartment_num = administrated_compartment_num
        self.has_depot = has_depot
        self.transit = transit
        if has_depot :
            self.distribution_bool_matrix.append([False] * len(self.distribution_bool_matrix))
            for row in self.distribution_bool_matrix :
                row.append(False)
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
        
            self.depot_compartment_num = len(self.distribution_bool_matrix) - 1

        if has_depot and transit > 0 :
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = False
            length = len(self.distribution_bool_matrix)
            for row in range(transit) :
                self.distribution_bool_matrix.append([False] * length)
            
            for row in self.distribution_bool_matrix :
                for i in range(transit):
                    row.append(False)
            
            #depot to transit 0
            self.distribution_bool_matrix[self.depot_compartment_num][self.depot_compartment_num+1] = True
            transit_start = self.depot_compartment_num + 1 
            for i in range(transit_start, transit_start + transit - 1):
                self.distribution_bool_matrix[i][i+1] = True
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
            
        dCdts = self._get_dCdts(is_infusion=False)
        initial_states = self._get_initial_states(is_infusion=False)
        cs = self._solve(dCdts, initial_states)

        if is_infusion :
        
            t_sym = sym.symbols('t', real=True, negative = False, finite = True)
            r_sym, dose_sym = sym.symbols('r, d', real=True, positive = True, finite = True)
            dCdts_infusion = self._get_dCdts(is_infusion=True)
            initial_states_infusion = self._get_initial_states(is_infusion=True)
            cs_infusion = self._solve(dCdts_infusion, initial_states_infusion)

            initial_states[sym.symbols('c_' + str(self.depot_compartment_num), cls=sym.Function, real = True, negative=False, finite = True)(0)] = cs_infusion[self.depot_compartment_num].rhs.subs({t_sym: dose_sym/r_sym})  # type: ignore
            cs = self._solve(dCdts, initial_states)
            
            for i in range(len(cs)) :
                cs[i] = sym.Eq(cs[i].lhs, cs[i].rhs.subs({t_sym: t_sym - dose_sym/r_sym}))
            

            comps_infusion = []
            for comp in cs_infusion :
                comps_infusion.append(comp.rhs)
            self.infusion_model = spt.SymPyModule(expressions=comps_infusion)
        
        comps = []
        for comp in cs :
            comps.append(comp.rhs)
        self.model = spt.SymPyModule(expressions=comps)
    
    

    def _check_square_matrix(self, m : list[list[bool]], error_massage) :
        
        length = len(m)
        for row in m :
            if len(row) != length :
                raise RuntimeError(error_massage)

    def _get_initial_states(self, is_infusion : bool = False) :
        """
        Returns:
            ics: initial states of compartments
        """
        comps_num = len(self.distribution_bool_matrix)
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, real = True, negative = False, finite = True) for i in range(comps_num)]  # type: ignore
        d = sym.symbols('d', real = True, Positve = True, finite = True) #dose

        ics = {comp(0): 0 for comp in comps}
        ics[comps[self.depot_compartment_num](0)] = 0 if is_infusion else d
        return ics

    def _get_dCdts(self, is_infusion : bool = False) -> List[sym.Eq]:
        """
        Returns:
            eqs: differential equations of compartments
        """
        self._check_square_matrix(self.distribution_bool_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_bool_matrix)

        t = sym.symbols('t', negative = False, real=True, finite = True) #time
        r = sym.symbols('r', positive = True, real=True, finite = True) #infusion rate
        
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, negative = False, real = True, finite = True) for i in range(comps_num)]  # type: ignore

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i), positive = True, real=True, finite = True) if self.distribution_bool_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_bool_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j), positive = True, real=True, finite = True)
        
        comps_matrix = sym.Matrix([comps[i](t) for i in range(comps_num)])
        dcdt_eqs = distribution_rate.T * comps_matrix \
                    - sym.diag(*(distribution_rate * sym.ones(comps_num, 1))) * comps_matrix \
                    - elimination_rate * comps_matrix
        if is_infusion :
            dcdt_eqs[self.depot_compartment_num] = dcdt_eqs[self.depot_compartment_num] + r

        eqs = [sym.Eq(comps[i](t).diff(t), dcdt_eqs[i]) for i in range(comps_num)]
        return eqs
    
    def _solve(self, dCdts, ics) :
        """
        solve differential equations
        Args:
            eqs: differential equations of compartments
            ics: initial states of compartments
        Return :
            compartment functions by time

            
        """
        comps_num = len(dCdts)
        db = shelve.open('./ode.db')
        
        parameters = dict(
                distribution_bool_matrix=self.distribution_bool_matrix, 
                has_depot=self.has_depot, 
                transit=self.transit, 
                observed_compartment_num=self.observed_compartment_num, 
                administrated_compartment_num=self.administrated_compartment_num, 
                is_infusion=self.is_infusion)
        parameters = HashableDict(parameters)

        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, negative = False, real = True)(sym.symbols('t', negative = False, real = True)) for i in range(comps_num)]  # type: ignore

        try :
            function_serialized = db[str(parameters)]
            function = []
            for i, f in enumerate(function_serialized) :
                rhs = pickle.loads(f)
                lhs = comps[i]
                function.append(sym.Eq(lhs, rhs, simpify=False))
            return function
        except KeyError :
            function = sym.solvers.ode.systems.dsolve_system(dCdts, ics=ics, doit=True, simplify=True)[0]
            function_serialized = []
            for f in function :
                function_serialized.append(pickle.dumps(f.rhs))
            db[str(parameters)] = function_serialized
            return function
        finally :

        # function = sym.solvers.ode.systems.dsolve_system(dCdts, ics=ics, doit=True, simplify=True)[0]
        # function_serialized = []
        # for f in function :
        #     function_serialized.append(pickle.dumps(f.rhs))
        # db[str(parameters)] = function_serialized
            db.close()
        # return function

        

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



class CompartmentModelGeneratorNew(nn.Module) :
    def __init__(self, distribution_bool_matrix: Union[Tuple[Tuple[bool]], bool], has_depot : bool = False, transit : int = 0, observed_compartment_num = 0, administrated_compartment_num = 0, is_infusion : bool= False) -> None:
        super().__init__()

        

        self.distribution_bool_matrix = []
        if isinstance(distribution_bool_matrix, bool) :
            self.distribution_bool_matrix = [[distribution_bool_matrix]]
        elif isinstance(distribution_bool_matrix, Tuple) :
            for row in list(distribution_bool_matrix) :
                self.distribution_bool_matrix.append(list(row))

        self.is_infusion = is_infusion
        self.observed_compartment_num = observed_compartment_num
        self.administrated_compartment_num = administrated_compartment_num
        self.depot_compartment_num = administrated_compartment_num
        self.has_depot = has_depot
        self.transit = transit
        if has_depot :
            self.distribution_bool_matrix.append([False] * len(self.distribution_bool_matrix))
            for row in self.distribution_bool_matrix :
                row.append(False)
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
        
            self.depot_compartment_num = len(self.distribution_bool_matrix) - 1

        if has_depot and transit > 0 :
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = False
            length = len(self.distribution_bool_matrix)
            for row in range(transit) :
                self.distribution_bool_matrix.append([False] * length)
            
            for row in self.distribution_bool_matrix :
                for i in range(transit):
                    row.append(False)
            
            #depot to transit 0
            self.distribution_bool_matrix[self.depot_compartment_num][self.depot_compartment_num+1] = True
            transit_start = self.depot_compartment_num + 1 
            for i in range(transit_start, transit_start + transit - 1):
                self.distribution_bool_matrix[i][i+1] = True
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
            
        dCdts = self._get_dCdts(is_infusion=False)
        initial_states = self._get_initial_states(is_infusion=False)
        cs = self._solve(dCdts, initial_states)

        if is_infusion :
        
            t_sym = sym.symbols('t', real=True, negative = False, finite = True)
            r_sym, dose_sym = sym.symbols('r, d', real=True, positive = True, finite = True)
            dCdts_infusion = self._get_dCdts(is_infusion=True)
            initial_states_infusion = self._get_initial_states(is_infusion=True)
            cs_infusion = self._solve(dCdts_infusion, initial_states_infusion)

            initial_states[sym.symbols('c_' + str(self.depot_compartment_num), cls=sym.Function, real = True, negative=False, finite = True)(0)] = cs_infusion[self.depot_compartment_num].rhs.subs({t_sym: dose_sym/r_sym})  # type: ignore
            cs = self._solve(dCdts, initial_states)
            
            for i in range(len(cs)) :
                cs[i] = sym.Eq(cs[i].lhs, cs[i].rhs.subs({t_sym: t_sym - dose_sym/r_sym}))
            

            comps_infusion = []
            for comp in cs_infusion :
                comps_infusion.append(comp.rhs)
            self.infusion_model = spt.SymPyModule(expressions=comps_infusion)
        
        comps = []
        for comp in cs :
            comps.append(comp.rhs)
        self.model = spt.SymPyModule(expressions=comps)
    
    

    def _check_square_matrix(self, m : list[list[bool]], error_massage) :
        
        length = len(m)
        for row in m :
            if len(row) != length :
                raise RuntimeError(error_massage)

    def _get_initial_states(self, is_infusion : bool = False) :
        """
        Returns:
            ics: initial states of compartments
        """
        comps_num = len(self.distribution_bool_matrix)
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, real = True, negative = False, finite = True) for i in range(comps_num)]  # type: ignore
        d = sym.symbols('d', real = True, Positve = True, finite = True) #dose

        ics = {comp(0): 0 for comp in comps}
        ics[comps[self.depot_compartment_num](0)] = 0 if is_infusion else d
        return ics

    def _get_dCdts(self, is_infusion : bool = False) -> List[sym.Eq]:
        """
        Returns:
            eqs: differential equations of compartments
        """
        self._check_square_matrix(self.distribution_bool_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_bool_matrix)

        t = sym.symbols('t', negative = False, real=True, finite = True) #time
        r = sym.symbols('r', positive = True, real=True, finite = True) #infusion rate
        
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, negative = False, real = True, finite = True) for i in range(comps_num)]  # type: ignore

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i), positive = True, real=True, finite = True) if self.distribution_bool_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_bool_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j), positive = True, real=True, finite = True)
        
        comps_matrix = sym.Matrix([comps[i](t) for i in range(comps_num)])
        dcdt_eqs = distribution_rate.T * comps_matrix \
                    - sym.diag(*(distribution_rate * sym.ones(comps_num, 1))) * comps_matrix \
                    - elimination_rate * comps_matrix
        if is_infusion :
            dcdt_eqs[self.depot_compartment_num] = dcdt_eqs[self.depot_compartment_num] + r

        eqs = [sym.Eq(comps[i](t).diff(t), dcdt_eqs[i]) for i in range(comps_num)]
        return eqs
    
    def _solve(self, dCdts, ics) :
        """
        solve differential equations
        Args:
            eqs: differential equations of compartments
            ics: initial states of compartments
        Return :
            compartment functions by time

            
        """
        comps_num = len(dCdts)
        db = shelve.open('./ode.db')
        
        parameters = dict(
                distribution_bool_matrix=self.distribution_bool_matrix, 
                has_depot=self.has_depot, 
                transit=self.transit, 
                observed_compartment_num=self.observed_compartment_num, 
                administrated_compartment_num=self.administrated_compartment_num, 
                is_infusion=self.is_infusion)
        parameters = HashableDict(parameters)

        comps = [sym.symbols("c_"+ str(i), cls=sym.Function, negative = False, real = True)(sym.symbols('t', negative = False, real = True)) for i in range(comps_num)]  # type: ignore

        try :
            function_serialized = db[str(parameters)]
            function = []
            for i, f in enumerate(function_serialized) :
                rhs = pickle.loads(f)
                lhs = comps[i]
                function.append(sym.Eq(lhs, rhs, simpify=False))
            return function
        except KeyError :
            function = sym.solvers.ode.systems.dsolve_system(dCdts, ics=ics, doit=True, simplify=True)[0]
            function_serialized = []
            for f in function :
                function_serialized.append(pickle.dumps(f.rhs))
            db[str(parameters)] = function_serialized
            return function
        finally :

        # function = sym.solvers.ode.systems.dsolve_system(dCdts, ics=ics, doit=True, simplify=True)[0]
        # function_serialized = []
        # for f in function :
        #     function_serialized.append(pickle.dumps(f.rhs))
        # db[str(parameters)] = function_serialized
            db.close()
        # return function

        

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