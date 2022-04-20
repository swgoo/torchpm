from copy import deepcopy
from typing import ClassVar, List, Optional, Dict, Iterable, Union
import torch as tc
import numpy as np
import sympy as sym
import sympytorch as spt
from torch import nn

class LinearODE :
    def __init__(self, distribution_bool_matrix, is_infusion) -> None:
        """
        compartment phamacokinetics model function generator
        compartment 0 is applied drug.
        Args :
            distribution_bool_matrix : Whether there is a distribution in the compartment to another.
                                        diagonal part is elimination ratio from specific compartment.
                                        it must be square matrix.
            is_infusion : if it's True, compartment 0 is applied drug infusion.
        """
        self.distribution_bool_matrix = distribution_bool_matrix
        self.is_infusion = is_infusion

        self.dCdts = self._get_dCdts()
        self.initial_states = self._get_initial_states()
        self.solve()

    def _check_square_matrix(self, m : List[List[float]], error_massage) :
        length = len(m)
        for row in m :
            if len(row) != length :
                raise RuntimeError(error_massage)

    def _get_initial_states(self):
        """
        Returns:
            ics: initial states of compartments
        """
        comps_num = len(self.distribution_bool_matrix)
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function) for i in range(comps_num)]
        d = sym.symbols('d') #dose

        ics = {comp(0): 0 for comp in comps}
        ics[comps[0](0)] = d if not self.is_infusion else 0
        return ics

    def _get_dCdts(self) :
        """
        Returns:
            eqs: differential equations of compartments
        """
        self._check_square_matrix(self.distribution_bool_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_bool_matrix)

        t = sym.symbols('t', positive = True, real=True) #time
        r = sym.symbols('r', positive = True, real=True) #infusion rate
        
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function) for i in range(comps_num)]

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i), positive = True, real=True) if self.distribution_bool_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_bool_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j), positive = True, real=True)
        
        comps_matrix = sym.Matrix([comps[i](t) for i in range(comps_num)])
        dcdt_eqs = distribution_rate.T * comps_matrix \
                    - sym.diag(*(distribution_rate * sym.ones(comps_num, 1))) * comps_matrix \
                    - elimination_rate * comps_matrix
        dcdt_eqs[0] = dcdt_eqs[0] + r if self.is_infusion else dcdt_eqs[0]

        eqs = [sym.Eq(comps[i](t).diff(t), dcdt_eqs[i]) for i in range(comps_num)]
        return eqs
    
    def solve(self) :
        """
        solve differential equations
        Args:
            eqs: differential equations of compartments
            ics: initial states of compartments
        Return :
            compartment functions by time
        """
        function = sym.solvers.ode.systems.dsolve_system(self.dCdts, ics=self.initial_states, doit=True)
        self.cs = function[0]

class CompartmentModelGenerator(nn.Module) :
    def __init__(self, distribution_bool_matrix: List[List[bool]], has_depot : bool = False, transit : int = 0, administrated_compartment_num = 0, is_infusion : bool= False) -> None:
        super().__init__()

        self.distribution_bool_matrix = deepcopy(distribution_bool_matrix)
        self.is_infusion = is_infusion
        self.administrated_compartment_num = administrated_compartment_num
        self.depot_compartment_num = administrated_compartment_num
        if has_depot :
            self.distribution_bool_matrix.append([False] * len(self.distribution_bool_matrix))
            for row in self.distribution_bool_matrix :
                row.append(False)
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
        
            self.depot_compartment_num = len(distribution_bool_matrix)

        if transit > 0 :
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = False
            length = len(self.distribution_bool_matrix)
            for row in range(transit) :
                self.distribution_bool_matrix.append([False] * length)
            
            for row in self.distribution_bool_matrix :
                for i in range(transit):
                    row.append(False)
            
            #depot to transit 0
            self.distribution_bool_matrix[len(distribution_bool_matrix)][len(distribution_bool_matrix)+1] = True
            transit_start = len(distribution_bool_matrix) + 1 
            for i in range(transit_start, transit_start + transit - 1):
                self.distribution_bool_matrix[i][i+1] = True
                #TODO
                # self.distribution_bool_matrix[i+1][i] = True
            self.distribution_bool_matrix[-1][self.administrated_compartment_num] = True
            
        dCdts = self._get_dCdts(is_infusion=False)
        initial_states = self._get_initial_states(is_infusion=False)
        cs = self._solve(dCdts, initial_states)

        if is_infusion :
        
            t_sym, dose_sym, r_sym = sym.symbols('t d r', positive = True, real=True)
            dCdts_infusion = self._get_dCdts(is_infusion=True)
            initial_states_infusion = self._get_initial_states(is_infusion=True)
            infusion_cs = self._solve(dCdts_infusion, initial_states_infusion)

            initial_states[sym.symbols('c_' + str(self.depot_compartment_num), cls=sym.Function)(0)] = infusion_cs[self.depot_compartment_num].rhs.subs({t_sym: dose_sym/r_sym})
            cs = self._solve(dCdts, initial_states)
            
            for i in range(len(cs)) :
                cs[i] = sym.Eq(cs[i].lhs, cs[i].rhs.subs({t_sym: t_sym - dose_sym/r_sym}))
            

            infusion_comps = []
            for comp in infusion_cs :
                infusion_comps.append(comp.rhs)
            self.infusion_model = spt.SymPyModule(expressions=infusion_comps)

            
            

        comps = []
        for comp in cs :
            comps.append(comp.rhs)
        self.model = spt.SymPyModule(expressions=comps)
    
    

    def _check_square_matrix(self, m : List[List[bool]], error_massage) :
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
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function) for i in range(comps_num)]
        d = sym.symbols('d') #dose

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

        t = sym.symbols('t', positive = True, real=True) #time
        r = sym.symbols('r', positive = True, real=True) #infusion rate
        
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function) for i in range(comps_num)]

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i), positive = True, real=True) if self.distribution_bool_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_bool_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j), positive = True, real=True)
        
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
        function = sym.solvers.ode.systems.dsolve_system(dCdts, ics=ics, doit=True)
        return function[0]

    def forward(self, **variables):
        
        if self.is_infusion :
            infusion_end_time = variables['d'] / variables['r']

            infusion_t = tc.masked_select(variables['t'], variables['t'] <= infusion_end_time)
            elimination_t = tc.masked_select(variables['t'], variables['t'] > infusion_end_time)

            variables_infusion = deepcopy(variables)
            variables_infusion['t'] = infusion_t
            infusion_amt = self.infusion_model(**variables_infusion).t()

            variables_elimination = deepcopy(variables)
            variables_elimination['t'] = elimination_t
            amt = self.model(**variables_elimination).t()

            return tc.concat([infusion_amt, amt], dim = -1)
        else :
            return self.model(**variables).t()
            

#TODO sympy 수식 저장해서 불러오기
class Comp1GutModelFunction(nn.Module):
    distribution_bool_matrix = [[False, True],
                                [False, True]]
    function = LinearODE(distribution_bool_matrix, False)
    
    def __init__(self) -> None:
        super().__init__()

        comps = []

        for comp in Comp1GutModelFunction.function.cs :
            comps.append(comp.rhs)

        self.model = spt.SymPyModule(expressions=comps)
    
    def forward(self, t, k01, k11, dose):
        return self.model(t=t, k_01=k01, k_11= k11, d=dose).t()

#TODO sympy 수식 저장해서 불러오기
class Comp1InfusionModelFunction(nn.Module):
    distribution_bool_matrix = [[True]]
    t_sym, dose_sym, r_sym = sym.symbols('t d r', positive = True, real=True)

    infusion_function = LinearODE(distribution_bool_matrix, True)
    
    infusion = infusion_function.cs

    function = LinearODE(distribution_bool_matrix, False)
    function.initial_states[sym.symbols('c_0', cls=sym.Function)(0)] = infusion[0].rhs.subs({t_sym: dose_sym/r_sym})

    function.solve()
    for i in range(len(function.cs)) :
        function.cs[i] = sym.Eq(function.cs[i].lhs, function.cs[i].rhs.subs({t_sym: t_sym - dose_sym/r_sym}))

    def __init__(self) -> None:
        super().__init__()


        infusion_comps = []
        comps = []

        for comp in Comp1InfusionModelFunction.infusion_function.cs :
            infusion_comps.append(comp.rhs)


        for comp in Comp1InfusionModelFunction.function.cs :
            comps.append(comp.rhs)
        self.infusion_model = spt.SymPyModule(expressions=infusion_comps)
        self.model = spt.SymPyModule(expressions=comps)
    
    def forward(self, t, k00, dose, rate):

        infusion_end_time = dose/rate

        infusion_t = tc.masked_select(t, t <= infusion_end_time)
        elimination_t = tc.masked_select(t, t > infusion_end_time)

        infusion_amt = self.infusion_model(t=infusion_t, k_00=k00, d=dose, r=rate).t()

        amt = self.model(t=elimination_t, k_00=k00, d=dose, r=rate).t()


        return tc.concat([infusion_amt, amt], dim = -1)