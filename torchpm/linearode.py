from turtle import forward
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

    def _check_square_matrix(m, error_massage) :
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
        LinearODE._check_square_matrix(self.distribution_bool_matrix, 'distribution_bool_matrix must be square matrix')

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
'''
class Comp1InfusionModelFunctionOld(tc.autograd.Function) :
    distribution_bool_matrix = [[True]]
    t_sym, dose_sym, r_sym = sym.symbols('t d r')
    infusion = LinearODE.solve(*LinearODE.get_eqs_and_ics(distribution_bool_matrix, True))
    
    
    eqs, ics = LinearODE.get_eqs_and_ics(distribution_bool_matrix, False)
    ics[sym.symbols('c_0', cls=sym.Function)(0)] = infusion[0].rhs.subs({t_sym: dose_sym/r_sym})
    function = LinearODE.solve(eqs, ics)
    function = [sym.Eq(eq.lhs, eq.rhs.subs({'t': 't - d/r'})) for eq in function]
    
    
    diff_functions = LinearODE.diff_functions(distribution_bool_matrix=distribution_bool_matrix, function=function, eqs=eqs)
    function_numpy = LinearODE.lambdify(distribution_bool_matrix, function, **diff_functions)
    
    @staticmethod
    def forward(ctx, t, k00, dose, r):

        t = t.detach()
        k00 = k00.detach()
        dose = dose.detach()
        r = r.detach()

        # t = t - dose/r

        output = tc.stack([tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), r.numpy())) \
                           for fn in Comp1InfusionModelFunction.function_numpy['function']])
                           
        ctx.save_for_backward(t, k00, dose, r, output)

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        t, k00, dose, r, output = ctx.saved_tensors

        # t=t-dose/r

        grad_t = grad_k00 = None
        if ctx.needs_input_grad[0]:
            grad_t = tc.stack([fn(t.numpy(), k00, dose.numpy(), r.numpy(), output.numpy().sqeeze(axis=0)) \
                               for fn in Comp1InfusionModelFunction.function_numpy['diff_function_t']])*grad_output

        if ctx.needs_input_grad[1]:

            grad_k00 =[]
            for fn in Comp1InfusionModelFunction.function_numpy['diff_function_distribution_rate'][0][0] :
                if fn is None :
                    grad_k00.append(tc.zeros(output.size()[-1]))
                else :
                    grad_k00.append(tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), r.numpy())))
            
            grad_k00 = tc.stack(grad_k00) * grad_output

        return grad_t, grad_k00, None, None
'''

'''
class Comp1BolusModelFunction(tc.autograd.Function) :
    distribution_bool_matrix = [[True]]
        
    eqs, ics = LinearODE.get_eqs_and_ics(distribution_bool_matrix, False)

    function = LinearODE.solve(eqs, ics)
    diff_function = LinearODE.diff_functions(distribution_bool_matrix, function, eqs)
    function_numpy = LinearODE.lambdify(distribution_bool_matrix, function, **diff_function)
    
    @staticmethod
    def forward(ctx, t, k00, dose, r):

        t = t.detach()
        k00 = k00.detach()
        dose = dose.detach()
        # r = r.detach()

        output = tc.stack([tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), None)) \
                           for fn in Comp1BolusModelFunction.function_numpy['function']])
                           
        ctx.save_for_backward(t, k00, dose, r, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        t, k00, dose, r, output = ctx.saved_tensors

        grad_t = grad_k00 = None
        if ctx.needs_input_grad[0]:
            grad_t = tc.stack([fn(t.numpy(), k00, dose.numpy(), None, output.numpy().sqeeze(axis=0)) \
                               for fn in Comp1BolusModelFunction.function_numpy['diff_function_t']])*grad_output

        if ctx.needs_input_grad[1]:

            grad_k00 =[]
            for fn in Comp1BolusModelFunction.function_numpy['diff_function_distribution_rate'][0][0] :
                if fn is None :
                    grad_k00.append(tc.zeros(output.size()[-1]))
                else :
                    grad_k00.append(tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), None)))
            
            grad_k00 = tc.stack(grad_k00) * grad_output

        return grad_t, grad_k00, None, None


class Comp1InfusionModelFunction(tc.autograd.Function) :
    distribution_bool_matrix = [[True]]
    t_sym, dose_sym, r_sym = sym.symbols('t d r')

    infusion = LinearODE.solve(*LinearODE.get_eqs_and_ics(distribution_bool_matrix, True))
    eqs, ics = LinearODE.get_eqs_and_ics(distribution_bool_matrix, False)
    ics[sym.symbols('c_0', cls=sym.Function)(0)] = infusion[0].rhs.subs({t_sym: dose_sym/r_sym})
    function = LinearODE.solve(eqs, ics)
    function = [sym.Eq(eq.lhs, eq.rhs.subs({'t': 't - d/r'})) for eq in function]
    diff_functions = LinearODE.diff_functions(distribution_bool_matrix=distribution_bool_matrix, function=function, eqs=eqs)
    function_numpy = LinearODE.lambdify(distribution_bool_matrix, function, **diff_functions)
    
    @staticmethod
    def forward(ctx, t, k00, dose, r):

        t = t.detach()
        k00 = k00.detach()
        dose = dose.detach()
        r = r.detach()

        # t = t - dose/r

        output = tc.stack([tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), r.numpy())) \
                           for fn in Comp1InfusionModelFunction.function_numpy['function']])
                           
        ctx.save_for_backward(t, k00, dose, r, output)

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        t, k00, dose, r, output = ctx.saved_tensors

        # t=t-dose/r

        grad_t = grad_k00 = None
        if ctx.needs_input_grad[0]:
            grad_t = tc.stack([fn(t.numpy(), k00, dose.numpy(), r.numpy(), output.numpy().sqeeze(axis=0)) \
                               for fn in Comp1InfusionModelFunction.function_numpy['diff_function_t']])*grad_output

        if ctx.needs_input_grad[1]:

            grad_k00 =[]
            for fn in Comp1InfusionModelFunction.function_numpy['diff_function_distribution_rate'][0][0] :
                if fn is None :
                    grad_k00.append(tc.zeros(output.size()[-1]))
                else :
                    grad_k00.append(tc.Tensor(fn(t.numpy(), k00.numpy(), dose.numpy(), r.numpy())))
            
            grad_k00 = tc.stack(grad_k00) * grad_output

        return grad_t, grad_k00, None, None
'''