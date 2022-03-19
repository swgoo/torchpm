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
        self.cs = LinearODE._solve(self.dCdts, self.initial_states)

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

        t = sym.symbols('t') #time
        r = sym.symbols('r') #infusion rate
        
        comps = [sym.symbols("c_"+ str(i), cls=sym.Function) for i in range(comps_num)]

        elimination_rate = sym.eye(comps_num)
        for i in range(comps_num) :
            elimination_rate[i,i] = sym.symbols('k_' + str(i) + str(i)) if self.distribution_bool_matrix[i][i] else 0

        distribution_rate = sym.zeros(comps_num, comps_num)
        for i in range(comps_num) :
            for j in range(comps_num) :
                if i == j or not self.distribution_bool_matrix[i][j]:
                    continue
                distribution_rate[i,j] = sym.symbols('k_'+str(i)+str(j))
        
        comps_matrix = sym.Matrix([comps[i](t) for i in range(comps_num)])
        dcdt_eqs = distribution_rate.T * comps_matrix \
                    - sym.diag(*(distribution_rate * sym.ones(comps_num, 1))) * comps_matrix \
                    - elimination_rate * comps_matrix
        dcdt_eqs[0] = dcdt_eqs[0] + r if self.is_infusion else dcdt_eqs[0]

        eqs = [sym.Eq(comps[i](t).diff(t), dcdt_eqs[i]) for i in range(comps_num)]
        return eqs
    
    def _solve(eqs, ics) :
        """
        solve differential equations
        Args:
            eqs: differential equations of compartments
            ics: initial states of compartments
        Return :
            compartment functions by time
        """
        function = sym.solvers.ode.systems.dsolve_system(eqs, ics=ics, doit=True)
        return function[0]
    
    def diff_functions(self) :
        """
        Args :
            function: compartment function by time.
        Return:
            diff_function_t : (numpy version) differential function with respect to t
            diff_function_distribution_rate : (numpy version) matrix. differential functions with respect to distribution_rate
        """
        LinearODE._check_square_matrix(self.distribution_bool_matrix, 'distribution_bool_matrix must be square matrix')

        comps_num = len(self.distribution_bool_matrix)
        diff_argument_subs = {sym.symbols('c_'+str(i), cls=sym.Function)(sym.symbols('t')): sym.symbols('y_'+str(i)) for i in range(comps_num)}

        diff_function_t = [fn.rhs.subs(diff_argument_subs) for fn in self.dCdts]

        diff_function_distribution_rate = [[[None for k in range(comps_num) ] for j in range(comps_num)] for i in range(comps_num)]
        for i in range(comps_num) :
            for j in range(comps_num) :
                for k in range(comps_num) :
                    if self.distribution_bool_matrix[i][j] :
                        diff_function_distribution_rate[i][j][k] = self.cs[k].rhs.diff('k_'+str(i)+str(j))
        
        return {'diff_function_t': diff_function_t, 'diff_function_distribution_rate': diff_function_distribution_rate}
    
    '''
    def lambdify(self, dCdks):
        """
        Args :
            function: compartment function by time.
            diff_function_t : differential function with respect to t
            diff_function_distribution_rate : matrix. differential functions with respect to distribution_rate
        Return:
            diff_function_t : differential function with respect to t
            diff_function_distribution_rate : matrix. differential functions with respect to distribution_rate
        """
        comps_num = len(self.distribution_bool_matrix)
        distribution_rate_arguments = []
        for i in range(comps_num) :
            for j in range(comps_num) :
                if self.distribution_bool_matrix[i][j] :
                    distribution_rate_arguments.append('k_'+str(i)+str(j))
        distribution_rate_arguments_str = ' '.join(distribution_rate_arguments)
        distribution_rate_arguments_symbol = sym.symbols(distribution_rate_arguments_str)
        if type(distribution_rate_arguments_symbol) is not tuple :
            distribution_rate_arguments_symbol = (distribution_rate_arguments_symbol,)
                                                         
        function_numpy = [sym.lambdify([sym.symbols('t'),
                                             *distribution_rate_arguments_symbol,
                                             *sym.symbols('d r')], fn.rhs, modules='numpy') for fn in self.cs]
        ys = [sym.symbols('y_'+str(i)) for i in range(comps_num)]
        diff_function_t_numpy = [sym.lambdify([sym.symbols('t'), 
                                                    *distribution_rate_arguments_symbol,
                                                    *sym.symbols('d r'), *ys], fn, modules='numpy') for fn in self.dCdts]
        
        diff_function_distribution_rate_numpy = [[[None for k in range(comps_num)] for j in range(comps_num)] for i in range(comps_num)]
        for i in range(comps_num) :
            for j in range(comps_num) :
                for k in range(comps_num) :
                    fn = dCdks[i][j][k]
                    if fn is None :
                        continue
                    if fn != 0 :
                        diff_function_distribution_rate_numpy[i][j][k] = sym.lambdify([sym.symbols('t'),
                                                                                            *distribution_rate_arguments_symbol,
                                                                                            *sym.symbols('d r')], fn, modules='numpy')
        return {'cs': function_numpy,
                'dCdts': diff_function_t_numpy,
                'dCdks': diff_function_distribution_rate_numpy}        
        '''

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
        return self.model(t=t, k_01=k01, k_11= k11, d=dose)


'''
class Comp1GutModelFunction(tc.autograd.Function) :
    distribution_bool_matrix = [[False, True],
                                [False, True]]
    eqs, ics = LinearODE.get_eqs_and_ics(distribution_bool_matrix, False)
    function = LinearODE.solve(eqs, ics)
    functions_numpy = LinearODE.lambdify(distribution_bool_matrix, function, **LinearODE.diff_functions(distribution_bool_matrix, function, eqs))
    
    @staticmethod
    def forward(ctx, t, k01, k11, dose):

        k01, k11 = k01.detach(), k11.detach()

        t = t.detach()
        dose = dose.detach()

        output = tc.stack([tc.Tensor(fn(t.numpy(), k01.numpy(), k11.numpy(), dose.numpy(), None)) \
                           for fn in Comp1GutModelFunction.functions_numpy['function']])
                           
        ctx.save_for_backward(t, k01, k11, dose, output)

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        t, k01, k11, dose, output = ctx.saved_tensors

        grad_t = grad_k01 = grad_k11 = grad_dose = None
        if ctx.needs_input_grad[0]:
            grad_t = tc.stack([fn(t.numpy(), k01, k11, dose.numpy(), None, output.numpy().sqeeze(axis=0)) \
                               for fn in Comp1GutModelFunction.functions_numpy['diff_function_t']])*grad_output

        if ctx.needs_input_grad[1]:

            grad_k01 =[]
            for fn in Comp1GutModelFunction.functions_numpy['diff_function_distribution_rate'][0][1] :
                if fn is None :
                    grad_k01.append(tc.zeros(output.size()[-1]))
                else :
                    grad_k01.append(tc.Tensor(fn(t.numpy(), k01.numpy(), k11.numpy(), dose.numpy(), None)))
            
            grad_k01 = tc.stack(grad_k01)
            grad_k01 *= grad_output

        if ctx.needs_input_grad[2]:
            grad_k11 =[]
            for fn in Comp1GutModelFunction.functions_numpy['diff_function_distribution_rate'][1][1] :
                if fn is None :
                    grad_k11.append(tc.zeros(output.size()[-1]))
                else :
                    grad_k11.append(tc.Tensor(fn(t.numpy(), k01.numpy(), k11.numpy(), dose.numpy(), None)))
            
            grad_k11 = tc.stack(grad_k11)
            grad_k11 *= grad_output

        return grad_t, grad_k01, grad_k11, None

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