@dataclass(repr=False, eq=False)
class PredictionFunctionModule(tc.nn.Module):
    dataset : tc.utils.data.Dataset
    column_names : Iterable[str]
    theta_size : int
    eta_size : int
    eps_size : int
    pk_parameter : PKParameterGenerator
    pred_fn  : PredFunctionGenerator
    error_fn : ErrorFunctionGenerator
    theta_scale : Optional[Scale]

    def __post_init__(self):
        super(PredictionFunctionModule, self).__init__()
        self.ids = set()
        self.record_lengths : Dict[str, int] = {}
        self.max_record_length = 0
        for data in self.dataset :
            id = data[0][:, self.column_names.index('ID')][0]
            self.ids.add(int(id))
            self.record_lengths[str(int(id))] = data[0].size()[0]
            self.max_record_length = max(data[0].size()[0], self.max_record_length)

        self.theta = tc.nn.Parameter(tc.zeros(self.theta_size))
        self.etas = tc.nn.ParameterDict({})
        self.epss = tc.nn.ParameterDict({})
        with tc.no_grad() :
            for id in self.ids :
                eta_value = tc.zeros(self.eta_size)
                self.etas.update({str(int(id)): tc.nn.Parameter(eta_value)})

                eps_value = tc.zeros(self.record_lengths[str(int(id))], self.eps_size)
                self.epss.update({str(int(id)): tc.nn.Parameter(eps_value)})

        self.cov_indice = self._get_cov_indice(self.column_names)
    
    def _get_cov_indice(self, column_name) :
        ESSENTIAL_COLUMNS : Iterable[str] = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT']
        cov_indice = []
        for i, col_name in enumerate(column_name):
            if col_name not in ESSENTIAL_COLUMNS :
                cov_indice.append(i)
        return tc.tensor(cov_indice)
    
    def _get_amt_indice(self, dataset) :
        amts = dataset[:, self.column_names.index('AMT')]
        end = amts.size()[0]
        start_index = tc.squeeze(amts.nonzero(), 0)
        
        if start_index.size()[0] == 0 :
            return tc.tensor([0], device = dataset.device)
 
        if start_index[0] != 0 :
            start_index = tc.cat([tc.tensor([0], device = dataset.device), start_index], 0)
        
        if start_index[-1] != end - 1 :
            start_index = tc.cat([start_index, tc.tensor([end-1], device = dataset.device)] , 0)
 
        return start_index 
    
    def descale(self):
        with tc.no_grad() :
            if self.theta_scale is not None :
                self.theta.data = self.theta_scale(self.theta)
            self.theta_scale = None
        return self
    
    def forward(self, dataset):
        pass

@dataclass(repr=False, eq=False)
class PredictionFunctionByTime(PredictionFunctionModule):
 
    def forward(self, dataset) :
        if self.theta_scale is not None :
            theta = self.theta_scale(self.theta)
        else :
            theta = self.theta

        cov_indice = self.cov_indice.to(dataset.device)

        f = tc.zeros(dataset.size()[0], device = dataset.device)
        amt_indice = self._get_amt_indice(dataset)
        
        id = str(int(dataset[:,self.column_names.index('ID')][0]))
        eta = self.etas[id]
        eps = self.epss[id]
        
        pk_parameter_value = self.pk_parameter(theta, eta)
 
        for i in range(len(amt_indice) - 1):
            start_time_index = amt_indice[i]
 
            #누적하기 위해 앞부분 생성
            dataset_pre = dataset[:start_time_index, :]
            f_pre = tc.zeros(dataset_pre.size()[0], device = dataset.device)
 
            dataset_cur = dataset[start_time_index:, :]
            amt_cur = dataset_cur[0, self.column_names.index('AMT')]
            rate_cur = dataset_cur[0, self.column_names.index('RATE')]
            start_time = dataset_cur[0, self.column_names.index('TIME')]
            
            dataset_cur_tp = dataset_cur.transpose(0,1)
 
            times = dataset_cur_tp[self.column_names.index('TIME'), :]
            times = times - start_time
 
            cmts_cur = dataset_cur_tp[self.column_names.index('CMT'), :]
 
            cov_cur = dataset_cur_tp.index_select(0, cov_indice).unbind()
            
            f_cur = self.pred_fn(times, None, theta, eta, cmts_cur, amt_cur, rate_cur, pk_parameter_value, *cov_cur)
            f = f + tc.cat([f_pre, f_cur], 0)
        
        cmts = dataset[:, self.column_names.index('CMT')]
        cov = dataset.t().index_select(0, cov_indice).unbind()

        y_pred = self.error_fn(f, eps.t(), theta, eta, cmts, pk_parameter_value, *cov)
        mdv_mask = dataset[:,column_names.index('MDV')] == 0

        return y_pred, self.etas[id], self.epss[id], mdv_mask

@dataclass(repr=False, eq=False)
class PredictionFunctionByODE(PredictionFunctionModule):
    """
    ordinary equation solver
    Args:
        rtol: ratio tolerance about ordinary differential equation integration
        atol: absolute tolerance about ordinary differential equation integration
    """
    rtol : float = 1e-2
    atol : float = 1e-2
                
    def _get_element(self, data, name, index) :
        return data[index, self.column_names.index(name)]
 
    def ode_function(self, t, y):
        index = (self.cur_times < t).sum() -1
        cmt = self._get_element(self.cur_dataset, 'CMT', index)
        cov_cur = (cov_vector[index] for cov_vector in self.cur_cov)
        
        if self.theta_scale is not None :
            theta = self.theta_scale(self.theta)
        else :
            theta = self.theta

        return self.pred_fn(t, y, theta, self.cur_eta, cmt, None, None, self.pk_parameter_value, *cov_cur) + self.infusion_rate * (self.infusion_end_time > t)
 
    def forward(self, dataset) :
        if self.theta_scale is not None :
            theta = self.theta_scale(self.theta)
        else :
            theta = self.theta
        
        cov_indice = self.cov_indice.to(dataset.device)

        self.max_cmt = int(dataset[:,self.column_names.index('CMT')].max())

        self.cur_dataset = dataset
        self.cur_cov = self.cur_dataset.t().index_select(0, cov_indice).unbind()
        self.cur_times = self.cur_dataset[:,self.column_names.index('TIME')]
        id = str(int(dataset[:, self.column_names.index('ID')][0]))
        
        self.cur_eta = self.etas[id]
        eps = self.epss[id]

        self.pk_parameter_value = self.pk_parameter(theta, self.cur_eta)

        y_pred_arr = []
 
        y_init = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_rate = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_end_time = tc.zeros(self.max_cmt+1, device = dataset.device)
 
        amt_indice = self._get_amt_indice(dataset)
 
        for i in range(len(amt_indice) - 1):
            amt_slice = slice(amt_indice[i], amt_indice[i+1]+1)
            dataset_cur = dataset[amt_slice, :]
 
            times  = dataset_cur[:, self.column_names.index('TIME')]
 
            rate = self._get_element(dataset_cur, 'RATE', 0)
            cmt = self._get_element(dataset_cur, 'CMT', 0)
            amt = self._get_element(dataset_cur, 'AMT', 0)
            rate = self._get_element(dataset_cur, 'RATE', 0)
            if  rate == 0 :                    
                injection = tc.zeros(self.max_cmt + 1, device = dataset.device)
                
                injection[cmt.to(tc.int64)] = amt
                y_init = y_init + injection
            else :
                time = self._get_element(dataset_cur, 'TIME', 0)
 
                mask = tc.ones(self.max_cmt +1, device = dataset.device)
                mask[cmt] = 0
 
                rate_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                rate_vector[cmt] = rate
 
                infusion_during_time_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                infusion_during_time_vector[cmt] = time + amt / rate
 
                self.infusion_rate = self.infusion_rate * mask + rate_vector
                self.infusion_end_time = infusion_end_time * mask + infusion_during_time_vector
                
            result = odeint(self.ode_function, y_init, times, rtol=self.rtol, atol=self.atol)
            
            y_integrated = result
            y_init = result[-1]
            
            cmts_cur = dataset_cur[:, self.column_names.index('CMT')]
            cmt_mask = tc.nn.functional.one_hot(cmts_cur.to(tc.int64)).to(dataset.device)
            y_integrated = y_integrated.masked_select(cmt_mask==1)
            
            cov = dataset_cur.t().index_select(0, cov_indice).unbind()
 
            y_pred = self.error_fn(y_integrated, eps.t(), theta, self.cur_eta, cmts_cur, self.pk_parameter_value, *cov)
            
            y_pred_arr.append(y_pred)

        mdv_mask = dataset[:,column_names.index('MDV')] == 0

        return tc.cat(y_pred_arr), self.etas[id], self.epss[id], mdv_mask
