import torch as tc
from .misc import * 

class CovariateStep :
    def __init__(self) -> None:
        pass

    def covariance_step(self) :

        dataset = self.pred_function_module.dataset
 
        cov_mat_dim =  self.pred_function_module.theta.size()[0]

        for tensor in self.differential_module.omega :
            cov_mat_dim += tensor.size()[0]

        for tensor in self.differential_module.sigma :
            cov_mat_dim += tensor.size()[0]
 
        r_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
 
        s_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)

        dataloader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
 
        for data, y_true in dataloader:
            
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask = self(data)

            id = str(int(data[:,self.pred_function_module.column_names.index('ID')][0]))
            print('id', id)
 
            y_pred = y_pred.masked_select(mdv_mask)
            g = g.t().masked_select(mdv_mask).reshape((self.pred_function_module.eta_size,-1)).t()
            h = h.t().masked_select(mdv_mask).reshape((self.pred_function_module.eps_size,-1)).t()
 
            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
            gr = tc.autograd.grad(loss,[self.pred_function_module.theta, *self.differential_module.omega, *self.differential_module.sigma], create_graph=True, retain_graph=True, allow_unused=True)

            gr_cat = tc.cat(gr)
            
            with tc.no_grad() :
                s_mat.add_((gr_cat.detach().unsqueeze(1) @ gr_cat.detach().unsqueeze(0))/4)
            
            for i, gr_cur in enumerate(gr_cat) :
                hs = tc.autograd.grad(gr_cur, [self.pred_function_module.theta, *self.differential_module.omega, *self.differential_module.sigma], create_graph=True, retain_graph=True, allow_unused=True)

                hs_cat = tc.cat(hs)
                for j, hs_elem in enumerate(hs_cat) :
                    r_mat[i,j] = r_mat[i,j] + hs_elem.detach()/2

        invR = r_mat.inverse()
        
        cov = invR @ s_mat @ invR
        
        se = cov.diag().sqrt()
        
        
        correl = covariance_to_correlation(cov)
        
        ei_values, ei_vectors = correl.symeig(eigenvectors=False)
        ei_values_sorted, _ = ei_values.sort()
        inv_cov = r_mat @ s_mat.inverse() @ r_mat
        
        return {'cov': cov, 'se': se, 'cor': correl, 'ei_values': ei_values_sorted , 'inv_cov': inv_cov, 'r_mat': r_mat, 's_mat':s_mat}
        # return {'cov': cov, 'se': se, 'cor': correl, 'inv_cov': inv_cov, 'r_mat': r_mat, 's_mat':s_mat}