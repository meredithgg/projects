'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Meredith Green
CS 251 Data Analysis Visualization
Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        #info to change normalization
        self.og_mins = None
        self.og_ranges = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!


        found covariance equation here: https://corporatefinanceinstitute.com/resources/knowledge/finance/covariance/
        '''

        #center the data
        data = data - data.mean(axis = 0)

        #divide by length - 1
        denom = len(data) - 1

        dot = np.dot(data.T, data)
        
        cov_mat = dot/denom

        return cov_mat

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs


        Based on notebook from day 20
        '''

        #find the total sum of eigen values
        e_vals_sum = np.sum(e_vals)

        #find the order of eigen values
        e_vals_order = np.argsort(e_vals)[::-1]

        #sort the eigenvalues
        e_vals_sorted = e_vals[e_vals_order]

        #list of proportional variance
        values =[]
        for val in e_vals_sorted:
                values.append(val / e_vals_sum)

        return values

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs

        based on day 20 notebook
        '''
        sums = []
        cumsum = 0
        for var in prop_var:
            cumsum += var
            sums.append(cumsum)

        return sums


    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).

        normalization based on ulitify function local max-min normalization 
        --> however did it without matrix multiplication because was having issues

        help from day 21 notebook
        '''
        self.vars = vars
        #select relevant data and make it an array
        self.A = np.array(self.data[self.vars])

        self.normalized = normalize

        #normalize so from range 0 to 1
        if self.normalized:
            #info needed to undo
            # save the min and the range --> storing info
            self.og_mins = self.A.min(axis = 0).T
            self.og_ranges = self.A.max(axis = 0).T - self.og_mins

            #no zeros in denominator
            if (len(np.where(self.og_ranges == 0)[0])) == 0:
                self.A = (self.A - self.og_mins) / self.og_ranges

        #covariance matrix
        cov_mat = self.covariance_matrix(self.A)

        #vales and vecs
        (self.e_vals, self.e_vecs) = np.linalg.eig(cov_mat)

        #prob variables
        self.prop_var = self.compute_prop_var(self.e_vals)

        #cumulative sums
        self.cum_var = self.compute_cum_var(self.prop_var)
        

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method

        based on day 19 notebook 
        '''
        fig = plt.figure(figsize=(15,8))
        ax1 = fig.add_subplot(111)

        #plotting cumulative sums
        #with enlarged markers

        if num_pcs_to_keep == None:
            num_pcs_to_keep = len(self.cum_var)

        ax1.plot(self.cum_var[:num_pcs_to_keep],"go", markersize=10)

        ax1.set_ylim([0,1.03])

        #x and y labels
        ax1.set_xlabel('Number of Principal Components')
        ax1.set_ylabel('Cumulative Variance')
        if(num_pcs_to_keep < 15):
            ax1.set_xticks(range(0,num_pcs_to_keep), range(1, num_pcs_to_keep + 1))
        else:
            ax1.set_xticks(range(0,num_pcs_to_keep, int(num_pcs_to_keep/5)), range(1, num_pcs_to_keep + 1, int(num_pcs_to_keep/5)))
        ax1.set_title('Elbow Plot -- Cum Var vs # PC')

        pass

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`

        based on day 19 notebook
        '''
        #get the correct eigenvectors to keep
        v = self.e_vecs[:,pcs_to_keep]
    
        #project
        projected = self.A@v

        #assign A_proj
        self.A_proj = projected

        return projected

    def pca_project2(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`

        based on day 19 notebook
        '''
        #get the correct eigenvectors to keep
        cols_to_del = range(pcs_to_keep, self.e_vecs.shape[1])
        v = np.delete(self.e_vecs, cols_to_del, 1)
    
        #project
        projected = self.A@v

        #assign A_proj
        self.A_proj = projected

        return projected

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.

        based on day 21 notebook
        '''
        v = self.e_vecs[:, :top_k]
        projected = self.A_proj@v.T
        if self.normalized:
    
            #no zeros in denominator
            if (len(np.where(self.og_ranges == 0)[0])) == 0:
                projected = projected *self.og_ranges + self.og_mins
            
        return projected
