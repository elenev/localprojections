import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections.abc import Iterable
from patsy import dmatrices
from linearmodels.panel import PanelOLS
from matplotlib.colors import to_rgb

# Sources: lpirfs R package (https://journal.r-project.org/archive/2019/RJ-2019-052/RJ-2019-052.pdf)
# Vignette: https://adaemmerp.github.io/lpirfs/README_docs.html
# Jorda (2005) paper: https://www3.nd.edu/~nmark/Climate/Jorda%20-%20Local%20Projections.pdf


class LP:
    r"""
    The LP class implements the local projections methodology of Jorda (2005) and the R package 
    lpirfs.

    Parameters
    ----------
    df : pandas.DataFrame 
        DataFrame containing the time series or panel data to be used in the estimation.
    maxlags : int, optional
        Maximum number of lags to include in the model. The default is 1.
    sample : function, optional
        Function that takes a pandas.DataFrame and returns a boolean array of the same length 
        indicating whether to include that observation in the estimation. The default is True for all.
        Reason for this is that the user may want to exclude observations after lags are computed.
    endogvars : dict or list, optional
        Lagged variables to include in the model. The default is None, in which case all variables
        are included. If a dict is supplied, the keys are the names of the variables and the values
        are the lags to include. Lags can be an integer ``n``, in which case lags will be 1,...,``n''.
        Lags can also be a tuple ``(n,m)``, in which case lags will be ``n,...,m```. Or they can be an
        exact list of lags. If a list is supplied, maxlags is used to determine the lags.
    responsevars : list or string, optional
        List of variables to use as response variables. The default is None, in which case all
        ``endogvars`` are used.
    shocks : list or string, optional
        List of variables to use as shocks. The default is None, in which case endogvars are used.
        If shocks are not supplied, identification must be supplied to indicate how reduced-form
        innovations in endogvars are transformed into structural shocks.
    ci : float, optional
        Confidence interval to use for the impulse response plots. The default is 90%.
    use_t : bool, optional
        In the case of time series data, whether to use t-statistics instead of z-statistics. The default
        is True. This is ignored for panel data, which always uses t-statistics.
    use_correction : bool, optional
        In the case of time series data, whether to use small sample correction for the standard errors.
        The default is True. This is ignored for panel data, which always uses small sample correction.
    interaction : string, optional
        Name of the variable to use for interaction terms. The default is None, in which case no
        interaction terms are used. If a categorical variable is supplied, a separate IRF path is 
        computed for each level. If a continuous variable is supplied, the base effect and interaction
        effect are computed.
    timelevel : string, optional
        Name of the time index in the DataFrame. The default is None, in which case the last index
        is assumed to be the time index.
    cluster : bool or string, optional
        Whether to use clustered standard errors in panel estimation. If True, all index levels
        other than the time index are used for clustering. If a string, that variable is used.
        If None, Driscoll-Kraay standard errors with Bartlett Kernel are used with a bandwidth of
        ``horizon + 1``. cluster is ignored for time series data, which always uses Newey-West standard
        errors with a bandwidth of ``horizon + 1``.
    identification : string or 2D ndarray, optional
        Method to use for identification if no shocks are supplied. 
        The default is ``'cholesky'``, in which case the Cholesky decomposition of the reduced-form
        covariance matrix is used using the ordering in endogvars. If a 2D ndarray is supplied, 
        it is used as the identification matrix.
    trend : int, optional
        Order of polynomial trend to include in the model. The default is 0, in which case no trend
        is included. If 1, a linear trend is included. If 2, a quadratic trend is included, etc.

    Notes
    -----
    Jorda (2005) defines local projections as a series of separately estimated regressions where
    a shock at time $t$ is used to predict the response variable at time $t+h$ for $h=0,1,...$.

    Without exogenously identified shocks, the model is written as:
    .. math::

        y_{t+h} = \\alpha_{h} + \\sum_{s=1}^p \\beta_{h,s}' x_{t-s} + \\epsilon_{t+h}, h=0,1,...

    where :math: `y_{t+h}` is the response variable at time :math: `t+h`, :math: `x_{t-s}` is 
    the vector of endogenous variables at time :math: `t-s` (potentially including :math: `y_{t}`).
    The impulse response to the reduced-form shock is :math: ``\\hat{beta}_{h,1}``. It is transformed
    into a structural shock using the identification matrix :math: `B` such that 
    .. math::

        I(h) = B \\hat{\\beta}_{h,1}

    :math: `B` is computed separately. By default, it will be estimated using the Cholesky decomposition
    of the equivalent VAR(p) model. Alternatively, the user can supply an identification matrix.

    With exogenously identified shocks, the model is written as:
    .. math::

        y_{t+h} = \\alpha_{h} + \\beta_h' z_t + \\sum_{s=1}^p \\gamma_{h,s}' x_{t-s} + \\epsilon_{t+h}, h=0,1,...
    
    A panel version estimates effect :math: `\\beta_h` of :math: `x_{i,t-1}` on :math: `y_{i,t+h}` using
    fixed effects :math: `\\alpha_{i,h}`.
    """

        
    def __init__(self, df, \
       maxlags=1, \
       sample=None, \
       endogvars=None, \
       responsevars=None, \
       shocks=None, \
       ci=0.9, \
       use_t=True, \
       use_correction=True, \
       interaction=None, \
       timelevel=None,
       cluster=None, \
       identification='cholesky', \
       trend=0):
        
        # Define defaults
        if sample is None:
            sample = lambda x: [True]*x.shape[0]
        if endogvars is None:
            endogvars = df.columns

        # If endogvars is a dict, flatten to figure out what the maxlags is
        # across all endogneous variables
        if isinstance(endogvars,dict):
            maxlags = max(flatten(endogvars.values())) if endogvars else 0
        else:
            endogvars = set_lags(endogvars,maxlags)

        if responsevars is None:
            responsevars = endogvars.keys()

        
        # In case there's only one response variable supplied as a scalar,
        # this will make sure it's a list so rest of the code can deal with it
        responsevars = make_iterable(responsevars)

        # In case there's only one response variable supplied as a scalar,
        # this will make sure it's a list so rest of the code can deal with it
        shocks = make_iterable(shocks)

        # Only keep the columns of df that we need
        allvars = endogvars.keys()|set(responsevars)
        if shocks is not None:
            allvars.update(shocks)
        if interaction is not None:
            allvars.add(interaction)
        if isinstance(cluster,str):
            allvars.add(cluster)

        allvars = list(allvars)
        df = df[allvars]

        if (timelevel is None):
            # Panel dataset but timelevel not supplied. Assume it's the last one.
            timelevel = df.index.names[-1]

        # Determine index names and whether the data is time series or panel
        nlevels = df.index.nlevels
        index_names = df.index.names
        df = df.reset_index()

        # If timelevel is a period, convert to numeric. PanelOLS can't handle periods.
        if pd.api.types.is_period_dtype(df[timelevel]):
            pers = df[timelevel].unique()
            min_per = pers.min()
            persdict = {p:(p-min_per).n for p in pers}
            df[timelevel] = df[timelevel].map(persdict)

        # If using a trend, set the trend variable to the time series.
        if trend>0:
            df['trend'] = df[timelevel]

        if (nlevels>1):
            # Panel dataset.
            # Identify cross-sectional and time index levels
            cxlevels = np.setdiff1d(index_names, timelevel)

            # If there are more than two CX levels, group them into one
            # So that the dataset only has 2 levels, as PanelOLS expects.
            if len(cxlevels)>1:
                cxlevel = 'cxid'
                df[cxlevel] = df.apply(lambda r: tuple(r[v] for v in cxlevels), axis=1)
                df.drop(columns=cxlevels, inplace=True)
            else:
                cxlevel = cxlevels[0]

            # Order indices so that timelevel is last
            df.set_index([cxlevel,timelevel], inplace=True)
        else:
            # Time series dataset
            df.set_index(timelevel, inplace=True)
            cxlevel=None

        # Fill in missing values in the index. This is necessary so that shift()
        # works properly.
        df = fill_index_level(df,timelevel)

        self.df = df
        self.maxlags = maxlags
        self.sample = sample
        self.endogvars = endogvars
        self.responsevars = responsevars
        self.shocks = shocks
        self.ci = ci
        self.use_t = use_t
        self.use_correction = use_correction
        self.interaction = interaction
        self.timelevel = timelevel
        self.cxlevel = cxlevel
        self.cluster = cluster
        self.trend = trend

        if shocks is None:
            # No shocks are supplied. Need to identify them.
            self.offset = 0
            self.ortho = self.orthogonalize(identification)
        else:
            self.offset = 1
            self.ortho = lambda b: b

        
    def gen_rhs(self, endogvars=None, shocks=None, responsevars=None, use_interaction=True):
        r"""
        This function generates the right-hand side of the model. It is used internally by the
        ``estimate`` and ``orthogonalize`` methods.

        Parameters
        ----------
        endogvars : dict, optional
            Lagged variables to include in the model. The default is None, in which the object
            instance's ``endogvars`` attribute is used.
        shocks : list, optional
            List of variables to use as shocks. The default is None, in which case the object
            instance's ``shocks`` attribute is used.
        responsevars : list, optional
            List of variables to use as response variables. The default is None, in which case the object
            instance's ``responsevars`` attribute is used.
        use_interaction : bool, optional
            Whether to include interaction terms. The default is True, which means the instance's
            ``interaction`` attribute is used. False means no interaction terms are used, whatever
            the instance's ``interaction`` attribute is. This is useful for the ``orthogonalize``
            method, which estimates a VAR model and isnt' compatible with interaction terms.

        Returns
        -------
        rhs : string
        """
        if endogvars is None: endogvars = self.endogvars
        if shocks is None: shocks = self.shocks
        if responsevars is None: responsevars = self.responsevars
        
        # Creates a lag range for a given endogenous variables, whatever the input is
        def make_lag_range(var):
            lag_input = endogvars[var]
            if np.isscalar(lag_input):
                # Only max lag is specified. Include lags from 1 to lags.
                # In other words, by default we don't include contemporaneous values.
                # Exception: if lag_input = 0, then we do include contemporaneous value only.
                return range(min(lag_input,1),lag_input+1)
            elif isinstance(lag_input,tuple):
                # Lag range is specified. If the lag range starts at 0 (include contemporaneous),
                # check if the variable is in the responsevars list. If it is, it doesn't make
                # sense to include the contemporaneous value as a control.
                if (var in responsevars) and (lag_input[0]==0):
                    start = 1
                else:
                    start = lag_input[0]
                return range(start,lag_input[1]+1)
            elif isinstance(lag_input,list):
                # Specific lags are specified.
                return lag_input
            else:
                raise Exception(f'Invalid lag format {lag_input} for variable {var}')

        # Create lag ranges for each endogenous variable
        lags = {var:make_lag_range(var) for var in endogvars}

        # Join lags into a formula sum
        make_lags = lambda envar: '+'.join([f'lag({envar},{i})' \
                            for i in lags[envar]])
        
        # for each endogenous variable
        rhs_endog = ' + '.join([make_lags(envar) for envar in endogvars])


        # namedict maps the coefficient labels in the regression output to
        # impulse labels in the IRF output.
        if shocks is not None:
            # If shocks are supplied, use them as RHS variables.
            rhs_shock = ' + '.join(shocks)
            namedict = {shock: shock for shock in shocks}
        else:
            # If shocks are not supplied, use the endogenous variables as RHS variables.
            namedict = {f'lag({envar}, 1)':envar for envar in endogvars}

        # Add trend
        rhs = make_polynomial('trend',self.trend,post=' + ') + \
            '(' + ' + '.join(filter(None,[rhs_endog,rhs_shock])) + ')'

        # Account for interactions
        intr = self.interaction
        df = self.df
        if use_interaction and intr is not None:
            if pd.api.types.is_categorical_dtype(df[intr]):
                # If the interaction variable is categorical, create a separate
                # IRF path for each level. Do not include base terms so that each level
                # can be separately identified.
                operator = ':'

                # Create a cartesian product between interaction factors and impulse labels
                intrlevels = df[intr].unique()
                intdict = {f'{intr}[{lev}]':lev for lev in intrlevels}
                namedict = {k+':'+i:(namedict[k],intdict[i]) for k in namedict for i in intdict}
            else:
                # If the interaction variable is continuous, create a separate
                # IRF path for the base effect and the interaction effect. Include base levels.
                operator = '*'

                # Create a cartesian product between the (base,interaction) types and impulse labels
                intdict = {'':'Base', ':'+intr:intr}
                namedict = {k+i:(namedict[k],intdict[i]) for k in namedict for i in intdict}

            interaction_str = f' {operator} {self.interaction}' if use_interaction else ''
            rhs += interaction_str
        
        return rhs, namedict
    
    def design_matrices(self,rhs,responsevar):
        r"""
        This function generates the design matrices for the regression. It is used internally by the
        ``estimate`` and ``orthogonalize`` methods. It will always return the design matrices for
        a contemporaneous regression. Subsequent lags are handled by the ``shift_lhs`` method. This is
        done for efficiency, since creating the design matrices can be quite slow.

        Parameters
        ----------
        rhs : string
            Right-hand side of the regression.
        responsevar : string
            Response variable of the regression (LHS)

        Notes
        -----
        This function uses the ``patsy`` package to generate the design matrices. Maybe in the future
        we will switch to ``formulaic`` which is faster.

        Returns
        -------
        dflhs : DataFrame
            LHS design matrix
        dfrhs : DataFrame
            RHS design matrix
        """
        sample = self.sample
        df = self.df

        formula = responsevar + ' ~ ' + rhs
        dflhs, dfrhs = dmatrices(formula,data=df, \
                    return_type='dataframe') 
        dflhs = dflhs.loc[sample, :]
        dfrhs = dfrhs.loc[sample, :]

        return dflhs, dfrhs
    
    def shift_lhs(self, dflhs, dfrhs, horizon=0):
        r"""
        This function shifts the LHS design matrix by the horizon. It is used internally by the
        ``estimate`` and ``orthogonalize`` methods to compute the impulse response functions at
        different horizons. After shifting, the resulting missing rows are removed.

        Parameters
        ----------
        dflhs : DataFrame
            LHS design matrix
        dfrhs : DataFrame
            RHS design matrix
        horizon : int, optional
            Horizon to shift the LHS design matrix. The default is 0, which means no shift.

        Notes
        -----
        This function creates copies of the design matrices because the caller may need to use
        the original matrices for other horizons. We can refactor this later to avoid the copies
        if we can be sure that horizons are always incremented sequentially.

        Returns
        -------
        dflhs : DataFrame
            LHS design matrix shifted by the horizon.
        dfrhs : DataFrame
            RHS design matrix shifted by the horizon.
        """
        adjhorizon = horizon-1
        if adjhorizon!=0:
            dflhs = lag(dflhs,-adjhorizon)
            new_missing = pd.isna(dflhs.iloc[:,0])
            #dflhs=dflhs.drop(new_missing)
            #dfrhs=dfrhs.drop(new_missing)
            dflhs = dflhs.loc[~new_missing,:].copy()
            dfrhs = dfrhs.loc[~new_missing,:].copy()

        return dflhs, dfrhs

    def run_regression(self, dflhs, dfrhs, nwlags=0):
        r"""
        This function runs the regression for a given horizon. It is used internally by the
        ``estimate`` and ``orthogonalize`` methods. 

        Parameters
        ----------
        dflhs : DataFrame
            LHS design matrix
        dfrhs : DataFrame
            RHS design matrix
        nwlags : int, optional
            Number of Newey-West lags to use. The default is 0, which means no Newey-West correction.

        Retruns
        -------
        out : DataFrame
            DataFrame containing the regression results. The columns are:
                - ``params``: regression coefficients
                - ``lb``: lower bound of the confidence interval
                - ``ub``: upper bound of the confidence interval
        resids : Series
            Series containing the regression residuals.
        fit : RegressionResults from OLS or PanelOLSResults from PanelOLS
        """

        # Retrieve confidence interval level
        ci = self.ci
        if dflhs.index.nlevels>1:
            # Panel regression
            cluster = self.cluster

            # drop singletons
            drop_idx = drop_singletons(dflhs, self.cxlevel)
            dflhs.drop(drop_idx, inplace=True)
            dfrhs.drop(drop_idx, inplace=True)

            # Define regression model using ``linearmodels''' PanelOLS
            # Drop intercept column from the RHS
            mdl = PanelOLS(dflhs, dfrhs.iloc[:,1:], entity_effects=True)
            if cluster == True:
                # Cluster by entity
                cov_args = {'cov_type':'clustered', 'cluster_entity':True}
            elif cluster is not None:
                # Cluster by a specific variable
                s_cluster = self.df.loc[dflhs.index,cluster]
                cov_args = {'cov_type':'clustered', 'clusters':s_cluster}
            else:
                # Driscoll-Kraay standard errors
                cov_args = {'cov_type':'kernel','bandwidth':nwlags+1}
            
            # Run regression
            fit = mdl.fit(**cov_args)

            # Save coefficients and bounds
            params = fit.params
            ci = fit.conf_int(ci)
            lb = ci.loc[:,'lower']
            ub = ci.loc[:,'upper']

            # Save residuals (e.g. for covariance matrix estimation)
            resids = fit.resids
        else:
            # Time series regression
            use_t = self.use_t
            use_correction = self.use_correction

            # Define regression model using ``statsmodels''' OLS
            mdl = sm.OLS(endog=dflhs, exog=dfrhs)

            # Run regression with Newey-West SEs
            fit = mdl.fit(cov_type='HAC', \
                    cov_kwds={'maxlags':nwlags, 'use_correction':use_correction}, \
                    use_t=use_t)
            
            # Save coefficients and bounds
            params = fit.params
            ci = fit.conf_int(1-ci)
            lb = ci[0]
            ub = ci[1]

            # Save residuals (e.g. for covariance matrix estimation)
            resids = fit.resid

        # Combine coefficients and bounds into a single DataFrame
        out = pd.concat([params,lb,ub], keys=['coef','lb','ub'], axis=1)

        return out, resids, fit

    def estimate_var(self):
        r"""
        This function estimates a VAR model. It is used internally by the ``orthogonalize`` method
        to estimate the covariance matrix of the residuals, which is used to orthogonalize the
        shocks.
        
        Returns
        -------
        const : DataFrame
            DataFrame containing the constant term of the VAR model.
        outcoefs : dict
            Dictionary containing the coefficients of the VAR model. The keys are:
                - ``coef``: coefficients
                - ``lb``: lower bound of the confidence interval
                - ``ub``: upper bound of the confidence interval
            Each dictionary value is a DataFrame with a row for each response variable and
            a column for each impulse variable and lag. ``df[y,(x,s)]`` represents ``(y,x)``th element
            of the ``s``th lag of the VAR.
        Sigma : DataFrame
            DataFrame containing the covariance matrix of the residuals.
        """

        # Since it's a VAR, use the same number of lags for each endogenous variable
        endogvars = set_lags(self.endogvars,self.maxlags)

        # Generate the RHS of the regression
        rhs, _ = self.gen_rhs(endogvars,responsevars=endogvars.keys(),use_interaction=False)

        # Run the regression for a given response variable.
        def estimate_var_responsevar(responsevar):
            dflhs, dfrhs = self.design_matrices(rhs,responsevar)
            out, resids, _ = self.run_regression(dflhs, dfrhs)
            out.index.rename('coefname', inplace=True)
            return out, resids

        # Run the regressions
        outs = [estimate_var_responsevar(v) for v in endogvars]
        
        # Compute covariance matrix of residuals
        resids = pd.concat([f[1] for f in outs], keys=endogvars, axis=1)
        Sigma = resids.cov(ddof=1)

        # Combine coefficients into a single DataFrame
        coefs = pd.concat([f[0] for f in outs], keys=endogvars, names=['response']).reset_index()
        coefs['response'] = pd.Categorical(coefs['response'], categories=endogvars, ordered=True)
    
        levels_order = ['coefname','response']
        coefs.sort_values(levels_order, inplace=True)
        coefs.set_index(levels_order, inplace=True)

        # Pivot the DataFrame to get the coefficients in the right shape
        coefs.reset_index(inplace=True)
        const = coefs.loc[coefs['coefname']=='Intercept',:]
        coefs = coefs.loc[coefs['coefname']!='Intercept',:]
        coefs[['impulse','lag']] = coefs['coefname'].str.extract(r'lag\((.*?), ([0-9]+)\)', expand=True)
        coefs['impulse'] = pd.Categorical(coefs['impulse'], \
                                        categories=endogvars, \
                                            ordered=True)
        coefs.sort_values(['response','lag','impulse'],inplace=True)

        varnames = ('coef','lb','ub')
        outcoefs = {v:coefs.pivot(index='response',columns=['lag','impulse'],values=v) for v in varnames}

        return const, outcoefs, Sigma       

    def orthogonalize(self,identification):
        r"""
        This function orthogonalizes the shocks using the supplied identification strategy.

        Parameters
        ----------
        identification : string or ndarray
            If ``identification`` is a string, it must be one of the following:
                - ``'cholesky'``: use the Cholesky decomposition of the covariance matrix of the residuals.
            Otherwise, ``identification`` must be a square matrix with the same number of rows as
            endogenous variables. The matrix must be invertible.

        Returns
        -------
        ortho : function
            Function that takes a vector of shocks and returns a vector of orthogonalized shocks.
        """
        endogvars = self.endogvars

        if identification == 'cholesky':
            # Use Cholesky decomposition of the covariance matrix of the residuals
            _, _, Sigma = self.estimate_var()
            
            # Compute Cholesky decomposition and label axes
            D = pd.DataFrame(np.linalg.cholesky(Sigma),index=endogvars,columns=endogvars)
            for a in [0,1]: D.rename_axis('impulse', inplace=True, axis=a)
            #varirfs = varirf(outcoefs['coef'],maxhor,D=D)
            return lambda b: D @ b
        elif isinstance(identification,np.ndarray):
            # Use supplied identification matrix
            return lambda b: identification @ b
        else:
            raise Exception("Neither shock nor identification strategy supplied.")
            

    def estimate(self,max_horizon,shock_size=None):
        r"""
        This method estimates the impulse response functions using local projections.

        Parameters
        ----------
        max_horizon : int
            Maximum horizon to compute the impulse response functions.
        shock_size : float, optional
            Size of the shock. The default is 1 for identified shocks and standard deviation
            for orthogonalized shocks.

        Returns
        -------
        coefs : DataFrame
            DataFrame containing the impulse response functions. The indices are:
                - ``response``: response variable
                - ``impulse``: impulse variable
                - ``horizon``: horizon
                - ``interaction``: interaction variable (if applicable)
            The columns are:
                - ``coef``: impulse response function
                - ``lb``: lower bound of the confidence interval
                - ``ub``: upper bound of the confidence interval
        regresults : dict
            Nested dictionary containing the regression results. The keys are the horizons
            and the values are dictionaries containing the regression results for each spec.
        """

        # Extract attributes
        endogvars = self.endogvars
        shocks = self.shocks
        responsevars = self.responsevars
        offset = self.offset
        ortho = self.ortho

        # Define range of horizons
        horizons = range(0,max_horizon+1)

        # Generate the RHS of the regression
        rhs, namedict = self.gen_rhs()

        # Run the regression for a given response variable and horizon.
        def est_var_h(h,dflhs,dfrhs):
                dflhs, dfrhs = self.shift_lhs(dflhs, dfrhs, h)
                out, _, fit = self.run_regression(dflhs, dfrhs, nwlags=h)
                out = out.loc[namedict.keys(),:]
                out.rename(index=namedict,inplace=True)

                # Add interactions to the index
                if self.interaction is not None:
                    out.set_index(pd.MultiIndex.from_tuples( \
                        out.index, names=('impulse','interaction')), inplace=True)
                    out = out.groupby('interaction', group_keys=False).apply(ortho)
                else:
                    out.index.rename('impulse', inplace=True)
                    out = ortho(out)
                return out, fit

        # Run the regressions for a given response variable.
        def est_var(responsevar):
            dflhs, dfrhs = self.design_matrices(rhs,responsevar)
            out = {h:est_var_h(h+offset,dflhs,dfrhs) for h in horizons}
            coefs = pd.concat([out[h][0] for h in horizons], keys=horizons, names=['horizon'])
            regresults = {h:out[h][1] for h in horizons}
            return coefs, regresults
        
        # Run the regressions
        out = [est_var(v) for v in responsevars]
        coefs = pd.concat([x[0] for x in out], keys=responsevars, names=['response'])
        regresults = {v:x[1] for (v,x) in zip(responsevars,out)}

        # Organize the output
        coefs.reset_index(inplace=True)
        impulses = [x for x in [*shocks, *endogvars] if x in set(coefs['impulse'])]
        coefs['impulse'] = pd.Categorical(coefs['impulse'], categories=impulses, ordered=True)
        coefs['response'] = pd.Categorical(coefs['response'], categories=responsevars, ordered=True)
        
        # Set index and sort
        levels_order = ['impulse','response','horizon']
        if self.interaction is not None:
            levels_order.insert(2,'interaction')
        coefs.sort_values(levels_order, inplace=True)
        coefs.set_index(levels_order, inplace=True)
    
        # Rescale coefficients by shock_size
        if shock_size is not None:
            if self.shocks is not None:
                shock_size = make_iterable(shock_size,shocks)
                index_names = coefs.index.names
                coefs.reset_index(inplace=True)
                for k in shock_size:
                    coefs.loc[coefs['impulse']==k,'coef':'ub'] *= shock_size[k]
                coefs.set_index(index_names, inplace=True)
            else:
                impulses = coefs.index.get_level_values('impulse').unique()
                shock_size = make_iterable(shock_size,impulses)
                idx = pd.IndexSlice
                for impulse in impulses:
                    coefs.loc[idx[impulse,:,:],:] /= (coefs.loc[idx[impulse,impulse,0],'coef'] / shock_size[impulse])

        return coefs, regresults
            
    
# -------------------------- #
# --- Helper functions  ---- #
# -------------------------- #


def set_lags(endogvars_list, maxlags):
    """
    Assign the same max lags for each endogenous variable. Returns a dict 
    with the same keys as ``endogvars_list`` and the same value for all keys.
    """
    return dict( zip(endogvars_list, [maxlags]*len(endogvars_list)) )

def flatten(nested_list):
    """
    Recursively flatten a nested list
    """
    flat_list = []
    stack = [nested_list]
    
    while stack:
        sublist = stack.pop()
        for item in sublist:
            if isinstance(item, list) or isinstance(item,tuple):
                stack.append(item)
            else:
                flat_list.append(item)
                
    return flat_list

def drop_singletons(df,level):
    """
    Find levels of a MultiIndex that have only one value and drop them.
    """
    all_levels = df.index.names
    df = df.reset_index()
    counts = df[all_levels].groupby(level)['qdate'].count()
    counts = counts[counts==1]
    #return df.loc[df[level].isin(counts.index),:].set_index(all_levels)
    return counts.index

def fill_index_level(df, level=0):
    """
    Fill in missing values in a dataframe index. For MultiIndexes, this doesn't necessarily generate
    a balanced panel. It just fills in missing times within the entity-specific range of times.

    Parameters
    ----------
    df : DataFrame
        DataFrame with a MultiIndex to fill.
    level : int or str, optional
        Level of the index to fill. The default is 0.

    Returns
    -------
    df : DataFrame
        DataFrame with filled index.
    """
    all_levels = df.index.names

    # Get level name
    if type(level)==int:
        level=all_levels[level]

    # Figure out the type of range we will need to create
    dfi=df.reset_index()
    if pd.api.types.is_period_dtype(dfi[level]):
        # Periods have their own range function
        range_fcn = lambda min, max: pd.period_range(min, max + 1)
    else:
        range_fcn = lambda min, max: range(min, max+1)
    
    # Find other levels (if any)
    other_levels = [x for x in all_levels if x != level]
    if len(other_levels)>0:
        # It's a MultiIndex. Fill in missing values within each entity.
        extrema=dfi.groupby(other_levels)[level].agg(['min','max'])
        ranges=extrema.apply(lambda x: range_fcn(x['min'], x['max']),axis=1)

        # Function to return a tuple of the index values
        if len(other_levels)==1:
            # There is one other index
            tup = lambda idx, t: (idx,t)
        else:
            # There are multiple other indices
            tup = lambda idx, t: (*idx,t)
        
        # Create list of tuples and combine them into an index
        ranges=[tup(idx, t) for idx in ranges.index for t in ranges[idx]]
        new_idx=pd.MultiIndex.from_tuples(ranges)
        new_idx.rename(df.index.names, inplace=True)

    else:
        # It's a regular index. Fill in missing values between the min and max.
        idx=df.index
        new_idx=range_fcn(idx.min(),idx.max())

    return df.reindex(new_idx)

def make_polynomial(var, n, pre='', post=''):
    """
    Create a formula for a polynominal of degree `n` in `var`, prepending or postpending
    additional strings if needed.
    """
    if n==0:
        return ''
    
    return pre + '+'.join([f'I({var}**{i})' for i in range(1,n+1)]) + post


def make_iterable(x,n=1):
    """
    Make an object iterable. If ``x`` is already iterable, return it.
    Otherwise, if
         - ``n`` is an integer, return a list of length n with each element equal to ``x``
         - ``n`` is a list, return a dictionary with keys ``n`` and values ``x`` 
    """
    if x is None:
        return None
    if isinstance(x,Iterable) and not isinstance(x,str):
        return x
    else:
        if type(n)==int:
            return [x]*n
        else:
            return dict(zip(n,[x]*len(n)))

def lag(x, n=0, cxlevel=None):
    """
    Lag by ``n`` periods. If ``cxlevel`` is supplied, lag within
    each cross-section.

    Parameters
    ----------
    x : Series, DataFrame, or ndarray
        Object to lag.
    n : int, optional
        Number of periods to lag. The default is 0. Negative numbers shift forward.
    cxlevel : str or list, optional
        Name of the cross-section level(s) of the index. The default is None.
    """

    if n == 0:
        # No lag
        return x
    if isinstance(x,pd.Series) or isinstance(x,pd.DataFrame):
        # Pandas object: use shift()
        if x.index.nlevels==1:
            # Single index: use shift()
            return x.shift(n)
        else:
            # MultiIndex: use groupby, then shift
            if cxlevel is None:
                # Panel dataset, but CX index not supplied.
                # Assume last index is time, so all others are CX.
                cxlevel = x.index.names[0:-1]
            return x.groupby(level=cxlevel).shift(n)
    else:
        # Numpy array: use numeric indices
        x = x.copy()
        x[n:] = x[0:-n]
        x[:n] = np.nan
    return x


# -------------------------- #
# --- Plotting functions --- #
# -------------------------- #

def plot_irf(dftmp, impulse=None, response=None, interaction=None,\
             colorlevel='interaction', colorvalues=None, \
             ax=None,colormap=None,legend_here=True, \
             title_fcn=None):
    """
    Plot a single impulse response function. This function can be called directly
    by the user or by the ``plot_irfs`` function that plots a grid of IRFs.

    Parameters
    ----------
    dftmp : DataFrame
        DataFrame containing the impulse responses. The expected contents of the DataFrame
        depend on how the function is called.
            - If ``impulse`` and ``response`` are both supplied, then ``dftmp`` can contain multiple
                IRFs, but its indices must be, in order, ``impulse``, ``response``, ``horizon``.
            - If ``impulse`` and ``response`` are not supplied, then ``dftmp`` must contain a 
                single IRF.
        In either case, the DataFrame must contain the following columns:
            - ``coef``: impulse response function
            - ``lb``: lower bound of the confidence interval
            - ``ub``: upper bound of the confidence interval
        If ``colorlevel`` is supplied, then the DataFrame must also contain a column with the
        name ``colorlevel``.
    impulse : str, optional
        Name of the impulse variable. The default is None, in which case the DataFrame must
        contain a single IRF.
    response : str, optional
        Name of the response variable. The default is None, in which case the DataFrame must
        contain a single IRF.
    interaction : str, optional
        Name of the interaction variable. The default is None.
    colorlevel : str, optional
        Name of the column in ``dftmp`` that indexes the dimension that will be represented by color.
        The default is 'interaction'.
    colorvalues : list, optional
        List of values of ``colorlevel`` to plot. The default is None, in which case all values
        of the ``colorlevel`` column are plotted.
    ax : matplotlib Axes, optional
        Axes on which to plot. The default is None, in which case a new figure and axes are created.
    colormap : list, optional
        List of colors to use for each value of ``colorvalues``. The default is None, in which case
        the default matplotlib color cycle is used.
    legend_here : bool, optional
        Whether to plot the legend on the axes. The default is True.
    title_fcn : function, optional
        Function that takes ``impulse``, ``response``, and ``interaction`` as arguments and returns
        a string to use as the title. The default is None, in which case the title is
        ``impulse`` on ``response``.
    """

    if (impulse is not None) and (response is not None):
        # Filter dataframe to the desired (impulse,response)
        dftmp = dftmp.loc[pd.IndexSlice[impulse,response,:],:].droplevel([0,1])

    # Default function
    if title_fcn is None:
        lambda response, impulse, interaction: f'{impulse} on {response}'

    # Filter dataframe to the desired colorvalues
    dftmp.reset_index(inplace=True)
    multiplot = False
    if colorlevel in dftmp.columns:
        if colorvalues is None:
            colorvalues = dftmp[colorlevel].unique()
        
        multiplot = len(colorvalues) > 1
        dftmp = dftmp.loc[dftmp[colorlevel].isin(colorvalues),:].copy()
    
    # Create a new figure if needed
    if ax is None:
        fig, ax = plt.subplots()

    # Set up the colormap
    if colormap is None:
        colormap = [c for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    # Plot one IRF
    def plot_one(df):
        line_color = to_rgb(df['color'].iloc[0])
        # Lighten up the color for shading the confidence region
        shading_color = tuple(x+(1-x)*0.5 for x in line_color)
        # Plot IRF
        h,=ax.plot(df['horizon'], df['coef'], color=line_color)
        # Plot shaded region
        ax.fill_between(df['horizon'], df['lb'], df['ub'], color=shading_color, alpha=0.2)
        return h
    
    if multiplot:
        # Multiple lines/colors on one plot
        labels = dftmp[colorlevel].unique()
        # Keep enough colors
        colormap = pd.Series(colormap[0:len(labels)], index=labels, name='color')
        # Assign a color to each color value
        dftmp = dftmp.merge(colormap, left_on=colorlevel, right_index=True)
        # Plot
        handles=list(dftmp.groupby(colorlevel).apply(plot_one))
        if legend_here:
            ax.legend(handles,labels)
    else:
        # Plot the single desired IRF in black/gray
        dftmp.loc[:,'color']='k'
        handles = [plot_one(dftmp)]
        labels = ['IRF']

    ax.axhline(0,ls='--',color='grey')
    ax.set_title(title_fcn(response,impulse,interaction))
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xticks(np.arange(dftmp['horizon'].min(),dftmp['horizon'].max()+1e-5,2))

    return handles, labels


def plot_irfs(dfirf, impulses=None, responses=None, interactions=None, \
              rows='impulse', columns='response', color='interaction', \
              colormap=None, styles=None):
    """
    Plot a grid of impulse response functions.

    Parameters
    ----------
    dfirf : DataFrame
        DataFrame containing the impulse responses. It must contain the following indices:
            - ``impulse``: name of the impulse variable
            - ``response``: name of the response variable
            - ``horizon``: forecast horizon
            - ``interaction``: optional, name of the interaction variable
        It must also contain the following columns:
            - ``coef``: impulse response function
            - ``lb``: lower bound of the confidence interval
            - ``ub``: upper bound of the confidence interval
        The first argument returned by ``estimate()`` is a suitable DataFrame.
    impulses : list, optional
        List of impulse variables to plot. The default is None, in which case all
        impulse variables in ``dfirf`` are plotted.
    responses : list, optional
        List of response variables to plot. The default is None, in which case all
        response variables in ``dfirf`` are plotted.
    interactions : list, optional
        List of interaction variables to plot. The default is None, in which case all
        interaction variables in ``dfirf`` are plotted.
    rows : str, optional
        Name of the index level to use for the rows of the grid. The default is 'impulse'.
    columns : str, optional
        Name of the index level to use for the columns of the grid. The default is 'response'.
    color : str, optional
        Name of the index level to use for the color of the lines. The default is 'interaction'.
    colormap : list, optional
        List of colors to use for each value of ``colorvalues``. The default is None, in which case
        the default matplotlib color cycle is used.
    styles : list, optional
        NOT IMPLEMENTED YET: List of line styles to use for each value of ``colorvalues``. The default is None, in which
        case the default matplotlib line styles are used.

    """

    # Get default values
    if impulses is None:
        impulses = dfirf.index.get_level_values('impulse').unique()
    if responses is None:
        responses = dfirf.index.get_level_values('response').unique()
    if interactions is None:
        if 'interaction' in dfirf.index.names:
            interactions = dfirf.index.get_level_values('interaction').unique()

    # Reset index. It's easier to filter by columns than by index levels b/c we don't know
    # exactly how many levels there are.
    dfirf = dfirf.reset_index()

    # Define filter for the desired (impulses,responses,interactions)
    levels = {'impulse':impulses, 'response':responses, 'interaction':interactions} 

    # Create grid of plots
    fig, ax = plt.subplots(nrows=len(levels[rows]), ncols=len(levels[columns]), squeeze=False)
    fig.tight_layout() 
    
    # Loop through each row and column
    for row, rowname in enumerate(levels[rows]):
        for col, colname in enumerate(levels[columns]):
            # Filter dataframe to the desired (row,column) level
            dftmp = dfirf.loc[(dfirf[rows]==rowname) & (dfirf[columns]==colname), :]

            # Define title based on the row and column names
            title_fcn = lambda res, imp, inter: f'{rowname}:{colname}'

            # Plot IRFs for each value of the color level
            handles, labels = plot_irf(dftmp, colorlevel=color, colorvalues=levels[color], \
                                        ax=ax[row,col], legend_here=False, title_fcn=title_fcn)

    # Add legend if needed
    if len(labels)>1:
        plt.figlegend(handles, labels, loc='lower center', \
                  bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        