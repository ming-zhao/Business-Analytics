import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ipywidgets import *
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')
sns.set(color_codes=True)

dataurl = 'https://raw.githubusercontent.com/ming-zhao/Business-Analytics/master/data/regression/'

class LinRegressDisplay:
    
    def __init__(self, rand=20, num_points=36, slope=1.0):
        self.rand = rand
        self.num_points = num_points
        self.slope = slope
        self.output_widget = widgets.Output()  # will contain the plot
        self.container = widgets.VBox()  # Contains the whole app
        self.redraw_whole_plot()
        self.draw_app()

    def draw_app(self):
        """
        Draw the sliders and the output widget

        This just runs once at app startup.
        """
        self.num_points_slider = widgets.IntSlider(
            value=self.num_points,
            min=5,
            max=30,
            step=5,
            description='Number of points:',
            style = {'description_width': 'initial'}
        )
        self.num_points_slider.observe(self._on_num_points_change, ['value'])
#         self.slope_slider = widgets.FloatSlider(
#             value=self.slope,
#             min=-1,
#             max=5,
#             step=0.1,
#             description='Slope:'
#         )
#         self.slope_slider.observe(self._on_slope_change, ['value'])
        self.rand_slider = widgets.FloatSlider(
            value=self.rand,
            min=0,
            max=50,
            step=3,
            description='Randomness:', num_points=(10, 50, 5),
            style = {'description_width': 'initial'}
        )
        self.rand_slider.observe(self._on_rand_change, ['value'])
        self.container.children = [
            self.num_points_slider,
#             self.slope_slider,
            self.rand_slider ,
            self.output_widget
        ]

    def _on_num_points_change(self, _):
        """
        Called whenever the number of points slider changes.

        Updates the internal state, recomputes the random x and y and redraws the plot.
        """
        self.num_points = self.num_points_slider.value
        self.redraw_whole_plot()

    def _on_slope_change(self, _):
        """
        Called whenever the slope slider changes.

        Updates the internal state, recomputes the slope and redraws the plot.
        """
        self.slope = self.slope_slider.value
        self.redraw_slope()

    def _on_rand_change(self, _):
        self.rand = self.rand_slider.value
        self.redraw_whole_plot()

    def redraw_whole_plot(self):
        """
        Recompute x and y random variates and redraw whole plot

        Called whenever the number of points or the randomness changes.
        """
        pcent_rand = self.rand
        pcent_decimal = pcent_rand/100
        self.x = np.array([
            n*np.random.uniform(low=1-pcent_decimal, high=1+pcent_decimal) 
            for n in np.linspace(3, 9, self.num_points)
        ])
        self.y = np.array([
            n*np.random.uniform(low=1-pcent_decimal, high=1+pcent_decimal)
            for n in np.linspace(3, 9, self.num_points)
        ])
        self.redraw_slope()


    def redraw_slope(self):
        """
        Recompute slope line and redraw whole plot

        Called whenever the slope changes.
        """
        a = np.linspace(0, 9, self.num_points)
        b = [(self.slope * n) for n in a]

        self.output_widget.clear_output(wait=True)
        with self.output_widget as f:
            fig, ax = plt.subplots(1,1,figsize=(6, 4), dpi=100)
#             plt.ylim(ymax=max(self.y)+1)
#             plt.xlim(xmax=max(self.x)+1)

            plt.scatter(self.x, self.y)
#             plt.plot(a, b)
            plt.tick_params(
                axis='both',       # changes apply to the both-axis
                which='both',      # both major and minor ticks are affected
#                 bottom=False,      # ticks along the bottom edge are off
#                 top=False,         # ticks along the top edge are off
                labelbottom=False, labelleft=False) #

            plt.xlabel('Total rainfall (inch)', fontsize=10)
            plt.ylabel('Total sales', fontsize=10)
            from numpy.polynomial.polynomial import polyfit
            intercept, m = polyfit(self.x, self.y, 1)
            ax.vlines(self.x, self.y, intercept + m * self.x, label='residual')
            plt.plot(self.x, intercept + m * self.x, '-', c='orange',
                     label="$Y = {:.3f} X {} {:.3f}$".format(m, '+' if intercept>0 else '-', abs(intercept)))
            plt.legend()
            plt.show()
            
# printlvl = 0: no output or plot only return result
# printlvl = 1: no output only regplot
# printlvl = 2: no output, but plots: regplot and residual
# printlvl = 3: no output, all plots: regplot and residual and Q-Q
# printlvl = 4: summary, all plots: regplot and residual and Q-Q
# printlvl = 5: summary + ANOVA, all plots: regplot and residual and Q-Q

def analysis(df, y, x, printlvl):
    result = ols(formula=y+'~'+'+'.join(x), data=df).fit()
    if printlvl>=4:
        display(result.summary())
        print('\nstandard error of estimate:{:.5f}\n'.format(np.sqrt(result.scale)))
        
    if printlvl>=5:
        print("\nANOVA Table:\n")
        display(sm.stats.anova_lm(result, typ=2))
    
    if printlvl>=1:
        if len(x)==1:
            fig, axes = plt.subplots(1,1,figsize=(8,5))
            sns.regplot(x=x[0], y=y, data=df,
                        ci=None, 
                        line_kws={'color':'green', 
                                  'label':"$Y$"+"$={:.2f}X+{:.2f}$\n$R^2$={:.3f}".format(result.params[1],
                                                                                         result.params[0],
                                                                                         result.rsquared)},
                        ax=axes);
            axes.legend()

    if printlvl>=2:
        fig, axes = plt.subplots(1,3,figsize=(20,6))
        axes[0].relim()
        sns.residplot(result.fittedvalues, result.resid , lowess=False, scatter_kws={"s": 80},
                      line_kws={'color':'r', 'lw':1}, ax=axes[0])
        axes[0].set_title('Residual plot')
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Residuals')
        axes[1].relim()
        stats.probplot(result.resid, dist='norm', plot=axes[1])
        axes[1].set_title('Normal Q-Q plot')
        axes[2].relim()
        sns.distplot(result.resid, ax=axes[2]);
        if printlvl==2:
            fig.delaxes(axes[1])
            fig.delaxes(axes[2])
    plt.show()    
    return result  

def draw_sample(flag=1, id=1):
    size = 50
    replicate = 100
    np.random.seed(123)
    x = np.linspace(0, 5, size)
    x = np.repeat(x, replicate)
    y = 10 + 3*x+ 4*np.random.normal(size=size*replicate)
    
    sample_size = 20
    np.random.seed(id+111)
    fig, axes = plt.subplots(1, 1, figsize = (14,8))

    df_pop = pd.DataFrame({'X':x,'Y':y})

    slope, intercept, r_value, p_value, std_err = stats.linregress(df_pop.X,df_pop.Y)
    sns.regplot(x=df_pop.X, y=df_pop.Y, data=df_pop, marker='o', label='population (assumed)',
                    line_kws={'color':'maroon', 
                              'label':"$\overline{Y}$"+"$={:.2f}X+{:.2f}$".format(slope, intercept)},
                    scatter_kws={'s':2})

    if flag==1:
        if id==1:
            df_sample = df_pop.loc[
                ((df_pop['Y'] >= 16 + 3 * df_pop['X']) & (df_pop['X'] <= 2.5))
                |((df_pop['Y'] <= 4 + 4 * df_pop['X']) & (df_pop['X'] > 2.5))
            ].sample(n=sample_size)
        if id==2:
            df_sample = df_pop.loc[
                ((df_pop['Y'] <= 2 + 3 * df_pop['X']) & (df_pop['X'] <= 2.5))
                |((df_pop['Y'] >= 15 + 4 * df_pop['X']) & (df_pop['X'] > 2.5))
            ].sample(n=sample_size)
        if id==3:
            df_sample = df_pop.sample(n=sample_size)

        df_sample = df_sample.groupby('X').first()
        df_sample.reset_index(level=0, inplace=True)

        est_slope, est_intercept, r_value, p_value, std_err = stats.linregress(df_sample.X,df_sample.Y)
        sns.regplot(x=df_sample.X, y=df_sample.Y, data=df_sample, marker='o', label='sample', ci=False,
                        line_kws={'color':'orange',
                                  'label':"$Y={:.2f}X+{:.2f}$".format(est_slope, est_intercept)},
                        scatter_kws={'s':35, 'color':'green'})

        axes.vlines(df_sample.X-.02,df_sample.Y - .2*np.sign(df_sample.Y-slope*df_sample.X-intercept),
                    slope*df_sample.X+intercept, colors='darkred', label='error')
        axes.vlines(df_sample.X+.03,df_sample.Y - .2*np.sign(df_sample.Y-est_slope*df_sample.X+est_intercept),
                    est_slope*df_sample.X+est_intercept, colors='darkorange', label='residual')

        plt.title('estimated regression line with different samples')
    else:
        plt.title('population regression line joining means')
        
    plt.legend()
    plt.show()
    
def draw_qq(dist):
    np.random.seed(123)
    fig, ax = plt.subplots(1, 2, figsize = (16,5))
    if dist=='normal':
        residuals = np.random.normal(loc=20, scale=6, size=1000)   
    if dist=='student_t':
        residuals = np.random.standard_t(df=4, size=1000)
    if dist=='uniform':
        residuals = np.random.uniform(0,20,size=1000)
    if dist=='triangular':
        residuals = np.random.triangular(-2, 0, 9, 1000)

    stats.probplot(residuals, dist="norm", plot=ax[0])
    sns.distplot(residuals, ax=ax[1]);
    plt.show()
    
def plot_hist(column):
    fig, axes = plt.subplots(1, 1, figsize = (12,6))
    df_salary = pd.read_csv(dataurl+'salaries.csv', header=0, index_col='employee')
    ax = sns.countplot(x=column, data=df_salary, ax=axes)     
    total = len(df_salary[column])
    nbars = len(ax.patches)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        shift=.12
        if nbars<=2:
            shift=.1
        elif nbars>=10 and nbars<20:
            shift=.25
        elif nbars>=20:
            shift=.5
        x = p.get_x() + p.get_width()/2 - shift
        y = p.get_y() + p.get_height() + .5
        ax.annotate(percentage, (x, y))      
    plt.show()
    
def GL_Ftest(df, y, fullX, reducedX, alpha, typ=2):
    result = analysis(df, y, fullX, printlvl=0)
    if len(reducedX)>0:
        sse_f = sm.stats.anova_lm(result, typ=typ).sum_sq['Residual']
        df_f  = sm.stats.anova_lm(result, typ=typ).df['Residual']
        result = analysis(df, y, reducedX, printlvl=0)
        sse_r = sm.stats.anova_lm(result, typ=typ).sum_sq['Residual']
        df_r  = sm.stats.anova_lm(result, typ=typ).df['Residual']
        F = (sse_r-sse_f)/(df_r - df_f) / (sse_f/df_f)
        p = 1 - stats.f.cdf(F, df_r - df_f, df_f)
    else:
        F = result.fvalue
        p = result.f_pvalue
    if p<alpha:
        print('The F-value is {:.3f} and p-value is {:.3f}. Reject null hypothesis.\
              \nAt least one variable in {} has nonzero slope.'.format(F, p, set(fullX)-set(reducedX)))
    else:
        print('The F-value is {:.3f} and p-value is {:.3f}. Don\'t Reject null hypothesis.\
              \nAll variables in {} have zero slope.'.format(F, p, set(fullX)-set(reducedX)))
    return p