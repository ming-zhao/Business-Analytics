import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ipywidgets import *
from IPython.display import display, HTML, IFrame
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
        sns.residplot(result.fittedvalues, result.resid, lowess=False, scatter_kws={"s": 80},
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
    if printlvl>2:
        display(stats.kstest(result.resid, 'norm'))
    return result