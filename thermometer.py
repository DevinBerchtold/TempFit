import math
import numpy as np
import pandas as pd
from scipy.interpolate import *
from scipy.optimize import *
from scipy.signal import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
import re
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'Covariance of the parameters could not be estimated')



######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ###
##        #######  ##    ##  ######     ##    ####  #######  ##    ##

def r_squared(f, x, y):
    x = np.array(x)
    y = np.array(y)
    r = y-f(x)
    sr = np.sum(r**2)
    st = np.sum((y-np.mean(y))**2)
    return 1 - (sr / st)

def smooth_diff(y, dx, w=75, p=1):
    dydx = np.gradient(y,dx)
    return savgol_filter(dydx,window_length=w,polyorder=p, mode='interp')

class Function:
    def tx_j(x,a,b,c,d,e): # a*x**3 + b*x**2 + c*x + d*(1+x)**-1 + e
        return np.array([
            a*3*x**2 + b*2*x + c - d*(1+x)**-2,
            x**3, x**2, x, (1+x)**-1, 1
        ])
    def tx_h(x,a,b,c,d,e): # a*x**3 + b*x**2 + c*x + d*(1+x)**-1 + e
        return np.diag([
            a*6*x + b*2 + d*2*(1+x)**-3,
            1, 1, 1, 1, 0
        ])
    functions = { # f: function, f1: derivative, i: initial
        'p1': {
            'f': lambda x,a,b: a*x + b,
            'f1': lambda x,a,b: a,
            'f2': lambda x,a,b: 0.0,
            's': f'y=ax+b',
            'i': [1e2, -1e2],
            'b': [(0.0, None), (None, None)],
            'j': None,
            'h': None
        },
        'p2': {
            'f': lambda x,a,b,c: a*x**2 + b*x + c,
            'f1': lambda x,a,b,c: a*2*x + b,
            'f2': lambda x,a,b,c: a*2,
            's': f'y=ax²+bx+c',
            'i': [1e-2, -1e2, 1e4],
            'b': [(None, None), (None, None), (None, None)],
            'j': None,
            'h': None
        },
        'p3': {
            'f': lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d,
            'f1': lambda x,a,b,c,d: a*3*x**2 + b*2*x + c,
            'f2': lambda x,a,b,c,d: a*6*x + b*2,
            's': f'y=ax³+bx²+cx+d',
            'i': [-1e-3, 1e1, -1e3, 1e6],
            'b': [(0.0, None), (None, None), (None, None), (None, None)],
            'j': None,
            'h': None
        },
        'tx': {
            'f': lambda x,a,b,c,d,e: a*x**3 + b*x**2 + c*x + d*(1+x)**-1 + e,
            'f1': lambda x,a,b,c,d,e: a*3*x**2 + b*2*x + c - d*(1+x)**-2,
            'f2': lambda x,a,b,c,d,e: a*6*x + b*2 + d*2*(1+x)**-3,
            's': f'y=ax³+bx²+cx+dx⁻¹+e',
            'i': [-1e-3, 1e1, -1e3, -1e2, 1e6],
            'b': [(0.0, None), (None, None), (None, None), (None, 0.0), (None, None)],
            'j': tx_j,
            'h': tx_h
        }
    }

    def __init__(self, function: str):
        self.t = function
        self.f = Function.functions[function]['f']
        self.f1 = Function.functions[function]['f1']
        self.f2 = Function.functions[function]['f2']
        self.s = Function.functions[function]['s']
        self.p = Function.functions[function]['i'] # initial guess for p
        self.b = Function.functions[function]['b']
        self.j = Function.functions[function]['j']
        self.h = Function.functions[function]['h']

    def fx(self, parm=None, sub=0.0): # return f as a function of x
        if parm is not None:
            return lambda x: self.f(x, *parm) - sub
        else:
            return lambda x: self.f(x, *self.p) - sub

    def f1x(self, parm=None, sub=0.0): # return f1 as a function of x
        if parm is not None:
            return lambda x: self.f1(x, *parm) - sub
        else:
            return lambda x: self.f1(x, *self.p) - sub

    def f2x(self, parm=None, sub=0.0): # return f2 as a function of x
        if parm is not None:
            return lambda x: self.f2(x, *parm) - sub
        else:
            return lambda x: self.f2(x, *self.p) - sub

    def curve_fit(self, xx, yy):
        """Perform an unconstrained curve fit of `self.f` to the data"""
        self.p, _ = curve_fit(self.f, xx, yy, self.p)
        return self.p

    def minimize(self, xx, yy, slope, constrain=(True, True, True), bound=None):
        # def positive_d(p, x, y, f):
        #     ret = 0.0
        #     r = np.arange(x[0], x[-1] + (x[-1]-x[0]),1.0)
        #     for i in r:
        #         if f(i,p) < 0.0:
        #             ret += f(i,p)
        #     return ret

        def stall_inflection(p, f, lx, st): # f''(x) = 0 when f(x) = stall
            ix, _ = f.solve(st, x0=lx, parm=p)
            return f.f(ix,*p)+f.f2(ix,*p)-st

        constraints = [
            {
                'type':'eq',
                'fun': lambda p,f,lx,ly: ly - f(lx,*p),
                'args': (self.f, xx[-1], yy[-1])
            },
            {
                'type':'eq',
                'fun': lambda p,f,lx,s: s - f(lx,*p),
                'args': (self.f1, xx[-1], slope)
            },
            {
                'type':'eq', # inflection point at stall temperature
                'fun': stall_inflection,
                'args': (self, xx[-1], 155.0)
            },
            # {
            #     'type':'ineq', # derivative positive
            #     'fun': positive_d,
            #     'args': (self.fit_x, self.fit_y, func['f1p'])
            # },
        ]
        # filter based on input
        constraints = [c for c, b in zip(constraints, constrain) if b == True]

        obj = lambda p,x,y,f: np.sum((f(x, *p) - y) ** 2)
        # options={'maxiter': 100}
        res = minimize(obj, self.p, args=(xx, yy, self.f), constraints=constraints, bounds=self.b if bound else None)
        self.p = res.x
        return self.p

    def solve(self, y, x0=1, parm=None):
        """Return the x value where `self.f` is equal to `y`"""
        roots, _, ier, _ = fsolve(self.fx(sub=y, parm=parm), x0, full_output=True)
        # roots, _, ier, _ = fsolve(self.fx(sub=y, parm=parm), x0, fprime=lambda x: [self.f1x(parm=parm)(x)], full_output=True)
        return roots[0], ier == 1
        # sol = root_scalar(self.fx(sub=y, parm=parm), x0=x0, fprime=self.f1x(parm=parm), fprime2=self.f2x(parm=parm), method='halley')
        # return sol.root, sol.converged

    def string(self):
        """Return a string representation of the function and coefficients"""
        pn = [f'{p:.2g}' for p in self.p]
        pe = [re.sub(r'e(-?)\+?0?',r'e\1',f'{p:.1e}') for p in self.p]
        pm = [(a if len(a)<len(b) else b) for a, b in zip(pn, pe)]
        if pm[0][0] == '+':
            pm[0] = pm[0][1:]
        # return f'y={pm[0]}⋅x³{pm[1]}⋅x²{pm[2]}⋅x{pm[3]}⋅x⁻¹{pm[4]}'
        return f"{self.t}(x, {', '.join(pm)})"



######## ##     ## ######## ########  ##     ##  #######  ##     ## ######## ######## ######## ########
   ##    ##     ## ##       ##     ## ###   ### ##     ## ###   ### ##          ##    ##       ##     ##
   ##    ##     ## ##       ##     ## #### #### ##     ## #### #### ##          ##    ##       ##     ##
   ##    ######### ######   ########  ## ### ## ##     ## ## ### ## ######      ##    ######   ########
   ##    ##     ## ##       ##   ##   ##     ## ##     ## ##     ## ##          ##    ##       ##   ##
   ##    ##     ## ##       ##    ##  ##     ## ##     ## ##     ## ##          ##    ##       ##    ##
   ##    ##     ## ######## ##     ## ##     ##  #######  ##     ## ########    ##    ######## ##     ##

def format_minutes(minutes):
    h = math.floor(abs(minutes)/60)
    m = math.floor(abs(minutes)%60)
    if minutes >= 0:
        return f'{h}:{m:02}'
    else:
        return f'-{h}:{m:02}'

class Thermometer:
    # Dark Mode Style
    dark_style = {
        # 'axes.facecolor': '00000080',
        'axes.facecolor': '000000',
        'axes.prop_cycle': "cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])",
        'figure.facecolor': '1e1e1e', # vscode gray
        'figure.edgecolor': '1e1e1e', # vscode gray
        # 'savefig.facecolor': '0D1117', # github gray
        # 'savefig.edgecolor': '0D1117', # github gray
        'savefig.facecolor': '00000000',
        'savefig.edgecolor': '00000000',
        'lines.color': 'ffffff80',
        'figure.figsize': '6, 4',
        'figure.dpi': '100', # 640x360 (16:9)
    }
    # Light Mode
    light_style = {
        'savefig.facecolor': 'ffffff00',    # transparent
        'savefig.edgecolor': 'ffffff00',    # transparent
        'lines.color': '00000080',
        'figure.figsize': '6, 4',           # figure size in inches
        'figure.dpi': '100',                # figure dots per inch
    }
    
    def __init__(self, filename=None, temp=None, style='dark'):
        """Create a `Thermometer`, reading data from `filename` or initializing the first row with `temp`."""
        self.style = style
        if style == 'dark':
            plt.style.use(['dark_background', Thermometer.dark_style])
        elif style == 'light':
            plt.style.use(['default', Thermometer.light_style])
        if filename == None:
            self.filename = f'cooks/manual_{dt.date.today():%Y%m%d}.csv'
            t = np.datetime64('now')
            self.start = self.end = t
            data = {
                'Temperature': [float(temp)],
                'Minutes': [0.0]
            }
            self.dataframe = pd.DataFrame(data,index=[t])
            self.dataframe.index.name = 'Time'
        else:
            self.filename = filename
            # Excel Format:
            # self.dataframe = pd.read_csv(f,parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])
            # Inkbird Format:
            self.dataframe = pd.read_csv(filename,delimiter='\t',encoding='utf_16',parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])

            self.start = self.dataframe.index[0]
            self.end = self.dataframe.index[-1]
            
            # Number of minutes since beginning of period
            t0 = self.dataframe.index.values[0]
            self.dataframe['Minutes'] = [(t-t0)/np.timedelta64(1, 'm') for t in self.dataframe.index.values]
            self.trim()

    def add(self, temp):
        """Add the temperature `temp` to the dataframe with the current time as the index."""
        t = np.datetime64('now')
        data = [temp, (t-self.start)/np.timedelta64(1, 'm')]
        self.end = t
        # print(self.dataframe)
        self.dataframe.loc[t]=data
        # print(self.dataframe)

    def trim(self):
        """Remove datapoints before the lowest temperature and after the highest temperature."""
        imax = self.dataframe.idxmax()['Temperature']
        imin = self.dataframe.idxmin()['Temperature']
        if imin < imax:
            self.dataframe = self.dataframe[imin:imax]
            self.start = self.dataframe.index[0]
            self.end = self.dataframe.index[-1]

    def diff_plot(self):
        """Plot all temperature vs time data in the dataframe with differentiation."""
        x = self.dataframe['Minutes'].values
        y = self.dataframe['Temperature'].values
        itp = interp1d(x, y, fill_value='extrapolate', kind='linear')
        x_uniform = np.linspace(x[0],x[-1],len(x))
        y_uniform = itp(x_uniform)
        dx = (x[-1] - x[0]) / len(x)
        dydx = smooth_diff(y_uniform, dx)
        dydx2 = smooth_diff(dydx, dx)

        # roots = fsolve(itp2,[x[0],x[-1]])
        roots = [float(x) for x, y in enumerate(dydx2) if y==0.0] # get zeroes where datapoint is exactly zero
        for x, y in enumerate(dydx2[:-1]):
            x1 = float(x)
            y1 = float(y)
            x2 = float(x+1)
            y2 = float(dydx2[x+1])
            if (y1 > 0.0 and y2 < 0.0) or (y1 < 0.0 and y2 > 0.0):
                roots.append( (x1 - y1 * (y2-y1)/(x2-x1)) * dx) # add a zero in between the two samples
        for r in roots:
            print(f'root ({r}, {itp(r)})')

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('Derivatives (dT/dt)')
        ax1.plot(x_uniform, dydx/20, color='C0')
        ax1.plot(x_uniform, dydx2, color='C2')
        ax1.axhline(y=0.0)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Temps (F)')
        ax2.plot(x_uniform, y_uniform, color='C1')
        fig.show()

    def estimate(self, done_temp, time=180, fit_func='tx', fit_start=None, fit_end=None, constrain=True, bound=False, print=False, plot=False, filename=None):
        """Estimate when thermometer will reach desired temperature. 
        
        Args:
            done_temp: The temperature for which we want to find the time.
            time: The time in minutes before now to be used for estimating.
            fit_func: Estimated function with parameters substituted
            fit_start: Start time of the interval used for estimating.
            fit_end: End time of the interval used for estimating.
            print: Whether ETA string should be printed.
            plot: Whether fit chart should be displayed.
        """
        self.fit_done = done_temp
        if fit_end==None:
            self.fit_end = self.end
        else:
            self.fit_end = fit_end

        if fit_start==None:
            self.fit_start = self.fit_end - np.timedelta64(time, 'm')
        else:
            self.fit_start = fit_start
            
        # Get subset of dataframe with samples from `fit_start` to `fit_end`
        filtered = self.dataframe.loc[self.fit_start:self.fit_end]

        # Choose the input and output variables
        self.fit_x, self.fit_y = filtered["Minutes"].values, filtered["Temperature"].values
        last_time = self.fit_x[-1]
        last_temp = self.fit_y[-1]

        func = Function(fit_func)

        self.func_string = func.s
        # unconst_popt, _ = curve_fit(func['f'], self.fit_x, self.fit_y, func['i'])
        unconst_popt = func.curve_fit(self.fit_x, self.fit_y)

        n = 25
        itp = interp1d(self.fit_x, self.fit_y, kind='linear')

        x_uniform = np.linspace(self.fit_x[0],self.fit_x[-1],len(self.fit_x))
        y_uniform = itp(x_uniform)

        dx = (self.fit_x[-1] - self.fit_x[0]) / len(self.fit_x)
        dydx = smooth_diff(y_uniform, dx, w=n)
        lx = x_uniform[-n//2] # last x used for fit
        ly = itp(lx)
        slope = dydx[-n//2]
        intercept = ly-slope*lx
        self.linear_func = lambda x: slope*x+intercept # linear approximation at end
        self.unconst_func = None
        if constrain != False:
            # self.unconst_func = lambda x: func.fp(x, unconst_popt) # fit function without constraints
            self.unconst_func = func.fx(parm=unconst_popt)

            if constrain == True:
                constrain = (True, True, True)

            popt = func.minimize(self.fit_x, self.fit_y, slope, constrain, bound)
        else:
            popt = unconst_popt

        self.fit_func = func.fx(parm=popt)
        root, conv = func.solve(done_temp, x0=self.fit_x[-1])
        
        if not conv:
            root, conv = func.solve(done_temp, x0=self.fit_x[-1], parm=unconst_popt)
        if conv: # if we found a solution
            self.fit_eta = root
        else: # no solution, use linear
            self.fit_eta = (done_temp-intercept)/slope

        # Calculate time remaining and ETA
        time_left = self.fit_eta-last_time

        # Format and print it out
        if conv:
            # calculate goodness of fit
            rsqr = r_squared(self.fit_func, self.fit_x, self.fit_y)
            eta = self.fit_start+np.timedelta64(round(time_left),'m')
            self.fit_string = f"NOW: {self.fit_end} {last_temp}° --> ETA: {eta} {done_temp}° (in {format_minutes(time_left)}) (R2 {rsqr:.2%})"
        else:
            self.fit_string = f"NOW: {self.fit_end} {last_temp}° --> ETA: ???                 {done_temp}° (in ??:??)"
        if print:
            print(self.fit_string)

        if plot:
            self.plot_estimation(close=True)
            if filename:
                plt.savefig(filename)
            else:
                plt.show()

        if self.fit_eta < 0:
            return 0.0
        else:
            return self.fit_eta

    def plot_summary(self, done, interval=130, step=5, filename=None):
        """Run `estimate()` at multiple times throughout a cook and show the ETAs in one static plot.
        
        Args:
            done: The 'done temp' to estimate for.
            step: The step between every evaluation of `estimate()`
        """
        # Graph how estimation changed over time
        interval = np.timedelta64(interval, 'm')
        times = np.arange(-270,0,float(step))

        parms = [
            # {'fit_func': 'p2', 'constrain': (False, False, False), 'bound': False},
            # {'fit_func': 'p3', 'constrain': (False, False, False), 'bound': False},
            {'fit_func': 'tx', 'constrain': (False, False, False), 'bound': False},
            {'fit_func': 'tx', 'constrain': (True, True, False), 'bound': False},
            {'fit_func': 'tx', 'constrain': (True, True, True), 'bound': False},
        ]

        etas = [[] for _ in parms]
        temps = []
        for time in times:
            t = dt.timedelta(minutes=time)
            temps.append(self.dataframe.iloc[self.dataframe.index.get_loc(self.end+t, method='nearest')]['Temperature'])
            for n, parm in enumerate(parms):
                eta = self.estimate(done, fit_start=self.end+t-interval, fit_end=self.end+t, **parm)
                etas[n].append(eta)
        correct_eta = etas[-1][-1]
            
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('Temps (F)')
        ax1.plot(times, temps, color='tab:gray')
        ax1.set_xlim([-270.0, 0.0])
        # ax1.tick_params(axis ='y', labelcolor='tab:gray')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('ETAs (m)')
        for n, parm in enumerate(parms):
            func = parm['fit_func']
            const = parm['constrain']
            string = func+' c'+str(const[0])[0]+str(const[1])[0]+str(const[2])[0]
            i = -len(times)//2
            # rsqr = r_squared(lambda x: correct_eta, times[i:], etas[n][i:])
            l = [abs(correct_eta-y) for y in etas[n][i:]]
            rsqr = 1.0 - sum(l)/(len(l)*correct_eta)
            ax2.plot(times, etas[n], '.', label=f"{string} ({rsqr:.2%})")
        # ax2.tick_params(axis ='y', labelcolor='C0')
        ax2.set_ylim([correct_eta-120.0, correct_eta+120.0])
        ax2.axhline(y=correct_eta)
        ax2.legend(loc='upper left')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_animate(self, done, interval, time, step, fps=10, dpi=100, fit_func='tx', constrain=True, filename=None):
        """Create an animation of multiple estimations on a temperature dataset throughout time.
        
        Args:
            done: The temperature for which we want to find the time.
            interval: The time (minutes) in the past to use for estimation at east point.
            time: The time (minutes) over which the interval shifts.
            step: Change (minutes) of each frame/estimation.
            fps: FPS of output.
            dpi: DPI of output.
            fit_func: Template function to use during the fit. ('p1', 'p2', 'p3', or 'tx')
            constrain: Whether function is constrained during the fit.
        """
        # plt.close('all')
        fig, ax = plt.subplots()

        # Graph how estimation changed over time
        times = []
        for t in np.arange(-time,0,step,dtype=float):
            a = self.end+dt.timedelta(minutes=t)-dt.timedelta(minutes=interval)
            b = self.end+dt.timedelta(minutes=t)
            times.append((a, b))

        def anim(i):
            fig.clear()
            self.estimate(done, fit_start=times[i][0], fit_end=times[i][1], fit_func=fit_func, constrain=constrain)
            self.plot_estimation()
        ani = FuncAnimation(fig, anim, frames=len(times), interval=1000/fps, blit=False)

        if filename:
            if filename[-4:] == '.gif':
                ani.save(filename, writer='pillow', fps=fps, dpi=dpi, savefig_kwargs={'facecolor': '#404040'})
            else:
                ani.save(filename, writer='pillow', fps=fps, dpi=dpi)
            
        else:
            plt.show()

    def plot_estimation(self, close=False):
        """Plot the measured data and the estimation functions. Called by `estimate()` to show single plots and by `plot_animate()` for each frame of an animation. Requires estimation instance variables to have been set by `estimate()`"""
        if close:
            plt.close('all')
        plt.title(f'Temperature vs Time ({self.func_string})')
        plt.xlabel('Hours')
        plt.ylabel('Degrees Fahrenheit')
        
        # Plot actual data
        all_x, all_y = self.dataframe["Minutes"].values, self.dataframe["Temperature"].values
        maxx = max(self.fit_x[-1], all_x[-1])
        minx = self.fit_x[0]
        # plt.scatter([i/60.0 for i in all_x], all_y, marker='o', color='#808080')
        # plt.scatter([i/60.0 for i in self.fit_x], self.fit_y, marker='o', label=f'Measured ({format_minutes(minx)}-{format_minutes(self.fit_x[-1])})')
        plt.plot([i/60.0 for i in all_x], all_y, '-', linewidth=6, solid_capstyle='round', color='#80808080')
        plt.plot([i/60.0 for i in self.fit_x], self.fit_y, '-', linewidth=6, solid_capstyle='round', color='#20A0FF80', label=f'Measured ({format_minutes(minx)}-{format_minutes(self.fit_x[-1])})')

        # Plot the fitted function lines
        lastx = max(maxx, self.fit_eta)+min(maxx-minx,60) # from beginning of prediction data, to end of measured data plus one hour
        x_line = np.arange(float(minx), float(lastx), 1)
        y_line = self.fit_func(x_line)
        if self.linear_func != None:
            y_line1 = self.linear_func(x_line)
        if self.linear_func != None:
            y_line2 = self.unconst_func(x_line)

        x_samples, y_samples = zip(*[(x, y) for x, y in zip(all_x, all_y) if x>=minx and x<=lastx])
        x_hours = [i/60.0 for i in x_line]
        
        if self.linear_func != None:
            plt.plot(x_hours, y_line1, '--', color='C3', label=f"Linear ({r_squared(self.linear_func,x_samples,y_samples):.2%} fit)")
            plt.plot(x_hours, y_line2, '--', color='C1', label=f"Basic Fit ({r_squared(self.unconst_func,x_samples,y_samples):.2%} fit)")
            plt.plot(x_hours, y_line, '--', color='C2', label=f"Constrained ({r_squared(self.fit_func,x_samples,y_samples):.2%} fit)")
        else:
            plt.plot(x_hours, y_line, '--', color='C1', label=f"Estimated ({r_squared(self.fit_func,x_samples,y_samples):.2%} fit)")
        # Plot ETA estimation
        plt.axvline(x=self.fit_eta/60.0, label=f'Prediction ({self.fit_done}° at {format_minutes(self.fit_eta)})')
        plt.axhline(y=self.fit_done)
        plt.legend(loc='upper left') # Labels
        # plt.legend(loc='lower right') # Labels
        plt.xlim([all_x[0]/60.0, 1.2*all_x[-1]/60.0])
        plt.ylim([all_y[0], 1.2*all_y[-1]])
