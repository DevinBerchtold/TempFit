import math
import numpy as np
import pandas as pd
from scipy.optimize import *
from matplotlib import pyplot

def p2(x, a, b, c):
    return (a*(x**2))+(b*x)+c

def p3(x, a, b, c, d):
    return (a*(x**3))+(b*(x**2))+(c*x)+d

def tx(x, a, b, c, d, e):
    return (a*(x**3))+(b*(x**2))+(c*x)+(d*((1+x)**-1))+e

def format_minutes(minutes):
    h = math.floor(abs(minutes)/60)
    m = math.floor(abs(minutes)%60)
    if minutes >= 0:
        return f'{h}:{m:02}'
    else:
        return f'-{h}:{m:02}'

class Thermometer:
    def __init__(self, filename=None):
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

    def trim(self):
        imax = self.dataframe.idxmax()['Temperature']
        imin = self.dataframe.idxmin()['Temperature']
        if imin < imax:
            self.dataframe = self.dataframe[imin:imax]
            self.start = self.dataframe.index[0]
            self.end = self.dataframe.index[-1]

    def plot(self):
        self.dataframe.plot(y='Temperature')

    def func(self, done_temp, x, y, f, init):
        popt, _ = curve_fit(f, x, y, init)
        # print(f'coeffs: {popt}')
        new_f = lambda x: f(x, *popt)
        new_f_sub = lambda x: f(x, *popt)-done_temp
        roots = fsolve(new_f_sub, x[-1])
        # print(f'roots: {roots}')
        best = min(roots, key=(lambda i : abs(i - x[-1]) ) ) # Solution that is shortest time in the future
        return best, new_f, popt

    def estimate(self, done_temp, time=180, fit_func='tx', start=None, end=None, display=True, plot=False, debug=False):
        if debug:
            print("== D E B U G   O N ==")
            print(self.dataframe.__repr__())
            print(type(self.dataframe))
            print(start)
            print(end)

        if end==None:
            end = self.end
        if start==None:
            start = end - np.timedelta64(time, 'm')
        # Get subset of dataframe with samples from start to end
        filtered = self.dataframe.loc[start:end]

        # Choose the input and output variables
        x, y = filtered["Minutes"].values, filtered["Temperature"].values
        last_time = x[-1]
        last_temp = y[-1]
        if debug:
            print(f"X: {x}")
            print(f"Y: {y}")

        # roots, function = self.poly(done_temp, x, y, order)
        if fit_func == 'p2':
            fit_eta, function, popt = self.func(done_temp, x, y, p2, [1e-2, -1e2, 1e4])
        elif fit_func == 'p3':
            fit_eta, function, popt = self.func(done_temp, x, y, p3, [-1e-3, 1e1, -1e3, 1e6])
        elif fit_func == 'tx':
            fit_eta, function, popt = self.func(done_temp, x, y, tx, [-1e-3, 1e1, -1e3, -1e2, 1e6])

        if fit_eta == None: # We didn't get a valid answer
            if display:
                if debug:
                    print('== O U T P U T ==')
                print(f"NOW: {end} {last_temp}° --> ETA: ???                {done_temp}° (in ??:??)")
            return '??:??'

        # Calculate time remaining and ETA
        time_left = fit_eta-last_time

        now = end
        eta = start+np.timedelta64(round(time_left),'m')

        # Format and print it out
        if display:
            if debug:
                print('== O U T P U T ==')
            print(f"NOW: {now} {last_temp}° --> ETA: {eta} {done_temp}° (in {format_minutes(time_left)})")

        if plot:
            # Plot actual data
            pyplot.scatter([i/60.0 for i in x], y, label='Measured temperature')
            # Plot the fitted function line
            x_line = np.arange(min(x), max(max(x), fit_eta)+60, 1)
            y_line = function(x_line)
            x_line = [i/60.0 for i in x_line]
            pyplot.xlabel('Hours')
            pyplot.ylabel('Degrees Fahrenheit')
            coeffs = [f'{o:.2g}' for o in popt]
            pyplot.plot(x_line, y_line, '--', color='red', label=f"{fit_func}(x, {', '.join(coeffs)})")
            # Plot ETA estimation
            pyplot.axvline(x=fit_eta/60.0, label=f'{done_temp}° at {format_minutes(fit_eta)}')
            pyplot.axhline(y=done_temp)
            pyplot.legend() # Labels
            pyplot.show() # Show the chart

        return fit_eta
