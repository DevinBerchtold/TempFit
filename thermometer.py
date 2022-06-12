import math
import numpy as np
import pandas as pd
from scipy.optimize import *
from matplotlib import pyplot

def p2(x, a, b, c):
    return (a*(x**2))+(b*x)+c

def p3(x, a, b, c, d):
    return (a*(x**3))+(b*(x**2))+(c*x)+d

def t2(x, a, b, c, d, e, f):
    return (a*(x**2))+(b*x)+(c*((x+e)**2))+(d*(x+e))+f

class Thermometer:
    def __init__(self, filename=None):
        self.filename = filename

        # Excel Format:
        #self.dataframe = pd.read_csv(f,parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])
        # Inkbird Format:
        self.dataframe = pd.read_csv(filename,delimiter='\t',encoding='utf_16',parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])

        self.start = self.dataframe.index[0]
        self.end = self.dataframe.index[-1]
        
        # Number of minutes since beginning of period
        t0 = self.dataframe.index.values[0]
        self.dataframe['Minutes'] = [(t-t0)/np.timedelta64(1, 'm') for t in self.dataframe.index.values]

    def plot(self):
        self.dataframe.plot(y='Temperature')

    def poly(self, done_temp, x, y, order=3):
        # Use numpy to fit polynomial function to data
        coeffs = np.polyfit(x, y, order)
        # print(f'coeffs: {coeffs}')
        poly_func = np.poly1d(coeffs)
        temp_poly = np.poly1d([done_temp])
        subbed = np.polysub(poly_func, temp_poly)

        real_roots = [i.real for i in subbed.r if i.imag == 0]
        if real_roots == []:
            # If no roots of polynomial are real... that's a problem. Output a ?? guess
            return None, None
        # print(f'roots: {real_roots}')

        return real_roots, poly_func

    def func(self, done_temp, x, y, f, init):
        popt, pcov = curve_fit(f, x, y, init)
        # print(f'coeffs: {popt}')
        new_f = lambda x: f(x, *popt)
        new_f_sub = lambda x: f(x, *popt)-done_temp
        roots = fsolve(new_f_sub, [x[-1]])
        # print(f'roots: {roots}')

        return roots, new_f, popt

    def estimate(self, done_temp, time=180, fit_func='p3', start=None, end=None, display=True, plot=False, debug=False):
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
            roots, function, popt = self.func(done_temp, x, y, p2, [1e-2, -1e2, 1e4])
        elif fit_func == 'p3':
            roots, function, popt = self.func(done_temp, x, y, p3, [-1e-3, 1e1, -1e3, 1e6])
        elif fit_func == 't2':
            roots, function, popt = self.func(done_temp, x, y, t2, [1e-2, -1e2, 1e-2, -1e2, -1e3, 1e2])

        if roots == None: # We didn't get a valid answer
            if display:
                if debug:
                    print('== O U T P U T ==')
                print(f"NOW: {end} {last_temp}° --> ETA: ???                 {done_temp}° (in ??:??)")
            return '??:??'

        # Find the real root which is the shortest time in the future:
        best_root = min(roots, key=(lambda i : abs(i - last_time) ) )

        # Calculate time remaining and ETA
        time_left = best_root-last_time

        now = end
        eta = start+np.timedelta64(round(time_left),'m')

        hours = math.floor(time_left/60)
        minutes = math.floor(time_left%60)
        remaining = f'{hours}:{minutes:02}'
        if time_left < 0:
            hours = hours+12
            minutes = 60-minutes
            remaining = f'-{hours}:{minutes:02}'

        # Format and print it out
        if display:
            if debug:
                print('== O U T P U T ==')
            print(f"NOW: {now} {last_temp}° --> ETA: {eta} {done_temp}° (in {remaining})")

        if plot:
            # Plot actual data
            pyplot.scatter(x, y, label='Measured temperature')
            # Plot the fitted function line
            x_line = np.arange(min(x), max(x)+100, 1)
            y_line = function(x_line)
            coeffs = [f'{o:.2g}' for o in popt]
            pyplot.plot(x_line, y_line, '--', color='red', label=f"{fit_func}(x, *({', '.join(coeffs)}) )")
            # Plot ETA estimation
            pyplot.axvline(x=best_root, label=f'{done_temp}° {best_root:.1f}m')
            pyplot.axhline(y=done_temp)
            pyplot.legend() # Labels
            pyplot.show() # Show the chart

        return remaining
