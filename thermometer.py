import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot

class Thermometer:
    def __init__(self, filename=None):
        # Excel Format:
        #self.dataframe = pd.read_csv(f,parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])
        # Inkbird Format:
        self.filename = filename
        self.dataframe = pd.read_csv(filename,delimiter='\t',encoding='utf_16',parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1])

        self.start = self.dataframe.index[0]
        self.end = self.dataframe.index[-1]
        
        # Number of minutes since beginning of period
        t0 = self.dataframe.index.values[0]
        self.dataframe['Minutes'] = [(t-t0)/np.timedelta64(1, 'm') for t in self.dataframe.index.values]

    def plot(self):
        self.dataframe.plot(y='Temperature')

    def estimate(self, done_temp, time=180, order=3, start=None, end=None, display=True, plot=False, debug=False):
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
        if debug:
            print(f"X: {x}")
            print(f"Y: {y}")

        # Use numpy to fit polynomial function to data
        coeffs = np.polyfit(x, y, order)
        poly_func = np.poly1d(coeffs)
        temp_poly = np.poly1d([done_temp])
        subbed = np.polysub(poly_func, temp_poly)
        if debug:
            print(f'{poly_func}\n-\n{temp_poly}\n=\n{subbed}')

        last_temp = y[-1]
        last_time = x[-1]

        real_roots = [i for i in subbed.r if i.imag == 0]
        if real_roots == []:
            # If no roots of polynomial are real... that's a problem. Output a ?? guess
            if display:
                if debug:
                    print('== O U T P U T ==')
                print(f"NOW: {end} {last_temp}° --> ETA: ???                 {done_temp}° (in ??:??)")
            return '??:??'

        # If roots are real, we might have a good solution

        # Find the real root which is the shortest time in the future:
        best_root = min(real_roots, key=(lambda i : abs(i.real - last_time) ) )
        if debug:
            print(f'{best_root=}')

        # Calculate time remaining and ETA
        time_left = best_root.real-last_time
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
            pyplot.scatter(x, y)
            # Get a range of x values for plotting function
            x_line = np.arange(min(x), max(x)+100, 1)
            # Calcualte y values for each x value
            #y_line = objective(x_line, a, b, c)
            y_line = poly_func(x_line)
            # Plot the fitted function line
            pyplot.plot(x_line, y_line, '--', color='red')
            pyplot.show()

        return remaining
