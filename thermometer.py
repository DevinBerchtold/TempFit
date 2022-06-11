import glob
import sys
import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot
import datetime as dt

## CONSTANTS
# Turkey
breast_done = 155
stuffing_done = 150
thigh_done = 180

# Beef
brisket_done = 210

# Pork
butt_done = 195


## INITIALIZE

done_temp = brisket_done

day = dt.date.today()
daystring = day.strftime('%Y%m%d') # for filename
print(f'Today: {daystring}')


### OVERRIDE ###
# Use these overrides to test with data from a previous cook

# Brisket 1
daystring = '20211113' 
done_temp = 195.0
start_date = np.datetime64("2021-11-13T06:00:00")
end_date = np.datetime64("2021-11-13T16:00:00")
period = np.timedelta64(15, 'm')
intervals = [np.timedelta64(120, 'm'),np.timedelta64(240, 'm'),np.timedelta64(360, 'm')]
intervals = [np.timedelta64(360, 'm')]

# Turkey 1
# daystring = '20211201' 
# done_temp = 160.0
# start_date = np.datetime64("2021-11-25T14:00:00")
# end_date = np.datetime64("2021-11-25T18:00:00")
# period = np.timedelta64(10, 'm')
# intervals = [np.timedelta64(30, 'm'), np.timedelta64(60, 'm'), np.timedelta64(90, 'm')]
# done_temps = [stuffing_done, thigh_done, breast_done, breast_done]

#Pastrami 1
# daystring = '20211215' 
# done_temp = 195.0
# start_date = np.datetime64("2021-11-13T06:00:00")
# end_date = np.datetime64("2021-11-13T16:00:00")
# period = np.timedelta64(15, 'm')
# intervals = [np.timedelta64(120, 'm'),np.timedelta64(240, 'm'),np.timedelta64(360, 'm')]
# intervals = [np.timedelta64(360, 'm')]

#Butt 1
# daystring = '20211228'

# ???
daystring = '20220328'

print(f'Today: {daystring} (Overridden)')

### OVERRIDE ###


files = glob.glob(f'cooks/history_probe_?_{daystring}*.csv')

print(f'Found these files:\n{files}')


# Get all CSVs that match the filename pattern and read them into numpy dataframes
dataframes = []
for f in files:
    # Excel Format:
    #dataframes.append(pd.read_csv(f,parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1]))
    # Inkbird Format:
    dataframes.append(pd.read_csv(f,delimiter='\t',encoding='utf_16',parse_dates=True,infer_datetime_format=True,index_col=0,usecols=[0, 1]))

# Pick the last one (most recent) to process
dataframe = dataframes[-1]

def estimate(df, start, end, display=True, chart=False, debug=False):
    if debug:
        print("== D E B U G   O N ==")
        print(df.__repr__())
        print(type(df))
        print(start)
        print(end)
    
    # Get subset of dataframe with samples from start to end
    filtered = df.loc[start:end]
    
    # Number of minutes since beginning of period    
    # t0 = pd.Timestamp('2021-11-25T00:00:00')
    t0 = df.index.values[0]
    minutes = [(t-t0)/np.timedelta64(1, 'm') for t in filtered.index.values]

    # Add minutes to dataframe
    # filtered['Minutes'] = minutes
    # filtered = filtered.set_index(minutes)

    # Choose the input and output variables
    x, y = minutes, filtered["Temperature"].values
    if debug:
        print(f"X: {x}")
        print(f"Y: {y}")

    # Use numpy to fit polynomial function to data
    coeffs = np.polyfit(x, y, 3)
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
    
    if chart:
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


#print(f'{start_date=}, {end_date=}, {period=}, {intervals=}')

##x, y = dataframe.index.values, dataframe["Temperature"].values
#estimate(dataframes[0],start_date,end_date)
#estimate(dataframes[1],start_date,end_date)
#estimate(dataframes[2],start_date,end_date)
#estimate(dataframes[3],start_date,end_date)
#
##sys.exit(0)
#for interval in intervals:
#    print(f'{interval=}')
#    for s in np.arange(start_date,end_date,period):
#        #print(s)
#        estimate(dataframe,s-interval,s)

## Actual Test ##

end_date = dataframe.index[-1]

intervals = [np.timedelta64(i, 'm') for i in [60,120]]
times = [np.timedelta64(i, 'm') for i in range(90,-5,-5)]
print(f'Intervals: {intervals}')
print(f'Times: {times}')
# time_lists = [(end_date-t-i,end_date-t) for t in times for i in intervals]
# print(f'{time_lists=}')

for i in intervals:
    print(f'interval: {i}')
    for t in times:
        estimate(dataframe,end_date-t-i,end_date-t)

estimate(dataframe,end_date-max(intervals),end_date,chart=True,display=False)

# estimate(dataframe,end_date-np.timedelta64(30, 'm'),end_date,debug=True)
# estimate(dataframe,end_date-np.timedelta64(60, 'm'),end_date)
# estimate(dataframe,end_date-np.timedelta64(90, 'm'),end_date)
# estimate(dataframe,end_date-np.timedelta64(120, 'm'),end_date)
# estimate(dataframe,end_date-np.timedelta64(150, 'm'),end_date,chart=True)