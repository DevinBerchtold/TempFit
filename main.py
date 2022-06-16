from thermometer import *
import glob

## CONSTANTS

# Turkey
breast_done = 155
stuffing_done = 150
thigh_done = 180

# Beef
brisket_done = 195

# Pork
butt_done = 200


## INITIALIZE ##

done_temp = brisket_done

day = dt.date.today()
daystring = day.strftime('%Y%m%d') # for filename
print(f'Today: {daystring}')
daystring = '20211113' # override

files = glob.glob(f'cooks/history_probe_?_{daystring}*.csv')
print(f'Found these files:\n{files}')

# Get all that match filename pattern and read them into dataframes
therms = []
for file in files:
    therms.append(Thermometer(file))

# Pick the last one (most recent) to process
therm = therms[-1]

## Actual Test ##

intervals = [np.timedelta64(i, 'm') for i in [60,120]]
times = [np.timedelta64(i, 'm') for i in range(90,-5,-5)]

for i in intervals:
    print(f'interval: {i}')
    for t in times:
        therm.estimate(done_temp,start=therm.end-t-i,end=therm.end-t)

therm.estimate(done_temp,start=therm.end-max(intervals),end=therm.end,plot=True,display=False)
