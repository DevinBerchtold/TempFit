from thermometer import *
from os.path import exists

done_temp = 200

def print_over(string):
    print('\033[1A'+string+'\033[K')

# Manual mode
print('== TempFit Manual Mode ==')
filename = f'cooks/manual_{dt.date.today():%Y%m%d}.csv'
if exists(filename):
    print(f'Temperatures from {filename}:')
    t = Thermometer(filename)
    print(t.dataframe.to_string(max_rows=7))
else:
    temp = input('Enter the initial temperature... ')
    t = Thermometer(temp=temp)
    print(t.dataframe)

while True:
    temp = float(input('Enter a new datapoint... '))
    t.add(temp)

    # Save after every new datapoint so nothing is lost
    t.dataframe.to_csv(filename, sep='\t', encoding='utf_16')

    if t.dataframe.shape[0] < 5:
        print_over(f"NOW: {t.end} {temp}° --> ETA: ???                 {done_temp}° (in ??:??)")
    else:
        # print(t.dataframe.tail(5))
        t.estimate(done_temp, plot=True, display=False)
        print_over(t.fit_string)
