from thermometer import *
from os.path import exists

done_temp = 200.0

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
        # print(t.dataframe)
        pass
    else:
        # print(t.dataframe.tail(5))
        str = t.estimate(done_temp, plot=True, ret='string', display=False)
        print('\033[1A'+str+'\033[K')
