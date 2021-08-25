import sys
import config as cfg
from pathlib import Path

FILENAME = 'annotations.csv'
POINTSPERSEC = 1000 # 1000 10^-3 10 for 10^-1
ROUNDDIGITS = 3 # corresponds to POINTSPERSEC
FRAMESIZE = 3.0 # predefined by BirdNET
SKIPKEY = 's'
YESKEY = 'y'
NOKEY = 'n'
MINCONF = 0.0
MAXCONF = 1.0

print('***      Audio annotatation utility      ***')
print('')
print('Enter start and end points of bird voices in seconds or enter s to skip.')
print('')


def process_file(path):
    done = NOKEY
    lines = ''
    prev_end = 0.0
    prev_start = 0.0
    print('Annotating file ', path)
    selection_id = 1
    while done != YESKEY:
        start = input('Please enter startpoint or skip: ')
        prev_start = start
        if start == SKIPKEY:
            done = input('Done? y or n: ')
            if done == YESKEY:
                break
            else:
                continue
        if float(start) <= prev_end:
            print('Please list annotations sorted. Start has to be greater than previus End. Abborting')
            break
        end = float(input('Please enter endpoint: '))
        prev_end = end
        if end <= float(start):
            print('End has to be greater than Start. Abborting')
            break
        code = input('Please enter eBird Species Code:')
        for i in range(int(abs(end - float(start)) * POINTSPERSEC)):
            lines += f'{selection_id}{cfg.CSV_DLIM}Spectogram_1{cfg.CSV_DLIM}1{cfg.CSV_DLIM}{path}{cfg.CSV_DLIM}{start}{cfg.CSV_DLIM}{round(float(start) + FRAMESIZE, ROUNDDIGITS)}{cfg.CSV_DLIM}{-1}{cfg.CSV_DLIM}{-1}{cfg.CSV_DLIM}{code}{cfg.CSV_DLIM}{path}{cfg.CSV_DLIM}1.0{cfg.CSV_DLIM}1{cfg.CSV_DLIM}{cfg.SPEC_OVERLAP}\n'
            start = float(start)
            start += 1/POINTSPERSEC
            start = round(start, ROUNDDIGITS)
            selection_id += 1
        
        print(f'Added {code} between {str(prev_start)} and {str(end)}.')

    return lines

path_str = sys.argv[-1]
csv_f = Path(path_str)

if len(sys.argv[1:]) + 1 == 3:
    for p in sys.argv[1:-1]:
        lines = process_file(p)
        if csv_f.is_file():
            print(f'Created file {path_str}')
            print(f'Wrote annotations for {p} to file {path_str}')
            with open(path_str, 'a') as csvfile:
                csvfile.write(lines)
        else:
            print(f'Wrote annotations for {p} to file {path_str}')
            with open(path_str, 'w') as csvfile:
                csvfile.write(lines)
else:
    print('usage: $ python3 annotate.py [FILESPATHS]⁺ [OUTPUTPATH]') 

print('')
print('Done')
