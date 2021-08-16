import sys
import config as cfg

FILENAME = 'annotations.csv'
POINTSPERSEC = 10 # 1000 10^-3 10 for 10^-1
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
    print('Annotating file ', path)
    selection_id = 1
    while done != YESKEY:
        start = input('Please enter startpoint or skip: ')
        if start == SKIPKEY:
            done = input('Done? y or n: ')
            if done == YESKEY:
                break
            else:
                continue
        if float(start) <= float(prev_end):
            print('Please list annotations sorted. Start has to be greater than previus End. Abborting')
            break
        end = input('Please enter endpoint: ')
        prev_end = end
        if float(end) <= float(start):
            print('End has to be greater than Start. Abborting')
            break
        for i in range(int(abs(float(end) - float(start)) * POINTSPERSEC)):
            end += pow(10, -POINTSPERSEC)
            f'{selection_id}{cfg.CSV_DLIM}Spectogram_1{cfg.CSV_DLIM}1{cfg.CSV_DLIM}{path}{cfg.CSV_DLIM}{start}{cfg.CSV_DLIM}{end}{cfg.CSV_DLIM}{-1}{cfg.CSV_DLIM}{-1}{cfg.CSV_DLIM}{getCode(c[0])}{cfg.CSV_DLIM}{c[0].split("_")[1]}{cfg.CSV_DLIM}{c[1]}{cfg.CSV_DLIM}{rank}{cfg.CSV_DLIM}{cfg.SPEC_OVERLAP}\n'
            start += pow(10, -POINTSPERSEC)
            selection_id += 1
        print(f'Added annotation between {str(start)} and {str(end)}.')

    return lines

if len(sys.argv[1:]) == 0:
    print('No files given')
else:
    for p in sys.argv[1:]:
        confs = process_file(p)


print('')
print('Done')
