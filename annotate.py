import sys

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
    confs = []
    prev_end = 0.0
    print('Annotating file ', path)
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
        for i in range(int(abs(float(prev_end) - float(start)) * POINTSPERSEC)):
            confs.append(MINCONF)
        end = input('Please enter endpoint: ')
        prev_end = end
        if float(end) <= float(start):
            print('End has to be greater than Start. Abborting')
            break
        for i in range(int(abs(float(end) - float(start)) * POINTSPERSEC)):
            confs.append(MAXCONF)
        print(f'Added annotation between {str(start)} and {str(end)}.')

    return confs


for p in sys.argv[1:]:
    confs = process_file(p)
    
print('')
print('Done')
