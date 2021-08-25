import os
import argparse
import time
import numpy as np
import re
from pathlib import Path
from datetime import datetime

import config as cfg
from metadata import grid
from utils import audio
from model import model
from utils import log

import warnings
warnings.filterwarnings('ignore')


ROUNDDIGITS = 3
################### DATASAT HANDLING ####################


def parseTestSet(path, file_type='wav'):

    # Find all soundscape files
    dataset = []
    if os.path.isfile(path):
        dataset.append(path)
    else:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                if f.rsplit('.', 1)[-1].lower() == file_type:
                    dataset.append(os.path.abspath(os.path.join(dirpath, f)))

    # Dataset stats
    log.p(('FILES IN DATASET:', len(dataset)))

    return dataset

##################### LOAD MODEL #######################


def loadModel():

    # Load trained net
    snapshot = model.loadSnapshot('model/BirdNET_Soundscape_Model.pkl')

    # Build simple model
    net = model.buildNet()

    # Load params
    net = model.loadParams(net, snapshot['params'])

    # Compile test function
    test_function = model.test_function(net, layer_index=-2)

    return test_function

######################### EBIRD #########################


def loadGridData():
    grid.load()


def setSpeciesList(lat, lon, week):

    if not week in range(1, 49):
        week = -1

    if cfg.USE_EBIRD_CHECKLIST:
        cfg.WHITE_LIST, cfg.BLACK_LIST = grid.getSpeciesLists(
            lat, lon, week, cfg.EBIRD_THRESHOLD)
    else:
        cfg.WHITE_LIST = cfg.CLASSES

    log.p(('SPECIES:', len(cfg.WHITE_LIST)), new_line=False)

######################  EXPORT ##########################


def getTimestamp(start, end):

    m_s, s_s = divmod(start, 60)
    h_s, m_s = divmod(m_s, 60)
    start = str(h_s).zfill(2) + ":" + \
                str(m_s).zfill(2) + ":" + str(s_s).zfill(2)

    m_e, s_e = divmod(end, 60)
    h_e, m_e = divmod(m_e, 60)
    end = str(h_e).zfill(2) + ":" + str(m_e).zfill(2) + ":" + str(s_e).zfill(2)

    return start + '-' + end


def decodeTimestamp(t):

    start = t.split('-')[0].split(':')
    end = t.split('-')[1].split(':')
   
    if len(end) == 3 and len(start) == 3:
        start_seconds = float(re.sub('e', '', start[0])) * 3600 + float(
            re.sub('e', '', start[1])) * 60 + float(re.sub('e', '', start[2]))
        end_seconds = float(re.sub('e', '', end[0])) * 3600 + float(
            re.sub('e', '', end[1])) * 60 + float(re.sub('e', '', end[2]))
        return start_seconds, end_seconds
    else:
        return -1, -1


def getCode(label):

    codes = grid.CODES

    for c in codes:
        if codes[c] == label:
            return c

    return '????'


def get_csv_table(p, path):
    stable = ''
    dlim = cfg.CSV_DLIM
    selection_id = 0
    start_times = []
    end_times = []
    for timestamp in sorted(p):
        rstring = ''
        start, end = decodeTimestamp(timestamp)
        min_conf = 0
        rank = 1
        for c in p[timestamp]:
            if c[1] > cfg.MIN_CONFIDENCE + min_conf and c[0] in cfg.WHITE_LIST:
                selection_id += 1
                rstring += f'{selection_id}{dlim}Spectogram_1{dlim}1{dlim}{path}{dlim}{round(start, ROUNDDIGITS)}{dlim}{round(end, ROUNDDIGITS)}{dlim}{cfg.SPEC_FMIN}{dlim}{cfg.SPEC_FMAX}{dlim}{getCode(c[0])}{dlim}{c[0].split("_")[1]}{dlim}{c[1]}{dlim}{rank}{dlim}{cfg.SPEC_OVERLAP}\n'
                start_times.append(start)
                end_times.append(end)
                rank += 1
            if rank > 5:
                break

        # Write result string to file
        if len(rstring) > 0:
            stable += rstring

    return stable, selection_id, start_times, end_times


###################### ANALYSIS #########################


def analyzeFile(soundscape, test_function):

    ncnt = 0

    # Store analysis here
    analysis = {}

    # Keep track of timestamps
    pred_start = 0

    # Set species list accordingly
    setSpeciesList(
        cfg.DEPLOYMENT_LOCATION[0], cfg.DEPLOYMENT_LOCATION[1], cfg.DEPLOYMENT_WEEK)

    # Get specs for file
    spec_batch = []
    for spec in audio.specsFromFile(soundscape,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    win_len=cfg.WIN_LEN,
                                    spec_type=cfg.SPEC_TYPE,
                                    magnitude_scale=cfg.MAGNITUDE_SCALE,
                                    bandpass=True,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                                    offset=0,
                                    duration=None):

        # Prepare as input
        spec = model.prepareInput(spec)

        # Add to batch
        if len(spec_batch) > 0:
            spec_batch = np.vstack((spec_batch, spec))
        else:
            spec_batch = spec

        # Do we have enough specs for a prediction?
        if len(spec_batch) >= cfg.SPECS_PER_PREDICTION:

            # Make prediction
            p, _ = model.predict(spec_batch, test_function)

            # Calculate next timestamp
            pred_end = pred_start + cfg.SPEC_LENGTH + \
                ((len(spec_batch) - 1) * (cfg.SPEC_LENGTH - cfg.SPEC_OVERLAP))

            # Store prediction
            analysis[getTimestamp(pred_start, pred_end)] = p

            # Advance to next timestamp
            pred_start = pred_end - cfg.SPEC_OVERLAP
            spec_batch = []

    return analysis

def patches_consequtive(start, end, duration, delta):
    return abs(start - end) < (duration + delta)

######################## MAIN ###########################


def process(soundscape, sid, out_dir, out_type, test_function, start_time, write_csv_header):

    # Time
    start = time.time()
    log.p(('SID:', sid, 'PROCESSING:',
          soundscape.split(os.sep)[-1]), new_line=False)

    # Analyze file
    p = analyzeFile(soundscape, test_function)

    # Generate csv table
    csv_stable, x, start_times, end_times = get_csv_table(p, soundscape.split(os.sep)[-1])
    log.p(('DETECTIONS:', x), new_line=False)

    # Save results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dlim = cfg.CSV_DLIM
    
    dt_string = start_time.strftime("%d_%m_%Y-%H:%M:%S")
    path_str =  f'{out_dir}{dt_string}-{cfg.SPEC_OVERLAP}_{cfg.SENSITIVITY}_{cfg.MIN_CONFIDENCE}.csv'
    csv_header = f'Selection{dlim}View{dlim}Channel{dlim}Begin_File{dlim}Begin{dlim}End{dlim}Low_Freq{dlim}High_Freq{dlim}Species_Code{dlim}Name{dlim}Confidence{dlim}Rank{dlim}Overlap\n'

    duration = audio.get_duration(soundscape)

    # Time
    t=time.time() - start

    # Stats
    log.p(('TIME:', round(t, 3)))

    meta_path_str = f'{out_dir}meta.csv'
    meta_csv_header = f'File{dlim}Processing_time{dlim}Duration\n'

    meta_table = f'{soundscape.split(os.sep)[-1]}{dlim}{t}{dlim}{duration}\n'

    meta_csv_f = Path(meta_path_str)
    csv_f = Path(path_str)

    if csv_f.is_file():
        with open(path_str, 'a') as csvfile:
            csvfile.write(csv_stable)
    else:
        if write_csv_header:
            with open(path_str, 'w') as csvfile:
                csvfile.write(csv_header)
                csvfile.write(csv_stable)
        else:
            with open(path_str, 'a') as csvfile:
                csvfile.write(csv_stable)


    if meta_csv_f.is_file():
        with open(meta_path_str, 'a') as meta_file:
            meta_file.write(meta_table)
    else:
        log.p(('CREATING NEW META-INFO FILE ', meta_path_str), new_line=True)
        if write_csv_header:
            with open(meta_path_str, 'a') as meta_file:
                log.p(('WRITING CSV HEADER'), new_line=True)
                meta_file.write(meta_csv_header)
                meta_file.write(meta_table)
        else:
            with open(meta_path_str, 'a') as meta_file:
                log.p(('NOT WRITING CSV HEADER - TYPE --csv-header TO ENABLE'), new_line=True)
                meta_file.write(meta_table)

    log.p(('WRITING META-INFO TO FILE ', meta_path_str), new_line=True)

    return path_str

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument('--i', default='audio',
                        help='Path to input file or directory.')
    parser.add_argument(
        '--o', default='', help='Path to output directory. If not specified, the input directory will be used.')
    parser.add_argument('--filetype', default='wav',
                        help='Filetype of soundscape recordings. Defaults to \'wav\'.')
    parser.add_argument('--results', default='raven',
                        help='Output format of analysis results. Values in [\'audacity\', \'raven\']. Defaults to \'raven\'.')
    parser.add_argument('--lat', type=float, default=-1,
                        help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1,
                        help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1,
                        help='Week of the year when the recordings were made. Values in [1, 48]. Set -1 to ignore.')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Overlap in seconds between extracted spectrograms. Values in [0.0, 2.999].')
    parser.add_argument('--spp', type=int, default=1,
                        help='Combines probabilities of multiple spectrograms to one prediction. Defaults to 1.')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Sigmoid sensitivity; Higher values result in lower sensitivity. Values in [0.25, 2.0]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1,
                        help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--csv-header', dest='write_csv_header', action='store_true',
                        help='Defines if csv_header should be added to output file. Defaults to false.')
    parser.set_defaults(write_csv_header=False)
    args=parser.parse_args()

    # Parse dataset
    dataset=parseTestSet(args.i, args.filetype)
    if len(dataset) > 0:

        # Load model
        test_function=loadModel()

        # Load eBird grid data
        loadGridData()

        # Adjust config
        cfg.DEPLOYMENT_LOCATION=(args.lat, args.lon)
        cfg.DEPLOYMENT_WEEK=args.week
        cfg.SPEC_OVERLAP=min(2.999, max(0.0, args.overlap))
        cfg.SPECS_PER_PREDICTION=max(1, args.spp)
        cfg.SENSITIVITY=max(min(-0.25, args.sensitivity * -1), -2.0)
        cfg.MIN_CONFIDENCE=min(0.99, max(0.01, args.min_conf))
        if len(args.o) == 0:
            if os.path.isfile(args.i):
                result_path=args.i.rsplit(os.sep, 1)[0]
            else:
                result_path=args.i
        else:
            result_path=args.o

        # Analyze dataset
        start_time = datetime.now()
        path_str = ''
        for s in dataset:
            path_str = process(s, dataset.index(s) + 1, result_path,
                    args.results, test_function, start_time, args.write_csv_header)
        
        if args.write_csv_header:
            log.p(('WRITING CSV HEADER'), new_line=True)
        else:
            log.p(('NOT WRITING CSV HEADER - TYPE --csv-header TO ENABLE'), new_line=True)
        log.p(('WRITING RESULTS TO FILE ', path_str), new_line=True)

if __name__ == '__main__':

    main()
