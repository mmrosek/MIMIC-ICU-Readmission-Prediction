"""
    Reads NOTEEVENTS file, finds the discharge summaries, preprocesses them and writes out the filtered dataset.
"""
import csv

from nltk.tokenize import RegexpTokenizer

from tqdm import tqdm

#retain only alphanumeric

# w --> matches any alpha numeric character
# + --> causes resulting RE to match 1+ repetitions of the preceding RE
tokenizer = RegexpTokenizer(r'\w+')

note_events_path = "../../../../Data/"

def write_discharge_summaries(in_path, out_path):
    notes_file = '{}/NOTEEVENTS.csv'.format(in_path)
    print("processing notes file")
    with open(notes_file, 'r') as csvfile:
        with open(out_path, 'w') as outfile:
            print("writing to %s" % (out_path))
            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']) + '\n')
            notereader = csv.reader(csvfile)
            #header
            next(notereader)
            i = 0
            for line in tqdm(notereader):
                subj = int(line[1]) # DONT THINK NEEDED
                category = line[6]
                if category == "Discharge summary":
                    note = str(line[10]) # CHANGED unicode() --> str()

                    #tokenize, lowercase and remove numerics
                    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
                    text = '"' + ' '.join(tokens) + '"'
                    outfile.write(','.join([line[1], line[2], line[4], text]) + '\n')
                i += 1
    return out_path

if __name__ == "__main__":
    write_discharge_summaries(note_events_path,note_events_path+"/disch_clean.csv")