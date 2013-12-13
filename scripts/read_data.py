from pyma.data import write_data
import pandas as pd
import numpy as np
from glob import glob
from os.path import basename, join

def find_files(path):
    files = glob(join(path, "*"))

    return pd.DataFrame(
            [
                tuple(
                    map(int, basename(i).split(".")[3:6])
                ) + (i,)
                for i in files
            ], columns=["x", "y", "run", "filename"])

def read_data(files, size, t_max=32):
    # Shape of base data: run, t, matrix (size*2, size*2)
    runs = len(files.run.value_counts())
    raw_data = np.zeros((runs, t_max, size * 2, size * 2))

    for run, df in files.groupby("run"):
        for _, row in df.iterrows():
            x, y, fn = row["x"], row["y"], row["filename"]
            if x >= size or y >= size:
                continue
            
            data = pd.read_csv(fn, delim_whitespace=True, skiprows=1,
                         names=[
                             "spin",
                             "local/smeared",
                             "time",
                             "data",
                             "data_2"
                             ]
                         )
            
            for _, file_row in data.iterrows():
                # Local/smeared: 1 => fully local; 3 => smeared in x;
                #                5 => smeared in y; 7 => smeared in x and y
                ls = file_row["local/smeared"]
                x_prime = 2 * x + (ls == 3 or ls == 7)
                y_prime = 2 * y + (ls == 5 or ls == 7)
                raw_data[run][int(file_row["time"]) - 1][x_prime][y_prime] = \
                        file_row["data"]

    return raw_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Read correlation functions, "
                                                 "and save as hdf5")

    parser.add_argument("input_directory")
    parser.add_argument("output_file")
    parser.add_argument("-k", "--key", metavar="KEY", default="")
    parser.add_argument("-s", "--size", type=int, default=3,
                        help="Matrix dimension")

    args = parser.parse_args()

    assert args.key != ""

    files = find_files(args.input_directory)
    print("Parsing %s files ..." % len(files))
    
    data = read_data(files, args.size)

    print("Writing to %s/%s" % (args.output_file, args.key))
    write_data(args.output_file, args.key, data)
    print("Done")

