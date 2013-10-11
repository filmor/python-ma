import json
from os import path

from multiprocessing import Pool, get_logger, cpu_count
import pandas as pd
import traceback

def main(args):
    try:
        return main_(*args)
    except:
        get_logger().error(traceback.format_exc())
        return False

def main_(inp, fil, alg, output_file, overwrite=False):
    def import_(func, params):
        from importlib import import_module
        from functools import partial

        if func is None:
            return lambda data: data

        mod, _, func = func.rpartition(".")
        try:
            func = getattr(import_module(mod), func)
        except:
            raise RuntimeError("Could not load %s from %s" % (func, mod))

        return partial(func, **params)

    from test_algo import load_data, calculate, write_data
    
    input_name, input_config = inp
    filter_name, filter_func, filter_params = fil
    algo_name, algo_func, algo_params = alg
    key = "/".join(["", input_name, algo_name, filter_name])

    if not overwrite:
        try:
            pd.read_hdf(output_file, key)
            print("Skipping %s" % key)
            # TODO
            return True
        except (KeyError, IOError):
            pass

    print("Running %s ... " % key)

    ev_count = int(input_config["ev_count"])

    data = load_data(input_config["filename"], input_config["key"])
    filter = import_(filter_func, filter_params)
    algo = import_(algo_func, algo_params)

    res = calculate(data, algo, filter)

    print("Writing the first %s evs of %s to %s" % (ev_count, key, output_file))
    res[list(range(ev_count))].to_hdf(output_file, key)

    print("Done with %s" % key)

    print("Starting analysis on %s" % key)
    # TODO, from test_algo

    return True

def load_config(config_dir, name):
    try:
        return json.load(open(path.join(config_dir, name + ".conf")))
    except:
        print("Failed to load %s config." % name)
        raise

def unpack(config):
    """Takes a config object and unpacks all its (param) combinations"""
    for name in config:
        if name.startswith("_"):
            continue

        c = config[name]
        names = set()
        if "params" in c:
            for p in c["params"]:
                n = name + "_" + "_".join("%s%s" % a for a in sorted(p.items()))
                if n in names:
                    continue

                names.add(n)
                yield (n, c["func"], p)
        else:
            yield (name, c.get("func", None), {})

def config_iter(config, args=()):
    for inp in config["input"].items():
        for f in unpack(config["filter"]):
            for a in unpack(config["algorithm"]):
                yield (inp, f, a) + args

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Solve GEVPs")

    parser.add_argument('-c', '--config_dir', metavar='DIRECTORY', default="config")
    parser.add_argument('-f', '--overwrite', action='store_true', default=False)
    parser.add_argument('output_file', metavar='OUTPUT.h5')
    parser.add_argument('-p', '--processes', metavar='N', type=int,
            default=cpu_count())

    args = parser.parse_args()

    config_dir = args.config_dir
    output_file = args.output_file
    overwrite = args.overwrite

    config = {
            name: load_config(config_dir, name)
            for name in "input filter algorithm".split()
            }

    pool = Pool(processes=args.processes)
    result = pool.map(main, config_iter(config, (output_file, overwrite)))

    pool.close()
    pool.join()

    import sys
    sys.exit(0 if (all(result)) else -1)


