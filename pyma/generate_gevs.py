import json
from os import path

from multiprocessing import Pool, get_logger, cpu_count
import pandas as pd
import numpy as np
import traceback

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


def main(args):
    try:
        if args[0] == "filter":
            return run_filter(*args[1:])
        elif args[0] == "algo":
            return run_algo(*args[1:])
    except:
        get_logger().error(traceback.format_exc())
        return False

def check_key(filename, key):
    try:
        store = pd.HDFStore(filename, "r")
        result = key in store
        store.close()
    except IOError:
        return False
    return result

def run_filter(inp, fil, args):
    from test_algo import load_data, write_data
    input_name, input_config = inp
    filter_name, filter_func, filter_params = fil

    key = "/".join(["/data", input_name, filter_name])
    
    if not args.overwrite and check_key(args.output_file, key):
        print("Skipping %s" % key)
        return True

    print("Running %s ... " % key)
    data = load_data(input_config["filename"], input_config["key"])
    filter = import_(filter_func, filter_params)

    result = np.array([
        filter(t) for t in data
        ])

    print("Writing filtered data %s to %s" % (key, args.output_file))
    write_data(args.output_file, key, result)

    print("Done with %s" % key)
    return True


def run_algo(inp, fil, alg, args):
    from test_algo import load_data, calculate
    
    input_name, _ = inp
    filter_name, _, _ = fil

    input_key = "/".join(["/data", input_name, filter_name])

    algo_name, algo_func, algo_params = alg
    key = "/".join(["/gevs", input_name, algo_name, filter_name])

    if not args.overwrite and check_key(args.output_file, key):
        print("Skipping %s" % key)
        return True

    print("Running %s ... " % key)

    data = load_data(args.output_file, input_key) 
    algo = import_(algo_func, algo_params)

    res = calculate(data, algo)

    print("Writing the gevs of %s to %s" % (key, output_file))
    res.to_hdf(output_file, key)

    print("Done with %s" % key)

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

def filter_iter(config, args):
    for inp in config["input"].items():
        if inp[0].startswith("_"):
            continue
        for f in unpack(config["filter"]):
            yield ("filter", inp, f, args)

def algo_iter(config, args):
    for inp in config["input"].items():
        if inp[0].startswith("_"):
            continue
        for f in unpack(config["filter"]):
            for a in unpack(config["algorithm"]):
                yield ("algo", inp, f, a, args)

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
    result = pool.map(main, filter_iter(config, args))
    result += pool.map(main, algo_iter(config, args))

    pool.close()
    pool.join()

    import sys
    sys.exit(0 if (all(result)) else -1)

