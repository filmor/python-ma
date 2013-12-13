from pyma.data import load_data, write_data, symmetrise_data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Symmetrise a set of "
                                                 "correlation functions")

    parser.add_argument("input_file")
    parser.add_argument("-s", "--suffix", default="_sym")
    parser.add_argument("-k", "--key")

    args = parser.parse_args()

    assert args.suffix
    assert args.key != ""

    data = load_data(args.input_file, args.key)
    sym = symmetrise_data(data)
    write_data(args.input_file, args.key + args.suffix, sym)

