"""Print an instrument_sites.py result. Usage: python show_sites.py <json> [...]"""
import json, sys

for path in sys.argv[1:]:
    data = json.load(open(path))
    print(path.rsplit('/', 1)[-1], data['programs'])
    for row in data['rows']:
        print(' ', row['kind'].ljust(9), row['name'].ljust(46), str(row['count']).rjust(6), row['where'])
    for name, keys in data['param_keys'].items():
        print('  params', name, keys)
