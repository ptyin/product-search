import sys
from src.GraphSearch.run import run as run_graph_search
from src.MetaSearch.run import run as run_meta_search
from src.HEM.run import run as run_hem

if __name__ == '__main__':
    model_str = sys.argv[1:2]
    if len(model_str) != 0 and model_str[:2] == '--':
        run_str = 'run_{}()'.format(model_str)
        eval(run_str)
    else:
        # run_graph_search()
        run_meta_search()
        # run_hem()
