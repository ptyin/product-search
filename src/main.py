import sys
from GraphSearch.run import run as run_graph_search
from MetaSearch.run import run as run_meta_search
from LSE.run import run as run_lse
from HEM.run import run as run_hem
from AEM.run import run_aem, run_zam

if __name__ == '__main__':
    model_str = sys.argv[1:2]
    if len(model_str) != 0 and model_str[0][:2] != '--':
        model_str = model_str[0]
        run_str = 'run_{}()'.format(model_str)
        run_str = run_str.lower()
        del sys.argv[1]
        eval(run_str)
    else:
        # run_graph_search()
        # run_meta_search()
        # run_lse()
        run_hem()
        # run_aem()
        # run_zam()

