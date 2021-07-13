import sys
from QL.run import run_ql, run_uql
from TranSearchText.run import run as run_tran_search
from GraphSearch.run import run as run_graph_search
from MetaSearch.run import run as run_meta_search
from LSE.run import run as run_lse
from HEM.run import run as run_hem
from AEM.run import run_aem, run_zam


def train():
    pass


if __name__ == '__main__':
    model_str = sys.argv[1:2]
    if len(model_str) != 0 and model_str[0][:2] != '--':
        model_str = model_str[0]
        run_str = 'run_{}()'.format(model_str)
        run_str = run_str.lower()
        del sys.argv[1]
        eval(run_str)
    else:
        # ----------Default----------
        # run_ql()
        # run_uql()
        # run_tran_search()
        # run_graph_search()
        # run_meta_search()
        # run_lse()
        run_hem()
        # run_aem()
        # run_zam()

