"""
quick rewriting of the argparse functions
"""

import argparse


def create_default_parser():
    
    parser=argparse.ArgumentParser(description=""" 
    Description of you script
    """,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("PROTOCOLS",\
                        help="""
                        Mandatory argument, either:
                        - ssjdfh
                        - WN
                        - weoriu
                        """, default='WN')
    parser.add_argument("--mean",help="mean of the random values", type=float, default=5.)
    parser.add_argument("--std",help="std of the random values", type=float, default=10.)
    parser.add_argument("--n",help="number of random events", type=int, default=2000)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-s", "--save", help="save the figures", action="store_true")
    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    args = parser.parse_args()
    
    args_dict = vars(args) # dictionary version of the Namespace
    
    return args_dict
