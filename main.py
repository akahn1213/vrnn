import argparse
import yaml
from helpers.processing import process
from helpers.train import train
from helpers.plotting import make_eval_plots

def main(args):

    if args.process:
        process(args.n_consts)

    if args.train:
        train(args)

    if args.draw_plots:
        make_eval_plots(args)

if __name__ == "__main__":

    with open("helpers/defaults.yaml", "r") as def_yaml:
        defaults_all = yaml.safe_load(def_yaml)      
        
    defaults = defaults_all["defaults"]
    help_default = defaults_all["help"]
 
    parser = argparse.ArgumentParser()

    parser.add_argument("--process", "-p", help=help_default["process"], 
                        action=defaults["process"])
                        
    parser.add_argument("--train", "-t", help=help_default["train"], 
                        action=defaults["train"])
                        
    parser.add_argument("--evaluate", "-e", help=help_default["evaluate"], 
                        action=defaults["evaluate"])
                        
    parser.add_argument("--draw_plots", "-d", help=help_default["draw_plots"], 
                        action=defaults["draw_plots"])
                        
    parser.add_argument("--n_jets", "-j", help=help_default["n_jets"], 
                        type=int, default=defaults["n_jets"])
                        
    parser.add_argument("--n_consts", "-c", help=help_default["n_consts"], 
                        type=int, default=defaults["n_consts"])
                        
    parser.add_argument("--kl_weight", "-k", help=help_default["kl_weight"], 
                        type=float, default=defaults["kl_weight"])
                        
    parser.add_argument("--h_dim", "-l", help=help_default["h_dim"], 
                        type=int, default=defaults["h_dim"])
                        
    parser.add_argument("--z_dim", "-z", help=help_default["z_dim"], 
                        type=int, default=defaults["z_dim"])
                        
    parser.add_argument("--sample", "-s", help=help_default["sample"], 
                        default=defaults["sample"])

    args = parser.parse_args()

    main(args)
