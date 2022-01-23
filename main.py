import argparse
from helpers.processing import process
from helpers.train import train
#from helpers.evaluate import evaluate
#from helpers.plotting import plot


def main(args):

  print(args)

  if args.process:
    process(args.n_consts)

  if args.train:
    train(args)

  #if args.evaluate:
  #  evaluate(args.sample, args.n_jets, args.n_consts, args.kl_weight, args.h_dim, args.z_dim)

  #if args.draw_plots:
  #  plot()

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()

  parser.add_argument("--process", "-p", help="Pre-Process Data", action="store_true")
  parser.add_argument("--train", "-t", help="Train Model", action="store_true")
  parser.add_argument("--evaluate", "-e", help="Evaluate Model", action="store_true")
  parser.add_argument("--draw_plots", "-d", help="Draw Plots", action="store_true")
  parser.add_argument("--n_jets", "-j", help="Maximum number of jets to consider per event, sorted in pT decreasing", type=int, default=1)
  parser.add_argument("--n_consts", "-c", help="Maximum number of constituents to consider per jet, sorted in pT decreasing", type=int, default=20)
  parser.add_argument("--kl_weight", "-k", help="Weight of the KL-Divergence in the loss term", type=float, default=0.1)
  parser.add_argument("--h_dim", "-l", help="Hidden layer dimensionality", type=int, default=16)
  parser.add_argument("--z_dim", "-z", help="Latent layer dimensionality", type=int, default=2)
  parser.add_argument("--sample", "-s", help="Sample to train/evaluate over", default="2Prong")

  args = parser.parse_args()

  main(args)
