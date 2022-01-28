import matplotlib.pyplot as plt 
import numpy as np
import sys
import os
from sklearn.metrics import roc_curve, auc
from helpers.eval import get_eval_data

def make_roc_plot(fpr, tpr, fpr_v, tpr_v, roc_auc, roc_auc_v, epoch, train_name):
  
    plt.plot(fpr, tpr, label='Training Set (AUC = %0.3f)' % (roc_auc))
    plt.plot(fpr_v, tpr_v, label='Validation Set (AUC = %0.3f)' % (roc_auc_v))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'ROC: Epoch {epoch}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(f"plots/{train_name}/evaluation/roc_epoch{epoch}.png")
    plt.clf()

def make_eval_plots(args):

    train_name = f"train_{args.sample}_{args.n_consts}_top{args.n_jets}_{str(args.kl_weight).replace('.','p')}_{args.h_dim}_{args.z_dim}"

    plots_dir = f"plots/{train_name.replace('train', 'eval')}/"

    if not os.path.exists(sys.path[0]+"/"+plots_dir):
        try:
            os.makedirs(sys.path[0]+"/"+plots_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    train_data = get_eval_data(f"eval_data/training_{train_name}_eval_data.npy")
    val_data = get_eval_data(f"eval_data/validation_{train_name}_eval_data.npy")
    anom_data = get_eval_data(f"eval_data/anomaly_{train_name}_eval_data.npy")
       
    transformation = 0.5/np.mean(train_data["score"])
    train_data["score"] = 1 - (train_data["score"]*transformation)
    val_data["score"] = 1 - (val_data["score"]*transformation)
    anom_data["score"] = 1 - (anom_data["score"]*transformation)
    
    plt.hist(val_data["score"], bins=np.linspace(0, 1, 100), alpha=0.6, density=True, label="Background")
    plt.hist(anom_data["score"], bins=np.linspace(0, 1, 100), alpha=0.6, density=True, label="Signal")
    plt.xlim(0, 1)
    plt.title("Leading Jet Anomaly Score After Transformation", fontsize=14)
    plt.xlabel('Anomaly Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(f"{plots_dir}{args.sample}_J_Anomaly_Score.png")
    print("Saved Plot:", f"{plots_dir}{args.sample}_J_Anomaly_Score.png")
    plt.clf()
    
    clab = "Contaminated (10%)"
    slab = "Signal" 
    blab = "Background Only"
     
    plt.hist(train_data["m"], bins=np.linspace(0, 2000, 70), alpha=0.6, label=clab)
    plt.hist(anom_data["m"], bins=np.linspace(0, 2000, 70), alpha=0.6, label=slab)
    plt.hist(val_data["m"], bins=np.linspace(0, 2000, 70), alpha=0.6, histtype='step', linestyle='dashed', color='k', label=blab)
    plt.yscale("log")
    plt.xlim(0, 1500)
    plt.ylim(1, 2500)
    plt.title("Leading Jet Mass", fontsize=16)
    plt.xlabel(r'$M_{J}$ [GeV]', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(f"{plots_dir}{args.sample}_J_Mass.png")
    print("Saved Plot:", f"{plots_dir}{args.sample}_J_Mass.png")
    plt.clf()
    
    plt.hist(train_data["m"][np.nonzero(train_data["score"] > 0.6)], bins=np.linspace(0, 2000, 70), alpha = 0.6, label=clab)
    plt.hist(anom_data["m"][np.nonzero(anom_data["score"] > 0.6)], bins=np.linspace(0, 2000, 70), alpha = 0.6, label=slab)
    plt.hist(val_data["m"][np.nonzero(val_data["score"] > 0.6)], bins=np.linspace(0, 2000, 70), histtype='step', alpha = 0.6, linestyle='dashed', color='k', label=blab)
    plt.yscale("log")
    plt.xlim(0, 1500)
    plt.ylim(1, 2500)
    plt.title("Leading Jet Mass, Anomaly Score > 0.6", fontsize=16)
    plt.xlabel(r'$M_{J}$ [GeV]', fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(f"{plots_dir}{args.sample}_J_Mass_AnomScore0p6.png")
    print("Saved Plot:", f"{plots_dir}{args.sample}_J_Mass_AnomScore0p6.png")
    plt.clf()
  
