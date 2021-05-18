import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import LogNorm
import sys, os

from sklearn.metrics import roc_curve, auc

plots_dir = "plots/evaluation/"


if not os.path.exists(sys.path[0]+"/"+plots_dir):
  try:
    os.makedirs(sys.path[0]+"/"+plots_dir)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
      raise



prongs = sys.argv[1]

ad_data = np.load("eval_data/ProcR_DrCut_Lead_"+prongs+"Prong_Contaminated_"+prongs+"Prong_Contaminated_Weights_Leading_ConstOnly_Eval_Data.npy")
bg_data = np.load("eval_data/ProcR_DrCut_Lead_Background_"+prongs+"Prong_Contaminated_Weights_Leading_ConstOnly_Eval_Data.npy")
sg_data = np.load("eval_data/ProcR_DrCut_Lead_"+prongs+"Prong_Signal_"+prongs+"Prong_Contaminated_Weights_Leading_ConstOnly_Eval_Data.npy")

j_pt  = ad_data[0]
j_m  = ad_data[1]
j_eta  = ad_data[2]
j_phi  = ad_data[3]
j_score  = ad_data[4]
j_c2  = ad_data[5]
j_d2  = ad_data[6]
j_t1  = ad_data[7]
j_t2  = ad_data[8]
j_t3  = ad_data[9]
j_t21  = ad_data[10]
j_t32  = ad_data[11]
j_t31  = ad_data[12]
j_s12  = ad_data[13]
j_s23  = ad_data[14]

bg_j_pt  = bg_data[0]
bg_j_m  = bg_data[1]
bg_j_eta  = bg_data[2]
bg_j_phi  = bg_data[3]
bg_j_score  = bg_data[4]
bg_j_c2  = bg_data[5]
bg_j_d2  = bg_data[6]
bg_j_t1  = bg_data[7]
bg_j_t2  = bg_data[8]
bg_j_t3  = bg_data[9]
bg_j_t21  = bg_data[10]
bg_j_t32  = bg_data[11]
bg_j_t31  = bg_data[12]
bg_j_s12  = bg_data[13]
bg_j_s23  = bg_data[14]

sg_j_pt  = sg_data[0]
sg_j_m  = sg_data[1]
sg_j_eta  = sg_data[2]
sg_j_phi  = sg_data[3]
sg_j_score  = sg_data[4]
sg_j_c2  = sg_data[5]
sg_j_d2  = sg_data[6]
sg_j_t1  = sg_data[7]
sg_j_t2  = sg_data[8]
sg_j_t3  = sg_data[9]
sg_j_t21  = sg_data[10]
sg_j_t32  = sg_data[11]
sg_j_t31  = sg_data[12]
sg_j_s12  = sg_data[13]
sg_j_s23  = sg_data[14]

transformation = 0.5/np.mean(j_score)
j_score = 1 - (j_score*transformation)
bg_j_score = 1 - (bg_j_score*transformation)
sg_j_score = 1 - (sg_j_score*transformation)


plt.hist(bg_j_score, bins=np.linspace(0, 1, 100), alpha=0.6, density=True, label="Background")
plt.hist(sg_j_score, bins=np.linspace(0, 1, 100), alpha=0.6, density=True, label="Signal")
plt.xlim(0, 1)
plt.title("Leading Jet Anomaly Score After Transformation", fontsize=14)
plt.xlabel('Anomaly Score', fontsize=14)
plt.legend(fontsize=12)
plt.savefig(plots_dir+""+prongs+"Prong_J_Anomaly_Score.png")
print("Saved Plot:", plots_dir+""+prongs+"Prong_J_Anomaly_Score.png")
#plt.show()
plt.clf()



clab = "Contaminated (10%)"
slab = "Signal" 
blab = "Background Only"


plt.hist(j_m, bins=np.linspace(0, 2000, 70), alpha=0.6, label=clab)
plt.hist(sg_j_m, bins=np.linspace(0, 2000, 70), alpha=0.6, label=slab)
plt.hist(bg_j_m, bins=np.linspace(0, 2000, 70), alpha=0.6, histtype='step', linestyle='dashed', color='k', label=blab)
plt.yscale("log")
plt.xlim(0, 1500)
plt.ylim(1, 250000)
plt.title("Leading Jet Mass", fontsize=16)
plt.xlabel(r'$M_{J}$ [GeV]', fontsize=14)
plt.legend(fontsize=12)
plt.savefig(plots_dir+""+prongs+"Prong_J_Mass.png")
print("Saved Plot:", plots_dir+""+prongs+"Prong_J_Mass.png")
plt.clf()

plt.hist(j_m[np.nonzero(j_score > 0.65)], bins=np.linspace(0, 2000, 70), alpha = 0.6, label=clab)
plt.hist(sg_j_m[np.nonzero(sg_j_score > 0.65)], bins=np.linspace(0, 2000, 70), alpha = 0.6, label=slab)
plt.hist(bg_j_m[np.nonzero(bg_j_score > 0.65)], bins=np.linspace(0, 2000, 70), histtype='step', alpha = 0.6, linestyle='dashed', color='k', label=blab)
plt.yscale("log")
plt.xlim(0, 1500)
plt.ylim(1, 250000)
plt.title("Leading Jet Mass, Anomaly Score > 0.65", fontsize=16)
plt.xlabel(r'$M_{J}$ [GeV]', fontsize=14)
plt.legend(fontsize=12)
#plt.show()
plt.savefig(plots_dir+""+prongs+"Prong_J_Mass_AnomScore0p65.png")
print("Saved Plot:", plots_dir+""+prongs+"Prong_J_Mass_AnomScore0p65.png")
plt.clf()

