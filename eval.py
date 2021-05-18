import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN
import numpy as np
import pickle
import h5py

import sys, os
from sklearn.metrics import roc_curve, auc

import time

dataDir = 'Output_h5/'

train_hlvs = False

proc = sys.argv[1]
topN = sys.argv[2]
sample = sys.argv[3]
maxconsts = int(sys.argv[4])
load = bool(int(sys.argv[5]))
kl_weight = float(sys.argv[6])
h_dim = int(sys.argv[7])
z_dim = int(sys.argv[8])
cuda_idx = int(sys.argv[9])
suffix = "_DrCut_Lead"

if(train_hlvs):
  train_name = "lhco_train_"+sample+"_"+str(maxconsts)+"_"+proc+"_"+"top"+topN+suffix+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)
else:
  train_name = "lhco_train_"+sample+"_"+str(maxconsts)+"_"+proc+"_"+"top"+topN+suffix+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)+"_constonly"


if not os.path.exists(sys.path[0]+"/eval_data"):
  try:
    os.makedirs(sys.path[0]+"/eval_data")
  except OSError as exc:
    if exc.errno != errno.EEXIST:
      raise


def get_data(infile = None, n_const = None):
    if(infile is None): infile = h5py.File(dataDir+"events_anomalydetection_"+sample+"_VRNN_"+proc+"_"+str(maxconsts)+"C"+suffix+"_preprocessed.hdf5", "r+")
    data = dict()
    avg_jets = dict()
    hlvs = dict()
    vecs = dict()
    all_hlvs = []
    jetgroup = infile["jets/top"+topN]
    for n_c in jetgroup.keys():
      if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
        print(n_c, np.shape(jetgroup[n_c+"/hlvs"]))
        for i in range(len(jetgroup[n_c+"/hlvs"][()])):
          all_hlvs.append(jetgroup[n_c+"/hlvs"][(i)])
    all_hlvs = np.array(all_hlvs)
    hlv_means = np.mean(all_hlvs, axis=0)
    hlv_stds = np.std(all_hlvs, axis=0)
    for n_c in jetgroup.keys():
      if(len(jetgroup[n_c+"/constituents"][()]) > 0):
        tmp_consts = jetgroup[n_c+"/constituents"][()]
        data.update({n_c: tmp_consts})
        avg_jets.update({n_c: np.mean(tmp_consts, axis=0)[:,0]})
      if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      else:
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      vecs.update({n_c: jetgroup[n_c+"/4vecs"]}) 
    n_events = len(infile["events"].keys())
    return data, hlvs, hlv_means, hlv_stds, n_events, avg_jets, vecs

def get_val_data(hlv_means, hlv_stds, infile = None, n_const = None):
    if(infile is None): infile = h5py.File(dataDir+"events_anomalydetection_Background_VRNN_"+proc+"_"+str(maxconsts)+"C"+suffix+"_preprocessed.hdf5", "r+")
    data = dict()
    hlvs = dict()
    vecs = dict()
    jetgroup = infile["jets/top"+topN]
    for n_c in jetgroup.keys():
      if(len(jetgroup[n_c+"/constituents"][()]) > 0):
        tmp_consts = jetgroup[n_c+"/constituents"][()]
        data.update({n_c: tmp_consts})
      if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      else:
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      vecs.update({n_c: jetgroup[n_c+"/4vecs"]}) 
    n_events = len(infile["events"].keys())
    return data, hlvs, n_events, vecs

def get_anom_data(hlv_means, hlv_stds, infile = None, n_const = None):
    if(infile is None): infile = h5py.File(dataDir+"events_anomalydetection_"+sample.replace("Contaminated","Signal")+"_VRNN_"+proc+"_"+str(maxconsts)+"C"+suffix+"_preprocessed.hdf5", "r+")
    data = dict()
    hlvs = dict()
    vecs = dict()
    jetgroup = infile["jets/top"+topN]
    for n_c in jetgroup.keys():
      if(len(jetgroup[n_c+"/constituents"][()]) > 0):
        tmp_consts = jetgroup[n_c+"/constituents"][()]
        data.update({n_c: tmp_consts})
      if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      else:
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      vecs.update({n_c: jetgroup[n_c+"/4vecs"]}) 
    n_events = len(infile["events"].keys())
    return data, hlvs, n_events, vecs

def evaluate_training(hlv_means, hlv_stds, n_events):
  kld_losses = []
  vecs_save = []
  hlvs_save = []
  print("Evaluating Training")
  for n_c in data_train.keys():
    if(int(n_c.replace("C","")) in const_list):
      start = time.time()
      n_jets = len(data_train[n_c])
      batch_counter = 0
      avgs = avg_jets[n_c]
      while(batch_counter < n_jets and batch_counter < max_batches*batch_size):
        batch_start = time.time()
        batch_step = min(batch_size, n_jets - batch_counter)
        data = torch.tensor(data_train[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        #print(data.size())
        hlvs = torch.tensor(hlvs_train[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        vecs = torch.tensor(vecs_train[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        data = Variable(data.transpose(0, 1))
        #kld_loss, _, _, y_mean = model(data)
        kld_loss, _, _, y_mean = model(data, hlvs, avgs, kl_weight)
        loss_kld = kld_loss.data.cpu().numpy()
        kld_losses = np.concatenate((kld_losses, loss_kld))

        if(len(vecs_save) == 0): vecs_save = vecs.cpu().numpy()
        else: vecs_save = np.concatenate((vecs_save, vecs.cpu().numpy()), axis=0)
        if(len(hlvs_save) == 0): hlvs_save = hlvs.cpu().numpy()
        else: hlvs_save = np.concatenate((hlvs_save, hlvs.cpu().numpy()), axis=0)

        batch_counter += batch_step

  return kld_losses, vecs_save, hlvs_save

def evaluate_validation(hlv_means, hlv_stds, n_events):
  kld_losses = []
  vecs_save = []
  hlvs_save = []
  print("Evaluating Validation")

  for n_c in data_val.keys():
    if(int(n_c.replace("C","")) in const_list):
      start = time.time()
      n_jets = len(data_val[n_c])
      batch_counter = 0
      avgs = avg_jets[n_c]
      while(batch_counter < n_jets and batch_counter < max_batches*batch_size):
        batch_start = time.time()
        batch_step = min(batch_size, n_jets - batch_counter)
        data = torch.tensor(data_val[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        hlvs = torch.tensor(hlvs_val[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        vecs = torch.tensor(vecs_val[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        data = Variable(data.transpose(0, 1))
        #kld_loss, _, _, y_mean = model(data)
        kld_loss, _, _, y_mean = model(data, hlvs, avgs, kl_weight)
        loss_kld = kld_loss.data.cpu().numpy()
        kld_losses = np.concatenate((kld_losses, loss_kld))

        if(len(vecs_save) == 0): vecs_save = vecs.cpu().numpy()
        else: vecs_save = np.concatenate((vecs_save, vecs.cpu().numpy()), axis=0)
        if(len(hlvs_save) == 0): hlvs_save = hlvs.cpu().numpy()
        else: hlvs_save = np.concatenate((hlvs_save, hlvs.cpu().numpy()), axis=0)

        batch_counter += batch_step

  return kld_losses, vecs_save, hlvs_save

def evaluate_anom(hlv_means, hlv_stds, n_events):
  kld_losses = []
  vecs_save = []
  hlvs_save = []
  print("Evaluating Anom")

  for n_c in data_anom.keys():
    if(int(n_c.replace("C","")) in const_list):
      start = time.time()
      n_jets = len(data_anom[n_c])
      batch_counter = 0
      avgs = avg_jets[n_c]
      while(batch_counter < n_jets and batch_counter < max_batches*batch_size):
        batch_start = time.time()
        batch_step = min(batch_size, n_jets - batch_counter)
        data = torch.tensor(data_anom[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        hlvs = torch.tensor(hlvs_anom[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        vecs = torch.tensor(vecs_anom[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        data = Variable(data.transpose(0, 1))
        #kld_loss, _, _, y_mean = model(data)
        kld_loss, _, _, y_mean = model(data, hlvs, avgs, kl_weight)
        loss_kld = kld_loss.data.cpu().numpy()
        kld_losses = np.concatenate((kld_losses, loss_kld))

        if(len(vecs_save) == 0): vecs_save = vecs.cpu().numpy()
        else: vecs_save = np.concatenate((vecs_save, vecs.cpu().numpy()), axis=0)
        if(len(hlvs_save) == 0): hlvs_save = hlvs.cpu().numpy()
        else: hlvs_save = np.concatenate((hlvs_save, hlvs.cpu().numpy()), axis=0)

        batch_counter += batch_step

  return kld_losses, vecs_save, hlvs_save


#hyperparameters
x_dim = 3
hlv_dim = 10
n_layers =  1
n_epochs  = 500
clip = 10
learning_rate = 1e-5
l2_norm = 0
batch_size = 256
seed = 128
print_every = 100
max_batches = 5000
plot_every = 20

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("CUDA Device:", torch.cuda.get_device_name())
torch.cuda.set_device(cuda_idx)
print("CUDA Index:", torch.cuda.current_device())

data_train, hlvs_train, hlv_means, hlv_stds, n_train_events, avg_jets, vecs_train = get_data()
data_val, hlvs_val, n_val_events, vecs_val = get_val_data(hlv_means, hlv_stds)
data_anom, hlvs_anom, n_anom_events, vecs_anom = get_anom_data(hlv_means, hlv_stds)

print(data_train.keys())
const_list = range(3, maxconsts+1)

losses_train = []
losses_val = []
model = VRNN(x_dim, hlv_dim, h_dim, z_dim, n_layers, train_hlvs)

model.eval()

fn = 'saves/vrnn_state_dict_'+train_name+'_epoch_100.pth'
state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)

epoch = 100

eval_time = time.time()
scores_normal, vecs_t, hlvs_t = evaluate_training(hlv_means, hlv_stds, n_train_events)
scores_val, vecs_v, hlvs_v = evaluate_validation(hlv_means, hlv_stds, n_val_events)
scores_anom, vecs_a, hlvs_a = evaluate_anom(hlv_means, hlv_stds, n_anom_events)
print("Evaluating Time:", time.time()-eval_time)

scores_normal = 1-np.exp(np.multiply(scores_normal, -1))
scores_val = 1-np.exp(np.multiply(scores_val, -1))
scores_anom = 1-np.exp(np.multiply(scores_anom, -1))

scores_all = scores_normal
scores_all = np.append(scores_all, scores_anom)
scores_all_val = scores_val
scores_all_val = np.append(scores_all_val, scores_anom)

labels = np.append(np.zeros(len(scores_normal)), np.ones(len(scores_anom)))
labels_val = np.append(np.zeros(len(scores_val)), np.ones(len(scores_anom)))

fpr, tpr, _ = roc_curve(labels, scores_all)
fpr_v, tpr_v, _ = roc_curve(labels_val, scores_all_val)

roc_auc = auc(fpr, tpr) # compute area under the curve
roc_auc_v = auc(fpr_v, tpr_v) # compute area under the curve

print(np.shape(vecs_t))
print(vecs_t[0])
print(np.shape(hlvs_t))
print(hlvs_t[0])
print(np.shape(scores_normal))

t_j_pt = []
t_j_m = []
t_j_eta = []
t_j_phi = []
t_j_score = []
t_j_c2 = []
t_j_d2 = []
t_j_t1 = []
t_j_t2 = []
t_j_t3 = []
t_j_t21 = []
t_j_t32 = []
t_j_t31 = []
t_j_s12 = []
t_j_s23 = []
for i in range(len(scores_normal)):
  t_j_pt.append(vecs_t[i][0]) 
  t_j_eta.append(vecs_t[i][1]) 
  t_j_phi.append(vecs_t[i][2]) 
  t_j_m.append(vecs_t[i][3]) 
  t_j_score.append(scores_normal[i]) 
  t_j_c2.append(hlvs_t[i][0]) 
  t_j_d2.append(hlvs_t[i][1]) 
  t_j_t1.append(hlvs_t[i][2]) 
  t_j_t2.append(hlvs_t[i][3]) 
  t_j_t3.append(hlvs_t[i][4]) 
  t_j_t21.append(hlvs_t[i][5]) 
  t_j_t32.append(hlvs_t[i][6]) 
  t_j_t31.append(hlvs_t[i][7]) 
  t_j_s12.append(hlvs_t[i][8]) 
  t_j_s23.append(hlvs_t[i][9]) 

t_j_pt  = np.array(t_j_pt)
t_j_m  = np.array(t_j_m)
t_j_eta  = np.array(t_j_eta)
t_j_phi  = np.array(t_j_phi)
t_j_score = np.array(t_j_score)
t_j_c2  = np.array(t_j_c2)
t_j_d2  = np.array(t_j_d2)
t_j_t1  = np.array(t_j_t1)
t_j_t2  = np.array(t_j_t2)
t_j_t3  = np.array(t_j_t3)
t_j_t21  = np.array(t_j_t21)
t_j_t32  = np.array(t_j_t32)
t_j_t31  = np.array(t_j_t31)
t_j_s12  = np.array(t_j_s12)
t_j_s23  = np.array(t_j_s23)

t_save_data = []
t_save_data.append(t_j_pt)
t_save_data.append(t_j_m)
t_save_data.append(t_j_eta)
t_save_data.append(t_j_phi)
t_save_data.append(t_j_score)
t_save_data.append(t_j_c2)
t_save_data.append(t_j_d2)
t_save_data.append(t_j_t1)
t_save_data.append(t_j_t2)
t_save_data.append(t_j_t3)
t_save_data.append(t_j_t21)
t_save_data.append(t_j_t32)
t_save_data.append(t_j_t31)
t_save_data.append(t_j_s12)
t_save_data.append(t_j_s23)

t_save_data = np.array(t_save_data)
np.save("eval_data/"+proc+suffix+"_"+sample+"_"+sample+"_Weights_Leading_ConstOnly_Eval_Data.npy", t_save_data)

v_j_pt = []
v_j_m = []
v_j_eta = []
v_j_phi = []
v_j_score = []
v_j_c2 = []
v_j_d2 = []
v_j_t1 = []
v_j_t2 = []
v_j_t3 = []
v_j_t21 = []
v_j_t32 = []
v_j_t31 = []
v_j_s12 = []
v_j_s23 = []
for i in range(len(scores_val)):
  v_j_pt.append(vecs_v[i][0]) 
  v_j_eta.append(vecs_v[i][1]) 
  v_j_phi.append(vecs_v[i][2]) 
  v_j_m.append(vecs_v[i][3]) 
  v_j_score.append(scores_val[i]) 
  v_j_c2.append(hlvs_v[i][0]) 
  v_j_d2.append(hlvs_v[i][1]) 
  v_j_t1.append(hlvs_v[i][2]) 
  v_j_t2.append(hlvs_v[i][3]) 
  v_j_t3.append(hlvs_v[i][4]) 
  v_j_t21.append(hlvs_v[i][5]) 
  v_j_t32.append(hlvs_v[i][6]) 
  v_j_t31.append(hlvs_v[i][7]) 
  v_j_s12.append(hlvs_v[i][8]) 
  v_j_s23.append(hlvs_v[i][9]) 

v_j_pt  = np.array(v_j_pt)
v_j_m  = np.array(v_j_m)
v_j_eta  = np.array(v_j_eta)
v_j_phi  = np.array(v_j_phi)
v_j_score = np.array(v_j_score)
v_j_c2  = np.array(v_j_c2)
v_j_d2  = np.array(v_j_d2)
v_j_t1  = np.array(v_j_t1)
v_j_t2  = np.array(v_j_t2)
v_j_t3  = np.array(v_j_t3)
v_j_t21  = np.array(v_j_t21)
v_j_t32  = np.array(v_j_t32)
v_j_t31  = np.array(v_j_t31)
v_j_s12  = np.array(v_j_s12)
v_j_s23  = np.array(v_j_s23)

v_save_data = []
v_save_data.append(v_j_pt)
v_save_data.append(v_j_m)
v_save_data.append(v_j_eta)
v_save_data.append(v_j_phi)
v_save_data.append(v_j_score)
v_save_data.append(v_j_c2)
v_save_data.append(v_j_d2)
v_save_data.append(v_j_t1)
v_save_data.append(v_j_t2)
v_save_data.append(v_j_t3)
v_save_data.append(v_j_t21)
v_save_data.append(v_j_t32)
v_save_data.append(v_j_t31)
v_save_data.append(v_j_s12)
v_save_data.append(v_j_s23)

v_save_data = np.array(v_save_data)
np.save("eval_data/"+proc+suffix+"_Background_"+sample+"_Weights_Leading_ConstOnly_Eval_Data.npy", v_save_data)

a_j_pt = []
a_j_m = []
a_j_eta = []
a_j_phi = []
a_j_score = []
a_j_c2 = []
a_j_d2 = []
a_j_t1 = []
a_j_t2 = []
a_j_t3 = []
a_j_t21 = []
a_j_t32 = []
a_j_t31 = []
a_j_s12 = []
a_j_s23 = []
for i in range(len(scores_anom)):
  a_j_pt.append(vecs_a[i][0]) 
  a_j_eta.append(vecs_a[i][1]) 
  a_j_phi.append(vecs_a[i][2]) 
  a_j_m.append(vecs_a[i][3]) 
  a_j_score.append(scores_anom[i]) 
  a_j_c2.append(hlvs_a[i][0]) 
  a_j_d2.append(hlvs_a[i][1]) 
  a_j_t1.append(hlvs_a[i][2]) 
  a_j_t2.append(hlvs_a[i][3]) 
  a_j_t3.append(hlvs_a[i][4]) 
  a_j_t21.append(hlvs_a[i][5]) 
  a_j_t32.append(hlvs_a[i][6]) 
  a_j_t31.append(hlvs_a[i][7]) 
  a_j_s12.append(hlvs_a[i][8]) 
  a_j_s23.append(hlvs_a[i][9]) 

a_j_pt  = np.array(a_j_pt)
a_j_m  = np.array(a_j_m)
a_j_eta  = np.array(a_j_eta)
a_j_phi  = np.array(a_j_phi)
a_j_score = np.array(a_j_score)
a_j_c2  = np.array(a_j_c2)
a_j_d2  = np.array(a_j_d2)
a_j_t1  = np.array(a_j_t1)
a_j_t2  = np.array(a_j_t2)
a_j_t3  = np.array(a_j_t3)
a_j_t21  = np.array(a_j_t21)
a_j_t32  = np.array(a_j_t32)
a_j_t31  = np.array(a_j_t31)
a_j_s12  = np.array(a_j_s12)
a_j_s23  = np.array(a_j_s23)

a_save_data = []
a_save_data.append(a_j_pt)
a_save_data.append(a_j_m)
a_save_data.append(a_j_eta)
a_save_data.append(a_j_phi)
a_save_data.append(a_j_score)
a_save_data.append(a_j_c2)
a_save_data.append(a_j_d2)
a_save_data.append(a_j_t1)
a_save_data.append(a_j_t2)
a_save_data.append(a_j_t3)
a_save_data.append(a_j_t21)
a_save_data.append(a_j_t32)
a_save_data.append(a_j_t31)
a_save_data.append(a_j_s12)
a_save_data.append(a_j_s23)

a_save_data = np.array(a_save_data)
np.save("eval_data/"+proc+suffix+"_"+sample.replace("Contaminated", "Signal")+"_"+sample+"_Weights_Leading_ConstOnly_Eval_Data.npy", a_save_data)

print("Epoch: "+str(epoch)+": Max "+str(maxconsts)+" Constituents Evaluation:")
print("Training AUC:", roc_auc)
print("Validation AUC:", roc_auc_v)
