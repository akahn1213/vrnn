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
import helpers.plotting as plotting
import helpers.eval as eval

import sys, os
from sklearn.metrics import roc_curve, auc

import time

batch_size = 256
const_list = list()
data_dir = 'Output_h5/'

#hyperparameters
x_dim = 3
hlv_dim = 10
n_layers =  1
n_epochs  = 100
clip = 10
learning_rate = 1e-5
l2_norm = 0
seed = 128
print_every = 100
save_every = 10
eval_every = 10
max_batches = 500000
plot_every = 20
train_hlvs = False

def get_data(sample_type, sample_set, n_consts, n_jets, hlv_means=None, hlv_stds=None):

    consts = dict()
    avg_jets = dict()
    hlvs = dict()
    vecs = dict()
    
    sample = f"{sample_type}_{sample_set}"
    if sample_set == "Background":
      sample = sample_set

    with h5py.File(f"{data_dir}events_anomalydetection_{sample}_VRNN_{n_consts}C_preprocessed.hdf5", "r+") as infile:  
      
      jetgroup = infile[f"jets/top{n_jets}"]
      
      if sample_set == "Contaminated":
          all_hlvs = []
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
          consts.update({n_c: tmp_consts})
          avg_jets.update({n_c: np.mean(tmp_consts, axis=0)[:,0]})
        if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
          hlvs.update({n_c: (jetgroup[n_c+"/hlvs"][()] - hlv_means)/hlv_stds})
        else:
          hlvs.update({n_c: jetgroup[n_c+"/hlvs"][()]})
        vecs.update({n_c: jetgroup[n_c+"/4vecs"][()]})
      n_events = len(infile["events"].keys())
    
    return consts, hlvs, vecs, n_events, hlv_means, hlv_stds, avg_jets

def run_model(consts, hlvs, avg_jets, kl_weight, model, optimizer, epoch, train=False):

  global const_list

  kld_losses = []
  run_loss = 0
  total_batches = 0
  for c in const_list:
    n_c = str(c)+"C"
    if not n_c in consts.keys(): continue
    else:
      start = time.time()
      n_jets = len(consts[n_c])
      total_batches += n_jets/batch_size
      batch_counter = 0
      while(batch_counter < n_jets and batch_counter < max_batches*batch_size):
        batch_start = time.time()
        batch_step = min(batch_size, n_jets - batch_counter)
        consts_tensor = torch.tensor(consts[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        hlvs_tensor = torch.tensor(hlvs[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        consts_tensor = Variable(consts_tensor.transpose(0, 1))
        
        if train: 
          optimizer.zero_grad()
        
        kld_loss, nll_loss, loss, y_mean = model(consts_tensor, hlvs_tensor, avg_jets[n_c], kl_weight)
        kld_losses = np.concatenate((kld_losses, kld_loss.data.cpu().numpy()))
        
        if train: 
          loss.backward()       
          optimizer.step()
          nn.utils.clip_grad_norm(model.parameters(), clip)

        run_loss += loss.data
        batch_counter += batch_step

  
  run_loss /= total_batches

  return run_loss.cpu().numpy(), kld_losses

def train(args):

  load = False
  cuda_idx = 0

  train_name = f"train_{args.sample}_{args.n_consts}_top{args.n_jets}_{str(args.kl_weight).replace('.','p')}_{args.h_dim}_{args.z_dim}"
  
  if not os.path.exists(sys.path[0]+"/plots/"+train_name):
    try:
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/roc")
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/scores")
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/trends")
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/evaluation")
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise
  
  if not os.path.exists(sys.path[0]+"/saves"):
    try:
      os.makedirs(sys.path[0]+"/saves")
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise

  
  print("CUDA Available:", torch.cuda.is_available())
  print("CUDA Version:", torch.version.cuda)
  print("CUDA Device:", torch.cuda.get_device_name())
  torch.cuda.set_device(cuda_idx)
  print("CUDA Index:", torch.cuda.current_device())
  
  consts_train, hlvs_train, vecs_train, n_train_events, hlv_means, hlv_stds, avg_jets = get_data(args.sample, "Contaminated", args.n_consts, args.n_jets)
  consts_val, hlvs_val, vecs_val, n_val_events, _, _, _ = get_data(args.sample, "Background", args.n_consts, args.n_jets, hlv_means, hlv_stds)
  consts_anom, hlvs_anom, vecs_anom, n_anom_events, _, _, _ = get_data(args.sample, "Signal", args.n_consts, args.n_jets, hlv_means, hlv_stds)
  
  print(consts_train.keys())
  global const_list
  const_list = range(3, args.n_consts+1)
  
  losses_train = []
  losses_val = []
  roc_trend = []
  roc_v_trend = []
  model = VRNN(x_dim, hlv_dim, args.h_dim, args.z_dim, n_layers, train_hlvs)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_norm)
  do_train=True
  
  if(load):
    fn = 'saves/vrnn_state_dict_'+train_name+'_epoch_100.pth'
    state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
  
  if(do_train):
    for epoch in range(1, n_epochs + 1):
      l_train = 0
      l_val = 0
      #training + testing
      start_time = time.time()
      model.train()
      l_train, scores_train = run_model(consts_train, hlvs_train, avg_jets, args.kl_weight, model, optimizer, epoch, True)
      model.eval()
      with torch.no_grad():
        l_val, scores_val = run_model(consts_val, hlvs_val, avg_jets, args.kl_weight, model, optimizer, epoch)
        l_anom, scores_anom = run_model(consts_anom, hlvs_anom, avg_jets, args.kl_weight, model, optimizer, epoch)
      losses_train.append(l_train)
      losses_val.append(l_val)
      
      print(f" -- Epoch: {epoch} Average loss:")
      print(" -- Training:   {:.4f}".format(l_train))
      print(" -- Validation: {:.4f}".format(l_val))
      print(" -- Anomaly:    {:.4f}".format(l_anom))
      
      if(epoch % eval_every == 0):

        #Evaluate training, validation, signal

        #Transform
        scores_train = 1-np.exp(np.multiply(scores_train, -1))
        scores_val = 1-np.exp(np.multiply(scores_val, -1))
        scores_anom = 1-np.exp(np.multiply(scores_anom, -1))
  
        #Make ROCs
        scores_all = scores_train
        scores_all = np.append(scores_all, scores_anom)
        scores_all_val = scores_val
        scores_all_val = np.append(scores_all_val, scores_anom)
  
        labels = np.append(np.zeros(len(scores_train)), np.ones(len(scores_anom)))
        labels_val = np.append(np.zeros(len(scores_val)), np.ones(len(scores_anom)))
  
        fpr, tpr, _ = roc_curve(labels, scores_all)
        fpr_v, tpr_v, _ = roc_curve(labels_val, scores_all_val)
  
        roc_auc = auc(fpr, tpr) # compute area under the curve
        roc_auc_v = auc(fpr_v, tpr_v) # compute area under the curve

        #Make ROC Plots

        plotting.make_roc_plot(fpr, tpr, fpr_v, tpr_v, roc_auc, roc_auc_v, epoch, train_name)

        roc_trend.append(roc_auc)
        roc_v_trend.append(roc_auc_v)
  
        print(f" ==> Epoch: {epoch}: Max {args.n_consts} Constituents Evaluation:")
        print(" ==> Training AUC:   {:.4f}".format(roc_auc))
        print(" ==> Validation AUC: {:.4f}".format(roc_auc_v))
  
      #saving model
      if epoch % save_every == 0:
        fn = 'saves/vrnn_state_dict_'+train_name+'_epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)
  
    #Save training losses and AUROC trends
    np.save(f"plots/{train_name}/trends/losses_train_{train_name}.npy", losses_train)
    np.save(f"plots/{train_name}/trends/losses_val_{train_name}.npy", losses_val)
    np.save(f"plots/{train_name}/trends/aucs_train_{train_name}.npy", roc_trend)
    np.save(f"plots/{train_name}/trends/aucs_val_{train_name}.npy", roc_v_trend)
    
    eval.make_eval_data("Training", vecs_train, hlvs_train, scores_train, const_list, train_name)
    eval.make_eval_data("Validation", vecs_val, hlvs_val, scores_val, const_list, train_name)
    eval.make_eval_data("Anomaly", vecs_anom, hlvs_anom, scores_anom, const_list, train_name)
