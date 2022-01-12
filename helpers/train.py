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
          #print(n_c, i, jetgroup[n_c+"/hlvs"][(i)])
    all_hlvs = np.array(all_hlvs)
    hlv_means = np.mean(all_hlvs, axis=0)
    hlv_stds = np.std(all_hlvs, axis=0)
    for n_c in jetgroup.keys():
      if(len(jetgroup[n_c+"/constituents"][()]) > 0):
        tmp_consts = jetgroup[n_c+"/constituents"][()]
        data.update({n_c: tmp_consts})
        avg_jets.update({n_c: np.mean(tmp_consts, axis=0)[:,0]})
      if(len(jetgroup[n_c+"/hlvs"][()]) > 0):
        hlvs.update({n_c: (jetgroup[n_c+"/hlvs"][()] - hlv_means)/hlv_stds})
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
        hlvs.update({n_c: (jetgroup[n_c+"/hlvs"][()] - hlv_means)/hlv_stds})
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
        hlvs.update({n_c: (jetgroup[n_c+"/hlvs"][()] - hlv_means)/hlv_stds})
      else:
        hlvs.update({n_c: jetgroup[n_c+"/hlvs"]})
      vecs.update({n_c: jetgroup[n_c+"/4vecs"]}) 
    n_events = len(infile["events"].keys())
    return data, hlvs, n_events, vecs

def train(epoch):
  train_loss = 0
  total_batches = 0
  for n_c in data_train.keys():
    if(int(n_c.replace("C","")) in const_list):
      start = time.time()
      n_jets = len(data_train[n_c])
      total_batches += n_jets/batch_size
      batch_counter = 0
      avgs = avg_jets[n_c]
      while(batch_counter < n_jets and batch_counter < max_batches*batch_size):
        batch_start = time.time()
        batch_step = min(batch_size, n_jets - batch_counter)
        data = torch.tensor(data_train[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        hlvs = torch.tensor(hlvs_train[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        data = Variable(data.transpose(0, 1))
        optimizer.zero_grad()
        kld_loss, nll_loss, loss, y_mean = model(data, hlvs, avgs, kl_weight)
        #kld_loss, nll_loss, loss, y_mean = model(data)
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        train_loss += loss.data
        batch_counter += batch_step

      
  print('====> Epoch: {} Average loss: {:.4f}'.format(
    epoch, train_loss / (max_batches)))
    #epoch, train_loss / (total_batches)))
  train_loss /= total_batches

  return train_loss.cpu().numpy()

def test(epoch):
  v_loss = 0
  total_batches = 0
  for n_c in data_val.keys():
    if(int(n_c.replace("C","")) in const_list):
      n_jets = len(data_val[n_c])
      total_batches += n_jets/batch_size
      batch_counter = 0
      avgs = avg_jets[n_c]
      #for i in range(len(batched_val)):
      while(batch_counter < n_jets):
        batch_step = min(batch_size, n_jets - batch_counter)
        data = torch.tensor(data_val[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        hlvs = torch.tensor(hlvs_val[n_c][batch_counter:batch_counter+batch_step]).float().cuda()
        data = Variable(data.transpose(0, 1))
        kld_loss, nll_loss, loss, y_mean = model(data, hlvs, avgs, kl_weight)
        #kld_loss, nll_loss, loss, y_mean = model(data)
        v_loss += loss.data
        batch_counter += batch_step

  v_loss /= total_batches 
  return v_loss

def evaluate_training(hlv_means, hlv_stds, n_events):
  kld_losses = []
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

        batch_counter += batch_step

  return kld_losses

def evaluate_validation(hlv_means, hlv_stds, n_events):
  kld_losses = []
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

        batch_counter += batch_step

  return kld_losses

def evaluate_anom(hlv_means, hlv_stds, n_events):
  kld_losses = []
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

        batch_counter += batch_step

  return kld_losses




def train(sample, topN, maxconsts, kl_weight, h_dim, z_dim):

  #hyperparameters
  x_dim = 3
  hlv_dim = 10
  n_layers =  1
  n_epochs  = 100
  clip = 10
  learning_rate = 1e-5
  l2_norm = 0
  batch_size = 256
  seed = 128
  print_every = 100
  save_every = 1
  eval_every = 10
  max_batches = 500000
  plot_every = 20
  train_hlvs = False

  load = bool(int(sys.argv[5]))

  dataDir = 'Output_h5/'
  
  
  
  if(train_hlvs):
    train_name = "lhco_train_"+sample+"_"+str(maxconsts)+"_"+proc+"_"+"top"+topN+suffix+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)
  else:
    train_name = "lhco_train_"+sample+"_"+str(maxconsts)+"_"+proc+"_"+"top"+topN+suffix+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)+"_constonly"
  
  if not os.path.exists(sys.path[0]+"/plots/"+train_name):
    try:
      os.makedirs(sys.path[0]+"/plots/"+train_name)
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/roc")
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/scores")
      os.makedirs(sys.path[0]+"/plots/"+train_name+"/trends")
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise
  
  if not os.path.exists(sys.path[0]+"/saves"):
    try:
      os.makedirs(sys.path[0]+"/saves")
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        raise



  
  roc_trend = []
  roc_v_trend = []
  
  print("CUDA Available:", torch.cuda.is_available())
  print("CUDA Version:", torch.version.cuda)
  print("CUDA Device:", torch.cuda.get_device_name())
  torch.cuda.set_device(cuda_idx)
  print("CUDA Index:", torch.cuda.current_device())
  
  
  #manual seed
  torch.manual_seed(seed)
  #plt.ion()
  
  data_train, hlvs_train, hlv_means, hlv_stds, n_train_events, avg_jets, vecs_train = get_data()
  data_val, hlvs_val, n_val_events, vecs_val = get_val_data(hlv_means, hlv_stds)
  data_anom, hlvs_anom, n_anom_events, vecs_anom = get_anom_data(hlv_means, hlv_stds)
  
  print(data_train.keys())
  const_list = range(3, maxconsts+1)
  
  losses_train = []
  losses_val = []
  model = VRNN(x_dim, hlv_dim, h_dim, z_dim, n_layers, train_hlvs)
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
      l_anom = 0 
      #training + testing
      start_time = time.time()
      model.train()
      l_train = train(epoch)
      model.eval()
      l_val = test(epoch)
      print("Time: ", time.time() - start_time)
      losses_train.append(l_train)
      losses_val.append(l_val)
      
      if(epoch % eval_every == 0):
        eval_time = time.time()
        scores_normal = evaluate_training(hlv_means, hlv_stds, n_train_events)
        scores_val = evaluate_validation(hlv_means, hlv_stds, n_val_events)
        scores_anom = evaluate_anom(hlv_means, hlv_stds, n_anom_events)
  
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
  
        np.save("plots/"+train_name+"/roc/fpr_train_all_"+train_name+"_epoch_"+str(epoch)+".npy", fpr)
        np.save("plots/"+train_name+"/roc/tpr_train_all_"+train_name+"_epoch_"+str(epoch)+".npy", tpr)
        np.save("plots/"+train_name+"/roc/fpr_v_all_"+train_name+"_epoch_"+str(epoch)+".npy", fpr_v)
        np.save("plots/"+train_name+"/roc/tpr_v_all_"+train_name+"_epoch_"+str(epoch)+".npy", tpr_v)
  
        np.save("plots/"+train_name+"/scores/scores_normal_all_"+train_name+"_epoch_"+str(epoch)+".npy", scores_normal)
        np.save("plots/"+train_name+"/scores/scores_val_all_"+train_name+"_epoch_"+str(epoch)+".npy", scores_val)
        np.save("plots/"+train_name+"/scores/scores_anom_all_"+train_name+"_epoch_"+str(epoch)+".npy", scores_anom)
  
        roc_trend.append(roc_auc)
        roc_v_trend.append(roc_auc_v)
  
        np.save("plots/"+train_name+"/trends/roc_t_train_scratch_"+proc+"_"+sample+"_Top"+topN+"_"+str(maxconsts)+"_ep"+str(epoch)+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)+".npy", roc_trend)
        np.save("plots/"+train_name+"/trends/roc_v_train_scratch_"+proc+"_"+sample+"_Top"+topN+"_"+str(maxconsts)+"_ep"+str(epoch)+"_"+str(kl_weight).replace(".","p")+"_"+str(h_dim)+"_"+str(z_dim)+".npy", roc_v_trend)
  
        print("Epoch: "+str(epoch)+": Max "+str(maxconsts)+" Constituents Evaluation:")
        print("Training AUC:", roc_auc)
        print("Validation AUC:", roc_auc_v)
  
      #saving model
      if epoch % save_every == 0:
        fn = 'saves/vrnn_state_dict_'+train_name+'_epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)
  
  
    #Save training losses
    np.save("plots/"+train_name+"/trends/losses_train_"+train_name+".npy", losses_train)
    np.save("plots/"+train_name+"/trends/losses_val_"+train_name+".npy", losses_val)
