import numpy as np
import sys
import os

def make_eval_data(sample_set, vecs_dict, hlvs_dict, scores, const_list, train_name):

    if not os.path.exists(sys.path[0]+"/eval_data"):
      try:
        os.makedirs(sys.path[0]+"/eval_data")
      except OSError as exc:
        if exc.errno != errno.EEXIST:
          raise


    vecs = []
    hlvs = []
    
    for c in const_list:
        n_c = str(c)+"C"

        if len(vecs_dict[n_c]) > 0:
            if len(vecs) == 0: vecs = vecs_dict[n_c]
            else: vecs = np.concatenate((vecs, vecs_dict[n_c]), axis=0)
        if len(hlvs_dict[n_c]) > 0:
            if len(hlvs) == 0: hlvs = hlvs_dict[n_c]
            else: hlvs = np.concatenate((hlvs, hlvs_dict[n_c]), axis=0)
             
    j_pt = []
    j_m = []
    j_eta = []
    j_phi = []
    j_score = []
    j_c2 = []
    j_d2 = []
    j_t1 = []
    j_t2 = []
    j_t3 = []
    j_t21 = []
    j_t32 = []
    j_t31 = []
    j_s12 = []
    j_s23 = []
    for i in range(len(scores)):
        j_pt.append(vecs[i][0]) 
        j_eta.append(vecs[i][1]) 
        j_phi.append(vecs[i][2]) 
        j_m.append(vecs[i][3]) 
        j_score.append(scores[i]) 
        j_c2.append(hlvs[i][0]) 
        j_d2.append(hlvs[i][1]) 
        j_t1.append(hlvs[i][2]) 
        j_t2.append(hlvs[i][3]) 
        j_t3.append(hlvs[i][4]) 
        j_t21.append(hlvs[i][5]) 
        j_t32.append(hlvs[i][6]) 
        j_t31.append(hlvs[i][7]) 
        j_s12.append(hlvs[i][8]) 
        j_s23.append(hlvs[i][9]) 

    j_pt = np.array(j_pt)
    j_m  = np.array(j_m)
    j_eta  = np.array(j_eta)
    j_phi  = np.array(j_phi)
    j_score = np.array(j_score)
    j_c2  = np.array(j_c2)
    j_d2  = np.array(j_d2)
    j_t1  = np.array(j_t1)
    j_t2  = np.array(j_t2)
    j_t3  = np.array(j_t3)
    j_t21  = np.array(j_t21)
    j_t32  = np.array(j_t32)
    j_t31  = np.array(j_t31)
    j_s12  = np.array(j_s12)
    j_s23  = np.array(j_s23)

    save_data = []
    save_data.append(j_pt)
    save_data.append(j_m)
    save_data.append(j_eta)
    save_data.append(j_phi)
    save_data.append(j_score)
    save_data.append(j_c2)
    save_data.append(j_d2)
    save_data.append(j_t1)
    save_data.append(j_t2)
    save_data.append(j_t3)
    save_data.append(j_t21)
    save_data.append(j_t32)
    save_data.append(j_t31)
    save_data.append(j_s12)
    save_data.append(j_s23)

    save_data = np.array(save_data)
    np.save(f"eval_data/{sample_set.lower()}_{train_name}_eval_data.npy", save_data)
    print(f"Saved {sample_set} Evaluation Data to: eval_data/{sample_set.lower()}_{train_name}_eval_data.npy")
            
def get_eval_data(filename):
    
    eval_data = np.load(filename)
        
    jet_dict = dict()

    jet_dict["pt"] = eval_data[0]
    jet_dict["m"] = eval_data[1]
    jet_dict["eta"] = eval_data[2]
    jet_dict["phi"] = eval_data[3]
    jet_dict["score"] = eval_data[4]
    jet_dict["c2"] = eval_data[5]
    jet_dict["d2"] = eval_data[6]
    jet_dict["t1"] = eval_data[7]
    jet_dict["t2"] = eval_data[8]
    jet_dict["t3"] = eval_data[9]
    jet_dict["t21"] = eval_data[10]
    jet_dict["t32"] = eval_data[11]
    jet_dict["t31"] = eval_data[12]
    jet_dict["s12"] = eval_data[13]
    jet_dict["s23"] = eval_data[14]

    return jet_dict
