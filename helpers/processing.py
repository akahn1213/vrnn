import numpy as np
from skhep.math.vectors import *
import h5py
import os
import copy

data_dir=""
out_dir = "Output_h5/"

samples_dict = {
  "2Prong_Contaminated": ["unprocessed_data/events_anomalydetection_VRNN_2Prong_Contaminated.hdf5"],
  "3Prong_Contaminated": ["unprocessed_data/events_anomalydetection_VRNN_3Prong_Contaminated.hdf5"],
  "2Prong_Signal": ["unprocessed_data/events_anomalydetection_VRNN_2Prong_Signal.hdf5"],
  "3Prong_Signal": ["unprocessed_data/events_anomalydetection_VRNN_3Prong_Signal.hdf5"],
  "Background": ["unprocessed_data/events_anomalydetection_VRNN_Background.hdf5"],
}

if not os.path.exists(f"{sys.path[0]}/{out_dir}"):
  try:
    os.makedirs(f"{sys.path[0]}/{out_dir}")
  except OSError as exc:
    if exc.errno != errno.EEXIST:
      raise

n_consts_min = 3

def process(n_consts_max):

  for sample_name in samples_dict.keys():
    if os.path.exists(f"{out_dir}events_anomalydetection_{sample_name}_VRNN_{n_consts_max}C_preprocessed.hdf5"):
      os.remove(f"{out_dir}events_anomalydetection_{sample_name}_VRNN_{n_consts_max}C_preprocessed.hdf5")
  
    sorted_constituents_list = []
    sorted_hlvs_list = []
    sorted_4vecs_list = []
    for i in range(10):
      sorted_constituents={}
      sorted_hlvs={}
      sorted_4vecs={}
      for j in range(n_consts_min, n_consts_max+1):
        sorted_constituents.update({str(j):[]})
        sorted_hlvs.update({str(j):[]})
        sorted_4vecs.update({str(j):[]})
      sorted_constituents_list.append(sorted_constituents)
      sorted_hlvs_list.append(sorted_hlvs)
      sorted_4vecs_list.append(sorted_4vecs)
  
    jets_processed = 0
    for sample in samples_dict[sample_name]:
      print("Starting Pre-Processing:", sample)
  
      infile = h5py.File(data_dir+sample, "r")
  
  
      data = infile["constituents"]
      hlvs = infile["hlvs"]
  
      print(len(data), "Events")
  
      event_counter = 0
      event_counter_unprocessed = 0
      for eventnum in range(len(data)):
  
        event_counter_unprocessed += 1
        jet_counter = 0
        jet_counter_unprocessed = 0
        good_event = False
  
        for jn in range(2): #num large R jets, zero padded
         
          try: d = data[eventnum][jn][0] #large R jet 4 vector
          except ValueError: continue 
         
          if (d[0] < 1e-10):  #down into the zero padded
            continue  
          jet_full = LorentzVector()
          jet_full.setptetaphim(d[0], d[1], d[2], d[3])
  
          ######################################################
          #                   Pre-Selection                    #
          ######################################################
  
  
          if(jet_full.eta > 2.): continue
  
  
  
          hlv = hlvs[eventnum][jn].tolist()
          jet4vec = d
  
          if(np.isnan(hlv).any()): continue
  
  
          ######################################################
          ######################################################
          ##                    JET LOOP                      ##
          ######################################################
          ######################################################
  
  
  
  
  
          ######################################################
          #               Build unprocessed jet                #
          ######################################################
          jet = LorentzVector()
          jet.setptetaphim(0, 0, 0, 0)
          for i in range(1, 51):
            c = data[eventnum][jn][i] #Current constituent
            if(c[0] < 1e-15): 
              last_const = i
              break #Stop after no more constituents (only zeros)
            cst = LorentzVector()
            cst.setptetaphim(c[0], c[1], c[2], 0)
            jet += cst
    
          
  
  
  
  
          ######################################################
          #             Rotate in Phi, Boost in Z              #
          ######################################################
          phi = jet.phi()
          bv = jet.boostvector
          bv.x = 0
          bv.y = 0
  
          jet_1d = jet.rotatez(-phi)
          jet_1d = jet_1d.boost(bv)
          if(jet_1d.m == 0.0 or jet_1d.pt == 0.0): 
            print('eventNum = ', eventnum)
            print('jet ' + str(jet_counter) + ' has 0 mass, BREAKING')
            print('jet four vector: ' , d)
            print('Constituents: ')
            for i in range(1, n_consts_max+1):
              c = data[eventnum][jn][i] #Current constituent
              if(c[0] > 1e-15): print(c)
            continue
  
          ######################################################
          #         Rescale Mass, Boost to Ref Energy          #
          ######################################################
  
          m_zero = 0.25
          e_zero = 1.
  
          m_rescale = m_zero/jet_1d.m
          jet_1d = m_rescale*jet_1d #Rescale Mass
  
          tvec_ref = LorentzVector()
          p_ref = jet_1d.vector
          p_ref = (np.sqrt(  (np.abs(np.square(e_zero) - np.square(jet_1d.m)))/(np.abs(np.square(jet_1d.e) - np.square(jet_1d.m))) ) )*p_ref
          p_ref = Vector3D(p_ref[0], p_ref[1], p_ref[2])
          try: tvec_ref.setptetaphim(p_ref.x, jet_1d.eta, jet_1d.phi(), jet_1d.m)
          except ValueError: continue
  
          bp_x = tvec_ref.boostvector.x
          if bp_x == 0:
            print('bp_x == 0')
            continue 
  
  
          beta = (jet_1d.e - (jet_1d.px/bp_x))/(jet_1d.p - (jet_1d.e/bp_x))
  
          jet_1d = jet_1d.boost(beta, 0, 0) #Boost jet to reference energy
  
  
          ######################################################
          #     Rotate about x so eta of first const = 0       #
          ######################################################
  
  
          alpha = 0
          for i in range(1, 51):
            c = data[eventnum][jn][i] #Current constituent
            if(c[0] < 1e-15): 
              last_const = i
              break #Stop after no more constituents (only zeros)
            cst = LorentzVector()
            cst.setptetaphim(c[0], c[1], c[2], 0)
            cst = cst.rotatez(-phi)
            cst = cst.boost(bv)
            cst = m_rescale*cst
            cst = cst.boost(beta, 0, 0)
            if(i == 1): 
              alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
            cst = cst.rotatex(alpha - np.pi/2)
            
          ######################################################
          #               Apply Delta R < 1 cut                #
          ######################################################
  
  
          fix_dr = False
          jet_drcut = LorentzVector()
          jet_drcut.setptetaphim(0, 0, 0, 0)
          consts_drcut = []
          for i in range(1, 51):
            c = data[eventnum][jn][i] #Current constituent
            if(c[0] < 1e-15): 
              last_const = i
              break #Stop after no more constituents (only zeros)
            cst = LorentzVector()
            cst.setptetaphim(c[0], c[1], c[2], 0)
            cst = cst.rotatez(-phi)
            cst = cst.boost(bv)
            cst = m_rescale*cst
            cst = cst.boost(beta, 0, 0)
            if(i == 1): 
              alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
            cst = cst.rotatex(alpha - np.pi/2)
            dr = np.sqrt(cst.eta**2 + cst.phi()**2)
            if(dr < 1): 
              consts_drcut.append([cst.pt, cst.eta, cst.phi()])
              jet_drcut += cst
            else: 
              fix_dr = True #Need to fix if a const dr is > 1
  
  
  
          kts = []
          for c in consts_drcut:
            kts.append(np.power(c[0], 2)*(c[1]**2 + c[2]**2))
          kt_indeces = np.argsort(kts)[::-1]
  
          consts_kt = []
          for idx in kt_indeces:
            consts_kt.append(consts_drcut[idx])
  
            
          skip_jet = False
          while(fix_dr):    
            fix_dr = False
            if(len(consts_kt) < n_consts_min):
              skip_jet = True
              break
  
            ######################################################
            #             Redefine Boost Parameters              #
            ######################################################
  
            phi = jet_drcut.phi()
            bv = jet_drcut.boostvector
            bv.x = 0
            bv.y = 0
  
            jet_1d = jet_drcut.rotatez(-phi)
            jet_1d = jet_1d.boost(bv)
            if(jet_1d.m == 0.0 or jet_1d.pt == 0.0): 
              print('eventNum = ', eventnum)
              print('jet ' + str(jet_counter) + ' has 0 mass, BREAKING')
              print('jet four vector: ' , d)
              print('Constituents: ')
              for i in range(1, n_consts_max+1):
                c = data[eventnum][jn][i] #Current constituent
                if(c[0] > 1e-15): print(c)
              continue
  
            m_rescale = m_zero/jet_1d.m
            jet_1d = m_rescale*jet_1d #Rescale Mass
  
            tvec_ref = LorentzVector()
            p_ref = jet_1d.vector
            p_ref = (np.sqrt(  (np.abs(np.square(e_zero) - np.square(jet_1d.m)))/(np.abs(np.square(jet_1d.e) - np.square(jet_1d.m))) ) )*p_ref
            p_ref = Vector3D(p_ref[0], p_ref[1], p_ref[2])
            try: tvec_ref.setptetaphim(p_ref.x, jet_1d.eta, jet_1d.phi(), jet_1d.m)
            except ValueError: continue
  
            bp_x = tvec_ref.boostvector.x
            if bp_x == 0:
              print('bp_x == 0')
              continue 
  
  
            beta = (jet_1d.e - (jet_1d.px/bp_x))/(jet_1d.p - (jet_1d.e/bp_x))
  
            jet_1d = jet_1d.boost(beta, 0, 0) #Boost jet to reference energy
  
  
  
            ######################################################
            #               Apply fixed parameters               #
            ######################################################
  
  
            consts_test = copy.deepcopy(consts_kt)
            consts_kt = []
            jet_drcut = LorentzVector()
            jet_drcut.setptetaphim(0, 0, 0, 0)
            for i in range(len(consts_test)):
              c = consts_test[i]
              cst = LorentzVector()
              cst.setptetaphim(c[0], c[1], c[2], 0)
              cst = cst.rotatez(-phi)
              cst = cst.boost(bv)
              cst = m_rescale*cst
              cst = cst.boost(beta, 0, 0)
              if(i == 0): 
                alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
              cst = cst.rotatex(alpha - np.pi/2)
              dr = np.sqrt(cst.eta**2 + cst.phi()**2)
              if(dr < 1):
                consts_kt.append([cst.pt, cst.eta, cst.phi()])
                jet_drcut += cst
              else: fix_dr = True
              
  
          ######################################################
          #               Apply Max Consts cut                 #
          ######################################################
  
          if(skip_jet): continue
  
  
          consts_final = []
  
          if(len(consts_kt) > n_consts_max): #Need to apply max consts cut
            jet_max_cut = LorentzVector()
            jet_max_cut.setptetaphim(0, 0, 0, 0)
            consts_max_cut = consts_kt[0:n_consts_max]
            for c in consts_max_cut:
              cst = LorentzVector()
              cst.setptetaphim(c[0], c[1], c[2], 0)
              jet_max_cut += cst
              
            ######################################################
            #             Reboost after max consts               #
            ######################################################
  
            phi = jet_max_cut.phi()
            bv = jet_max_cut.boostvector
            bv.x = 0
            bv.y = 0
  
            jet_1d = jet_max_cut.rotatez(-phi)
            jet_1d = jet_1d.boost(bv)
            if(jet_1d.m == 0.0 or jet_1d.pt == 0.0): 
              print('eventNum = ', eventnum)
              print('jet ' + str(jet_counter) + ' has 0 mass, BREAKING')
              print('jet four vector: ' , d)
              print('Constituents: ')
              for i in range(1, n_consts_max+1):
                c = data[eventnum][jn][i] #Current constituent
                if(c[0] > 1e-15): print(c)
              continue
  
            m_rescale = m_zero/jet_1d.m
            jet_1d = m_rescale*jet_1d #Rescale Mass
  
            tvec_ref = LorentzVector()
            p_ref = jet_1d.vector
            p_ref = (np.sqrt(  (np.abs(np.square(e_zero) - np.square(jet_1d.m)))/(np.abs(np.square(jet_1d.e) - np.square(jet_1d.m))) ) )*p_ref
            p_ref = Vector3D(p_ref[0], p_ref[1], p_ref[2])
            try: tvec_ref.setptetaphim(p_ref.x, jet_1d.eta, jet_1d.phi(), jet_1d.m)
            except ValueError: continue
  
            bp_x = tvec_ref.boostvector.x
            if bp_x == 0:
              print('bp_x == 0')
              continue 
  
  
            beta = (jet_1d.e - (jet_1d.px/bp_x))/(jet_1d.p - (jet_1d.e/bp_x))
  
            jet_1d = jet_1d.boost(beta, 0, 0) #Boost jet to reference energy
  
  
  
            ######################################################
            #               Apply fixed parameters               #
            ######################################################
  
  
            fix_dr = False
            consts_test = copy.deepcopy(consts_max_cut)
            consts_max_cut = []
            jet_max_cut = LorentzVector()
            jet_max_cut.setptetaphim(0, 0, 0, 0)
            for i in range(len(consts_test)):
              c = consts_test[i]
              cst = LorentzVector()
              cst.setptetaphim(c[0], c[1], c[2], 0)
              cst = cst.rotatez(-phi)
              cst = cst.boost(bv)
              cst = m_rescale*cst
              cst = cst.boost(beta, 0, 0)
              if(i == 0): 
                alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
              cst = cst.rotatex(alpha - np.pi/2)
              dr = np.sqrt(cst.eta**2 + cst.phi()**2)
              if(dr < 1):
                consts_max_cut.append([cst.pt, cst.eta, cst.phi()])
                jet_max_cut += cst
              else: fix_dr = True
  
            while(fix_dr):    
              fix_dr = False
              if(len(consts_max_cut) < n_consts_min):
                skip_jet=True
                break
  
              ######################################################
              #             Redefine Boost Parameters              #
              ######################################################
  
              phi = jet_max_cut.phi()
              bv = jet_max_cut.boostvector
              bv.x = 0
              bv.y = 0
  
              jet_1d = jet_max_cut.rotatez(-phi)
              jet_1d = jet_1d.boost(bv)
              if(jet_1d.m == 0.0 or jet_1d.pt == 0.0): 
                print('eventNum = ', eventnum)
                print('jet ' + str(jet_counter) + ' has 0 mass, BREAKING')
                print('jet four vector: ' , d)
                print('Constituents: ')
                for i in range(1, n_consts_max+1):
                  c = data[eventnum][jn][i] #Current constituent
                  if(c[0] > 1e-15): print(c)
                continue
  
              m_rescale = m_zero/jet_1d.m
              jet_1d = m_rescale*jet_1d #Rescale Mass
  
              tvec_ref = LorentzVector()
              p_ref = jet_1d.vector
              p_ref = (np.sqrt(  (np.abs(np.square(e_zero) - np.square(jet_1d.m)))/(np.abs(np.square(jet_1d.e) - np.square(jet_1d.m))) ) )*p_ref
              p_ref = Vector3D(p_ref[0], p_ref[1], p_ref[2])
              try: tvec_ref.setptetaphim(p_ref.x, jet_1d.eta, jet_1d.phi(), jet_1d.m)
              except ValueError: continue
  
              bp_x = tvec_ref.boostvector.x
              #bp_x = tvec_ref.boostvector.mag
              if bp_x == 0:
                print('bp_x == 0')
                continue 
  
  
              beta = (jet_1d.e - (jet_1d.px/bp_x))/(jet_1d.p - (jet_1d.e/bp_x))
  
              jet_1d = jet_1d.boost(beta, 0, 0) #Boost jet to reference energy
  
  
  
              ######################################################
              #               Apply fixed parameters               #
              ######################################################
  
  
              consts_test = copy.deepcopy(consts_max_cut)
              consts_max_cut = []
              jet_max_cut = LorentzVector()
              jet_max_cut.setptetaphim(0, 0, 0, 0)
              #for i in range(1, n_consts_max+1):
              for i in range(len(consts_test)):
                c = consts_test[i]
                cst = LorentzVector()
                cst.setptetaphim(c[0], c[1], c[2], 0)
                cst = cst.rotatez(-phi)
                cst = cst.boost(bv)
                cst = m_rescale*cst
                cst = cst.boost(beta, 0, 0)
                if(i == 0): 
                  alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
                cst = cst.rotatex(alpha - np.pi/2)
                dr = np.sqrt(cst.eta**2 + cst.phi()**2)
                if(dr < 1):
                  consts_max_cut.append([cst.pt, cst.eta, cst.phi()])
                  jet_max_cut += cst
                else: fix_dr = True
  
            if(skip_jet): continue
            consts_final = consts_max_cut
  
          else:
            consts_final = consts_kt
  
  
          ######################################################
          #                Relative kt Sort                    #
          ######################################################
  
  
          #Find highest pt fraction const
          lead_idx = 0
          lead_pt = 0
          for i in range(len(consts_final)):
            if(consts_final[i][0] > lead_pt):
              lead_idx = i
              lead_pt = consts_final[i][0]
  
  
          #Set Highest pt const to zeroth element
          consts_final[0], consts_final[lead_idx] = consts_final[lead_idx], consts_final[0]
          consts_tmp = consts_final
          consts_final = []
  
          #Re-do rotation
          alpha = 0
          for i in range(len(consts_tmp)):
            c = consts_tmp[i]
            cst = LorentzVector()
            cst.setptetaphim(c[0], c[1], c[2], 0)
            if(i == 0): 
              alpha = np.arctan2(cst.phi(), cst.eta) #Determine rotation angle
            cst = cst.rotatex(alpha - np.pi/2)
            if(dr < 1):
              consts_final.append([cst.pt, cst.eta, cst.phi()])
  
          consts_save = []
          consts_save.append(consts_final[0])
          matched_idxs = []
          matched_idxs.append(0)
          while(len(matched_idxs) < len(consts_final)):
            kt = -999
            idx = -10
            for i in range(len(consts_final)):
              if i in matched_idxs: continue
              c1 = consts_save[-1]
              c2 = consts_final[i]
              kt_tmp = c2[0]*np.sqrt((np.square(c2[1] - c1[1]) + np.square(c2[2] - c1[2])))
              if(kt_tmp > kt):
                kt = kt_tmp
                idx = i
            consts_save.append(consts_final[idx])
            matched_idxs.append(idx)
            
          ######################################################
          #                    Final Jet                       #
          ######################################################
  
          flip=False
          constituents = []
          for i in range(len(consts_save)):
            c = consts_save[i]
            cst = LorentzVector()
            cst.setptetaphim(c[0], c[1], c[2], 0)
            if(i == 1 and cst.eta < 0): 
              flip = True 
            if(flip): cst.setptetaphim(cst.pt, -cst.eta, cst.phi(), cst.m)
            constituents.append([cst.pt/jet_1d.pt, cst.eta, cst.phi()])
  
          n_c = len(constituents)
          if(n_c < n_consts_min): continue
  
          #Add jet to event
          outfile = h5py.File(f"{out_dir}events_anomalydetection_{sample_name}_VRNN_{n_consts_max}C_preprocessed.hdf5", "a")
          outfile.create_dataset(f"events/{event_counter}/jet{jet_counter}/constituents", data=constituents)
          outfile.create_dataset(f"events/{event_counter}/jet{jet_counter}/hlvs", data=hlv)
          outfile.create_dataset(f"events/{event_counter}/jet{jet_counter}/4vec", data=jet4vec)
          outfile.close()  
  
  
          for i in range(10):
            if(n_c > n_consts_max): n_c = n_consts_max
            if(n_c > 0 and i >= jet_counter):
              sorted_constituents_list[i][str(n_c)].append(constituents)
              sorted_hlvs_list[i][str(n_c)].append(hlv)
              sorted_4vecs_list[i][str(n_c)].append(jet4vec)
          
  
          jet_counter += 1
          jets_processed += 1
  
          
          good_event = True
  
        if(good_event): event_counter += 1
        if(event_counter%100 == 0):
          print('Events processed: ', event_counter)
  
    print("Total processed events: ", event_counter)
    print("Total unprocessed events: ", event_counter_unprocessed)
  
    outfile = h5py.File(f"{out_dir}events_anomalydetection_{sample_name}_VRNN_{n_consts_max}C_preprocessed.hdf5", "a")
    for i in range(n_consts_min, n_consts_max+1):
      for j in range(10):
        outfile.create_dataset(f"jets/top{j+1}/{i}C/constituents", data=sorted_constituents_list[j][str(i)])
        outfile.create_dataset(f"jets/top{j+1}/{i}C/hlvs", data=sorted_hlvs_list[j][str(i)])
        outfile.create_dataset(f"jets/top{j+1}/{i}C/4vecs", data=sorted_4vecs_list[j][str(i)])
    outfile.close()  
  
