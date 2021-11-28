import numpy as np
import matplotlib.pyplot as plt

def evaluate_idf1(distmat, q_pids,g_pids, q_camids, g_camids, num_ids_from_id, threshold, threshold_start, threshold_end):
    num_q, num_g = distmat.shape
    
    idf1_return_list = []
    indices = np.argsort(distmat, axis=1)
    idf1_list = []
    x_axis = []
    y_axis = []

    for thold in range(threshold_start, threshold_end):
        idf1_list = []
        nums_under_thold_list = []
        
        
        for i in range(num_q):
            nums_under_thold = np.sum(distmat[i, :] < thold)
            nums_under_thold_list.append(nums_under_thold)

        num_valid_q = 0. # number of valid query
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        #print(matches)
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            raw_cmc = matches[q_idx][
                keep] # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue
            
            num_valid_q += 1.

            raw_cmc = raw_cmc[:(nums_under_thold_list[q_idx])]
            idpt = np.sum(raw_cmc)
            ground_truth = num_ids_from_id[q_pid]
            computed = len(raw_cmc)
            idf1 = (2. * idpt) / (ground_truth + computed) 
            idf1_list.append(idf1)

            if(thold == threshold):
                print("Query id: ", q_pid, " Camera id: ", q_camid, " IDF1: ", idf1, "computed: ", computed," grtruth: ", ground_truth, "idpt: ",idpt)
                
        x_axis.append(thold)
        y_axis.append(sum(idf1_list) / len(idf1_list))
        if (thold == threshold):
            idf1_return_list = idf1_list
        assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    plt.plot(x_axis, y_axis)
    plt.show()
    return idf1_return_list