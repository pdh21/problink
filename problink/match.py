import numpy as np
from problink import xidmod
import problink

def link(prior,data_y1,data_y2):
    year_min = np.min([data_y1.year,data_y2.year])
    min_month = np.amin([np.amin(data_y1.dob_j), np.amin(data_y2.dob_j)])
    max_month = np.amax([np.amax(data_y1.dob_j), np.amax(data_y2.dob_j)])
    no_months = max_month - min_month
    Bayes_threshold = -100.0

    match_counter = 0
    Matches = np.empty((11, 30000))
    Matches[:, :] = -99.0
    results_table = np.empty((5, no_months))  # outputs total matches, total correct, total actual, rest is junk

    for m in np.arange(0, no_months):
        month_xid = min_month + m
        age_y1 = year_min - (month_xid / 12.0) - 1900  # age in years

        ind_y1_month = np.where(data_y1.dob_j == month_xid)[0]  # select records from y1 whose DOB is month_xid
        ind_y2_month = np.where(data_y2.dob_j == month_xid)[0]  # select records from y2 whose DOB is month_xid

        B_poss_matches = np.empty((ind_y1_month.size * ind_y2_month.size, 8))
        B_poss_matches[:, :] = 0.0
        problink.drawProgressBar(m/np.float(no_months),barLen=no_months)

        ii = 0


        if ind_y1_month.size >= 1.0 and ind_y2_month.size >= 1.0:
            Nstar_orig = float(np.minimum(ind_y1_month.size, ind_y2_month.size))
            P_o = Nstar_orig / float((ind_y2_month.size * ind_y1_month.size))
            P_o_old = 2.0 * P_o
            for y in range(0, ind_y1_month.size):
                for z in range(0, ind_y2_month.size):
                    # --first check status in each record. If dead in either don't do match----
                    if data_y1.status[ind_y1_month[y]] == 0.0:
                        # calculate Bayes factor for gender
                        data = np.array([data_y1.gender[ind_y1_month[y]], data_y2.gender[ind_y2_month[z]]])
                        B_poss_matches[ii, 0] = xidmod.gender_Bayes(prior.prior_q_H, prior.prior_q_K, prior.bins_q_H, prior.bins_q_K, data)
                        if B_poss_matches[ii, 0] > 0.1 and age_y1 > 2.0:  # can only do BMI and height if genders are the same
                            if data_y1.BMI[ind_y1_month[y]] > 0.1 and data_y2.BMI[ind_y2_month[z]] > 0.1:
                                data = np.array([data_y1.BMI[ind_y1_month[y]], data_y2.BMI[ind_y2_month[z]]])
                                age = np.array([age_y1, age_y1 + 1])
                                B_poss_matches[ii, 1] = xidmod.BMI_Bayes_curve(prior.prior_BMI, prior.bins_percentile_BMI, data,
                                                                               prior.sigma_BMI, age,
                                                                               data_y1.gender[ind_y1_month[y]])
                            if data_y1.hgt[ind_y1_month[y]] > 0.1 and data_y2.hgt[ind_y2_month[z]] > 0.1:
                                data = np.array([data_y1.hgt[ind_y1_month[y]], data_y2.hgt[ind_y2_month[z]]])
                                age = np.array([age_y1 * 12.0, (age_y1 + 1) * 12.0])
                                B_poss_matches[ii, 2] = xidmod.hgt_Bayes_curve(prior.prior_hgt, prior.bins_percentile_hgt, data,
                                                                               prior.sigma_hgt, age,
                                                                               data_y1.gender[ind_y1_month[y]])
                        # calculate Bayes factor for age at diagnosis
                        if data_y1.age_diag[ind_y1_month[y]] > -1.0 and data_y2.age_diag[ind_y2_month[z]] > -1.0:
                            data = np.array([data_y1.age_diag[ind_y1_month[y]], data_y2.age_diag[ind_y2_month[z]]])
                            B_poss_matches[ii, 3] = xidmod.age_dia_Bayes(prior.prior_q_H_age_dia, prior.bins_q_H_age_dia, data)
                        # calculate Bayes factor for genotype
                        data = np.array([data_y1.genotype[:, ind_y1_month[y]], data_y2.genotype[:, ind_y2_month[z]]])
                        B_poss_matches[ii, 4] = xidmod.mutation_Bayes(prior.prior_q_H_mu, prior.mut_pdf, prior.bins_q_H, data)
                        # Add Bayes factors
                        B_poss_matches[ii, 5] = B_poss_matches[ii, 0] + B_poss_matches[ii, 1] + B_poss_matches[ii, 2] + \
                                                B_poss_matches[ii, 3] + B_poss_matches[ii, 4]
                        # Store indices (used for checking actual IDs)
                        B_poss_matches[ii, 6] = y
                        B_poss_matches[ii, 7] = z
                        ii += 1
            B_poss_matches = B_poss_matches[0:ii, :]

            while np.abs((P_o_old - P_o) / P_o_old) > 0.01:
                Prob_poss_matches = (1.0 + ((1 - P_o) / (np.exp(B_poss_matches[:, 5]) * P_o))) ** -1
                Nstar = np.sum(Prob_poss_matches)
                P_o_old = P_o
                P_o = P_o / (Nstar / Nstar_orig)
            B_poss_matches[:, 5] = Prob_poss_matches
            # if P_o > P_o_old:
            #    print 'what is going on?',Nstar
            # print Prob_poss_matches,np.exp(B_poss_matches[:,5]),P_o,P_o_old





            # sort the matches by the total ln B
            B_poss_matches = B_poss_matches[B_poss_matches[:, 5].argsort(),]

            # take the best matches, removing duplicates
            Best_matches = np.array([],dtype=int)
            used_y1 = np.array([])
            used_y2 = np.array([])
            for i in np.arange(0, B_poss_matches.shape[0])[::-1]:
                if np.where(used_y1 == B_poss_matches[i, 6])[0].size == 0 and np.where(used_y2 == B_poss_matches[i, 7])[
                    0].size == 0 and B_poss_matches[i, 5] > Bayes_threshold:
                    used_y1 = np.append(used_y1, B_poss_matches[i, 6])
                    used_y2 = np.append(used_y2, B_poss_matches[i, 7])
                    Best_matches = np.append(Best_matches, i)

                    # work out how many correct ID matches there are with BM
            cross_match_y1 = np.array([], dtype='int')
            cross_match_y2 = np.array([], dtype='int')
            ii = 0
            # check for where ther are matches in both years

            if ind_y1_month.size > 1:
                for i in data_y1.ID[ind_y1_month]:
                    ind = np.where(data_y2.ID[ind_y2_month] == i)[0]
                    if ind.size != 0:
                        cross_match_y1 = np.append(cross_match_y1, ii)
                        cross_match_y2 = np.append(cross_match_y2, ind)
                ii += 1
            if ind_y1_month.size == 1:
                ind = np.where(data_y2.ID[ind_y2_month] == data_y1.ID[ind_y1_month])[0]
                if ind.size != 0:
                    cross_match_y1 = np.append(cross_match_y1, ii)
                    cross_match_y2 = np.append(cross_match_y2, ind)
            ##Now check which XID matches are correct
            total_correct = 0
            for i in np.arange(0, Best_matches.size):
                # Bayes_gender, Bayes_BMI, Bayes_height, Bayes_age, Bayes_geno,probability of match, ID1,ID2,0.0,ind_y1,ind_y2
                Matches[:, match_counter] = B_poss_matches[Best_matches[i], 0], B_poss_matches[Best_matches[i], 1], \
                                            B_poss_matches[Best_matches[i], 2], B_poss_matches[Best_matches[i], 3], \
                                            B_poss_matches[Best_matches[i], 4], B_poss_matches[Best_matches[i], 5], \
                                            data_y1.ID[ind_y1_month[np.int(B_poss_matches[Best_matches[i], 6])]], data_y2.ID[
                                                ind_y2_month[np.int(B_poss_matches[Best_matches[i], 7])]], 0.0, ind_y1_month[
                                                np.int(B_poss_matches[Best_matches[i], 6])], ind_y2_month[
                                                np.int(B_poss_matches[Best_matches[i], 7])]
                if data_y1.ID[ind_y1_month[np.int(B_poss_matches[Best_matches[i], 6])]] == data_y2.ID[
                    ind_y2_month[np.int(B_poss_matches[Best_matches[i], 7])]]:
                    total_correct += 1.0
                    Matches[8, match_counter] = 1.0
                match_counter += 1
            results_table[:, m] = Best_matches.size, total_correct, cross_match_y1.size, ind_y1_month.shape[0],ind_y2_month.shape[0]

    ind_match = np.where(Matches[8, :] > -1.0)[0]
    return Matches[0:8, ind_match]
