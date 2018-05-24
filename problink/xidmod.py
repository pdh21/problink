"""Probabalistic catalogue matching module: includes functions and tools used in the 
catalogue matching process"""
import numpy as np


def read_ECFSPR_all():
    import asciitable
    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/2008_2009/db0809_protected.txt', guess=False,
                        delimiter='\t', fill_values=[('', '-999')])
    return x


# -----BMI model setup-----------------------------------
def BMI_model(m, x):
    """given (intercept,slope), calculates predicted BMI assuming BMI=m[1]*x+m[0]"""
    return (x * m[1]) + m[0]


def log_likelihood(data, model, sigma, x):
    """data=BMI (N=2), model=(intercept,slope),sigma (N=2),x=np.array([1,2])"""
    chi = np.sum((data - BMI_model(model, x)) ** 2 / (2 * sigma ** 2))
    N = np.log(1.0 / (2 * np.pi * sigma ** 2))
    return N - chi


def BMI_prior_setup():
    from scipy import stats

    # load in prior information
    BMI_priors = np.load(
        'BMI_prior.npz')  # prior on intercept taken from the normalised histogram of BMIs from the 2008 and 2009 dataset (BMI_change.py)
    pdf_int = BMI_priors['arr_0']
    bins_int = BMI_priors['arr_1']

    # prior on slope (taken to be 0)
    bins_slope = np.arange(-0.5, 0.5, 0.05)
    pdf_slope = stats.norm.pdf(bins_slope, 0.0, 0.005)
    pdf_slope = np.array([pdf_slope])
    pdf_int = np.array([pdf_int])

    # make grid of prior values
    prior = np.dot(pdf_slope.T, pdf_int)
    # integrate to find normalisation factor
    norm_f = np.trapz(np.trapz(prior, axis=0, x=bins_slope), x=bins_int)
    prior = prior / norm_f
    return prior, bins_slope, bins_int


def ECFPR_data_BMI():
    import asciitable
    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/2008_2009/db0809_protected.txt', guess=False,
                        delimiter='\t', fill_values=[('', '-999')])

    yy = map(int, x['birth_yy'])
    year = map(int, x['year'])
    gender = np.array(map(float, x['gender']))
    mm = map(float, x['birth_mm'])
    dd = map(float, x['birth_dd'])
    ID = map(float, x['ID'])
    bmi = map(float, x['bmiECFSPR'])
    hgt = map(float, x['hgt'])

    hgt = np.array(hgt)
    ID = np.array(ID)
    BMI = np.array(bmi)
    year = np.array(year)
    dob_j = np.empty_like(yy)
    for i in range(0, len(yy)):
        dob_j[i] = ((yy[i] - 1900) * 12) + mm[i]
    # indices in dob_j for each year


    # first select patients which have a BMI and height measure
    b = np.array([], dtype=int)
    for i in range(0, len(BMI) - 1):
        if BMI[i] >= 0:
            if hgt[i] > 0:
                b = np.append(b, [[i]])
    BMI = BMI[b]
    ID = ID[b]
    year = year[b]
    gender = gender[b]
    dob_j = dob_j[b]
    hgt[b]

    # next, selet patients where there is a match in the database
    a = np.array([], dtype=int)
    for i in range(0, len(ID) - 1):
        if ID[i] == ID[i + 1]:
            a = np.append(a, [[i], [i + 1]])

    BMI = BMI[a]
    ID = ID[a]
    year = year[a]
    gender = gender[a]
    dob_j = dob_j[a]
    hgt = hgt[a]

    ind_2008, = np.nonzero(np.less(year, 2009))
    ind_2009, = np.nonzero(np.greater(year, 2008))
    return BMI, hgt, ID, ind_2008, ind_2009, dob_j, gender


def mlm_params(data, sigma, x):
    sig_sq = sigma ** 2
    S1 = np.sum(1.0 / sig_sq)
    Sx = np.sum(x / sig_sq)
    Sxx = np.sum(x ** 2 / sig_sq)
    Sxy = np.sum(x * data / sig_sq)
    Sy = np.sum(data / sig_sq)
    D = S1 * Sxx - Sx ** 2
    a = (Sy * Sxx - Sx * Sxy) / D
    b = (S1 * Sxy - Sx * Sy) / D
    return a, b


def test_mlm():
    BMI = np.array([25.0, 24.9])
    x = np.array([1, 2])
    sigma = np.array([1.2, 1.2])
    model = mlm_params(BMI, sigma, x)
    import pylab as plt
    print
    model
    intercept = np.arange(20, 30, 0.1)
    like = np.array([])
    for i in intercept:
        print(i, model[1])
        like = np.append(like, log_likelihood(BMI, (i, model[1]), sigma, x))
    print
    like
    plt.plot(intercept, like)
    plt.show()


# ---mutation based functions------------
def make_mutations_list():
    x = read_ECFSPR_all()
    mutations = np.array([])
    ii = 0
    for i in x['mut1']:
        if np.where(mutations == i)[0].size == 0 and i != '--':
            mutations = np.append(mutations, i)
            if mutations[-1] == '0.0':
                print
                '---', i, x['ID'][ii]
        ii += 1
    for i in x['mut2']:
        if np.where(mutations == i)[0].size == 0 and i != '--':
            mutations = np.append(mutations, i)
    return mutations


def create_mutation_pdf():
    import numpy as np
    x = read_ECFSPR_all()

    mut_count = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mut_labels = ['df508 df508', 'df508 other', 'df508 unknown', 'other other', 'other unknown', 'unknown unknown']
    for i in range(0, x.size):
        mut1 = x['mut1'][i]
        mut2 = x['mut2'][i]
        if mut1 == '[delta]F508':
            if mut2 == '[delta]F508':
                mut_count[0] += 1.0
            elif mut2 == 'unknown' or mut2 == '--':
                mut_count[2] += 1.0
            else:
                mut_count[1] += 1.0
        elif mut1 != 'unknown' or mut1 != '--':
            if mut2 == 'unknown' or mut2 == '--':
                mut_count[4] += 1.0
            else:
                mut_count[3] += 1.0
        else:
            mut_count[5] += 1.0
    mut_count = mut_count / x.size
    np.save('./data/mut_pdf', mut_count)


def load_mut_pdf():
    return np.load('./data/mut_pdf.npy')


def like_mutation_same(data, model):
    # define as Bernoulli distribution??
    if data[0, 0] == data[1, 0] and data[0, 1] == data[1, 1]:
        return model
    if data[0, 0] != data[1, 0] or data[0, 1] != data[1, 1]:
        return 1 - model


def mutation_not_same(mut_pdf, data):
    if data[0] == '[delta]F508':
        if data[1] == '[delta]F508':
            return mut_pdf[0]
        elif data[1] == 'unknown' or data[1] == '--':
            return mut_pdf[2]
        else:
            return mut_pdf[1]
    elif data[0] != 'unknown' or data[0] != '--':
        if data[1] == 'unknown' or data[1] == '--':
            return mut_pdf[4]
        else:
            return mut_pdf[3]
    else:
        return mut_pdf[5]


def mutation_prior_setup(min_q_H, max_q_H):
    bins_q_H = np.linspace(min_q_H, max_q_H, 10)
    prior_q_H = np.empty_like(bins_q_H)
    prior_q_H[:] = 1.0 / (max_q_H - min_q_H)
    return prior_q_H, bins_q_H


## #------------ GENDER based functions--------------------

def gender_prior_setup(min_q, max_q):
    bins_q = np.linspace(min_q, max_q, 10)
    prior_q = np.empty_like(bins_q)
    prior_q[:] = 1.0 / (max_q - min_q)
    return prior_q, bins_q


def like_gender_same(data, model):
    # define as Bernoulli distribution??
    if data[0] == data[1]:
        return model
    if data[0] != data[1]:
        return 1 - model


def like_gender_diff(data, model):
    # define as Bernoulli distribution
    if data == 1:
        return model
    if data == 2:
        return 1 - model


##-----age at diagnosis based functions-------------

def age_dia_prior_H_setup(min_q, max_q):
    bins_q = np.linspace(min_q, max_q, 100)
    prior_q = np.empty_like(bins_q)
    prior_q[:] = 1.0 / (max_q - min_q)
    return prior_q, bins_q


def age_dia_prior_K_setup(mu, sig):
    from scipy import stats
    bins_age_dia = np.arange(0.0, 80, 0.1)
    pdf_age_dia = stats.lognorm.pdf(bins_age_dia, mu, sig)
    # pdf_age_dia=np.empty_like(bins_age_dia)
    # pdf_age_dia[:]=1.0/80.0
    return bins_age_dia, pdf_age_dia


def like_age_dia_same(data, model):
    # define as Bernoulli distribution??
    if data[0] == data[1]:
        return model
    if data[0] != data[1]:
        return 1 - model


def age_dia_log_likelihood(data, model, sigma, x):
    """data=BMI (N=2), model=(intercept,slope),sigma (N=2),x=np.array([1,2])"""
    chi = np.sum((data - age_dia_model(model, x)) ** 2 / (2 * sigma ** 2))
    N = np.log(1.0 / (2 * np.pi * sigma ** 2))
    return N - chi


def age_dia_model(m, x):
    """given (intercept,slope), calculates predicted hgt assuming hgt=m[1]*x+m[0]"""
    return (x * 0.0) + m


def age_dia_linear_pdf(x, c):
    if x < 0 and x > -1:
        x = 0.0
    m = -0.5 * c ** 2
    return (m * x) + c


##-----DOB based functions-------------

def DOB_prior_setup(min_q, max_q):
    bins_q = np.linspace(min_q, max_q, 100)
    prior_q = np.empty_like(bins_q)
    prior_q[:] = 1.0 / (max_q - min_q)
    return prior_q, bins_q


# ----Height based functions------------

def hgt_prior_setup():
    from scipy import stats

    # load in prior information
    BMI_priors = np.load(
        'hgt_prior.npz')  # prior on intercept taken from the normalised histogram of BMIs from the 2008 and 2009 dataset (BMI_change.py)
    pdf_int = BMI_priors['arr_0']
    bins_int = BMI_priors['arr_1']

    # prior on slope (taken to be 0)
    bins_slope = np.arange(-0.5, 0.5, 0.05)
    pdf_slope = stats.norm.pdf(bins_slope, 0.0, 0.005)
    pdf_slope = np.array([pdf_slope])
    pdf_int = np.array([pdf_int])

    # make grid of prior values
    prior = np.dot(pdf_slope.T, pdf_int)
    # integrate to find normalisation factor
    norm_f = np.trapz(np.trapz(prior, axis=0, x=bins_slope), x=bins_int)
    prior = prior / norm_f
    return prior, bins_slope, bins_int


# ---------stuff required by height model and that should be stored in memory
Female = np.load('./data/Female_height.npz')
age_height_female = Female['arr_0']
height_female = Female['arr_1']
Male = np.load('./data/Male_height.npz')
age_height_male = Male['arr_0']
height_male = Male['arr_1']
percentiles = np.array([0.0, 3, 5, 10, 25, 50, 75, 90, 95, 97, 100.0])
from scipy.interpolate import interp1d


# ---------


def hgt_model(percentile, age, sex):
    """given age,sex and percentile interpolate """
    height = []
    if sex == 1:
        for i in age:
            ind_age = (np.abs(age_height_male - i)).argmin()
            f = interp1d(percentiles, height_male[ind_age, :], kind='cubic')
            height.append(f(percentile))
    if sex == 2:
        for i in age:
            ind_age = (np.abs(age_height_female - i)).argmin()
            f = interp1d(percentiles, height_female[ind_age, :], kind='cubic')
            height.append(f(percentile))
    # print height
    return height


def hgt_log_likelihood(data, percentile, sigma, age, sex):
    """data=BMI (N=2), model=(intercept,slope),sigma (N=2),x=np.array([1,2])"""
    chi = np.sum((data - hgt_model(percentile, age, sex)) ** 2 / (2 * sigma ** 2))
    N = np.log(1.0 / (2 * np.pi * sigma ** 2))
    # print N-chi
    return N - chi


# ---------stuff required by BMI model and that should be stored in memory
Female = np.load('./data/Female_BMI.npz')
age_BMI_female = Female['arr_0']
BMI_female = Female['arr_1']
Male = np.load('./data/Male_BMI.npz')
age_BMI_male = Male['arr_0']
BMI_male = Male['arr_1']
percentiles_BMI = np.array([0.0, 10, 25, 50, 75, 90, 100.0])


def BMI_model_curve(percentile, age, sex):
    """given age,sex and percentile interpolate """
    BMI = []
    if sex == 1:
        for i in age:
            ind_age = (np.abs(age_BMI_male - i)).argmin()
            # print percentiles_BMI.shape,BMI_male[ind_age,:].shape
            f = interp1d(percentiles_BMI, BMI_male[ind_age, :], kind='cubic')
            BMI.append(f(percentile))
    if sex == 2:
        for i in age:
            ind_age = (np.abs(age_BMI_female - i)).argmin()
            f = interp1d(percentiles_BMI, BMI_female[ind_age, :], kind='cubic')
            BMI.append(f(percentile))
    # print height
    return BMI


def BMI_log_likelihood(data, percentile, sigma, age, sex):
    """data=BMI (N=2), model=(intercept,slope),sigma (N=2),x=np.array([1,2])"""
    chi = np.sum((data - BMI_model_curve(percentile, age, sex)) ** 2 / (2 * sigma ** 2))
    N = np.log(1.0 / (2 * np.pi * sigma ** 2))
    # print N-chi
    return N - chi


# ----MASTER_functions--------
def get_ECFSPR_data():
    import asciitable
    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/2008_2009/db0809_protected.txt', guess=False,
                        delimiter='\t', fill_values=[('', '-999')])
    # x=asciitable.read('db0809_protected.txt', guess=False,delimiter='\t',fill_values=[('', '-999')])


    yy = map(int, x['birth_yy'])
    year = map(int, x['year'])
    gender = np.array(map(float, x['gender']))
    mm = map(float, x['birth_mm'])
    dd = map(float, x['birth_dd'])
    ID = map(float, x['ID'])
    bmi = map(float, x['bmiECFSPR'])
    hgt = map(float, x['hgt'])
    mut1 = np.array(x['mut1'])
    mut2 = np.array(x['mut2'])
    age_dia = np.array(x['age_dia'])
    ID = np.array(ID)
    BMI = np.array(bmi)
    hgt = np.array(hgt)
    year = np.array(year)
    dob_j = np.empty_like(yy)
    for i in range(0, len(yy)):
        dob_j[i] = ((yy[i] - 1900) * 12) + mm[i]
    # indices in dob_j for each year

    ind_2008, = np.nonzero(np.less(year, 2009))
    ind_2009, = np.nonzero(np.greater(year, 2008))
    return BMI, hgt, ID, ind_2008, ind_2009, dob_j, gender, np.array([mut1, mut2]), age_dia


def BMI_Bayes(prior, bins_int, bins_slope, data, sigma):
    """Calcualte the log Bayes factor for a pair, given prior info on BMI model and patient data"""
    comb_obj = np.copy(prior)
    obj1 = np.copy(prior)
    obj2 = np.copy(prior)
    x = np.array([1, 2])
    for i in range(0, bins_slope.size):
        # loop over every value in intercept
        for j in range(0, bins_int.size):
            # mulitply prior by likelihood at values bins_slope[i] and bins_int[j]
            comb_obj[i, j] = comb_obj[i, j] * np.exp(log_likelihood(data, (bins_int[j], bins_slope[i]), sigma, x))
            obj1[i, j] = obj1[i, j] * np.exp(log_likelihood(data[0], (bins_int[j], bins_slope[i]), sigma, x[0]))
            obj2[i, j] = obj2[i, j] * np.exp(log_likelihood(data[1], (bins_int[j], bins_slope[i]), sigma, x[1]))
            # calculate Bayes factor by doing double integral over slope and intercept, note multiplication of integral on bottom
    return np.log(np.trapz(np.trapz(comb_obj, axis=0, x=bins_slope), x=bins_int) / (
    np.trapz(np.trapz(obj1, axis=0, x=bins_slope), x=bins_int) * np.trapz(np.trapz(obj2, axis=0, x=bins_slope),
                                                                          x=bins_int)))


def hgt_Bayes(prior, bins_int, bins_slope, data, sigma):
    """Calcualte the log Bayes factor for a pair, given prior info on hgt model and patient data"""
    comb_obj = np.copy(prior)
    obj1 = np.copy(prior)
    obj2 = np.copy(prior)
    x = np.array([1, 2])
    for i in range(0, bins_slope.size):
        # loop over every value in intercept
        for j in range(0, bins_int.size):
            # mulitply prior by likelihood at values bins_slope[i] and bins_int[j]
            comb_obj[i, j] = comb_obj[i, j] * np.exp(log_likelihood(data, (bins_int[j], bins_slope[i]), sigma, x))
            obj1[i, j] = obj1[i, j] * np.exp(log_likelihood(data[0], (bins_int[j], bins_slope[i]), sigma, x[0]))
            obj2[i, j] = obj2[i, j] * np.exp(log_likelihood(data[1], (bins_int[j], bins_slope[i]), sigma, x[1]))
            # calculate Bayes factor by doing double integral over slope and intercept, note multiplication of integral on bottom
    return np.log(np.trapz(np.trapz(comb_obj, axis=0, x=bins_slope), x=bins_int) / (
    np.trapz(np.trapz(obj1, axis=0, x=bins_slope), x=bins_int) * np.trapz(np.trapz(obj2, axis=0, x=bins_slope),
                                                                          x=bins_int)))


def hgt_Bayes_curve(prior, bins_percentile, data, sigma, age, sex):
    """Calcualte the log Bayes factor for a pair, given prior info on hgt model and patient data"""
    comb_obj = np.copy(prior)
    obj1 = np.copy(prior)
    obj2 = np.copy(prior)
    x = np.array([1, 2])
    for i in range(0, bins_percentile.size):
        comb_obj[i] = comb_obj[i] * np.exp(hgt_log_likelihood(data, bins_percentile[i], sigma, age, sex))
        obj1[i] = obj1[i] * np.exp(hgt_log_likelihood(data[0], bins_percentile[i], sigma, [age[0]], sex))
        obj2[i] = obj2[i] * np.exp(hgt_log_likelihood(data[1], bins_percentile[i], sigma, [age[1]], sex))
        # calculate Bayes factor by doing double integral over slope and intercept, note multiplication of integral on bottom
    # import pylab as plt
    # plt.plot(bins_percentile,comb_obj,'g', label='comb'+str(data[1]))
    # plt.plot(bins_percentile,obj1,'r',label='obj1'+str(data[1]))
    # plt.plot(bins_percentile,obj2,'b',label='obj2'+str(data[1]))
    return np.log(np.trapz(comb_obj, axis=0, x=bins_percentile) / (
    np.trapz(obj1, axis=0, x=bins_percentile) * np.trapz(obj2, axis=0, x=bins_percentile)))


def BMI_Bayes_curve(prior, bins_percentile, data, sigma, age, sex):
    """Calcualte the log Bayes factor for a pair, given prior info on hgt model and patient data"""
    comb_obj = np.copy(prior)
    obj1 = np.copy(prior)
    obj2 = np.copy(prior)
    x = np.array([1, 2])
    for i in range(0, bins_percentile.size):
        comb_obj[i] = comb_obj[i] * np.exp(BMI_log_likelihood(data, bins_percentile[i], sigma, age, sex))
        obj1[i] = obj1[i] * np.exp(BMI_log_likelihood(data[0], bins_percentile[i], sigma, [age[0]], sex))
        obj2[i] = obj2[i] * np.exp(BMI_log_likelihood(data[1], bins_percentile[i], sigma, [age[1]], sex))
        # calculate Bayes factor by doing double integral over slope and intercept, note multiplication of integral on bottom
    # import pylab as plt
    # plt.plot(bins_percentile,comb_obj,'g', label='comb'+str(data[1]))
    # plt.plot(bins_percentile,obj1,'r',label='obj1'+str(data[1]))
    # plt.plot(bins_percentile,obj2,'b',label='obj2'+str(data[1]))
    return np.log(np.trapz(comb_obj, axis=0, x=bins_percentile) / (
    np.trapz(obj1, axis=0, x=bins_percentile) * np.trapz(obj2, axis=0, x=bins_percentile)))


def gender_Bayes(prior_q_H, prior_q_K, bins_q_H, bins_q_K, data):
    comb_obj = np.copy(prior_q_H)
    obj1 = np.copy(prior_q_K)
    obj2 = np.copy(prior_q_K)

    for i in range(0, bins_q_H.size):
        comb_obj[i] = comb_obj[i] * like_gender_same(data, bins_q_H[i])

    for i in range(0, bins_q_K.size):
        obj1[i] = obj1[i] * like_gender_diff(data[0], bins_q_K[i])
        obj2[i] = obj2[i] * like_gender_diff(data[1], bins_q_K[i])

    return np.log(np.trapz(comb_obj, x=bins_q_H) / (np.trapz(obj1, x=bins_q_K) * np.trapz(obj2, x=bins_q_K)))


def mutation_Bayes(prior_q_H, mut_pdf, bins_q_H, data):
    comb_obj = np.copy(prior_q_H)
    for i in range(0, bins_q_H.size):
        comb_obj[i] = comb_obj[i] * like_mutation_same(data, bins_q_H[i])
    obj1 = mutation_not_same(mut_pdf, data[0, :])
    obj2 = mutation_not_same(mut_pdf, data[1, :])
    return np.log(np.trapz(comb_obj, x=bins_q_H) / (obj1 * obj2))


def age_dia_Bayes(prior_q_H, bins_q_H, data):
    c = 0.025
    comb_obj = np.copy(prior_q_H)
    for i in range(0, bins_q_H.size):
        comb_obj[i] = comb_obj[i] * like_age_dia_same(data, bins_q_H[i])
    obj1 = age_dia_linear_pdf(data[0], c)
    obj2 = age_dia_linear_pdf(data[1], c)
    return np.log(np.trapz(comb_obj, x=bins_q_H) / (obj1 * obj2))


## def DOB_Bayes(prior_q_H,prior_q_K,bins_q_H,bins_q_K,data):
##     comb_obj=np.copy(prior_q_H)
##     obj1=np.copy(prior_q_K)
##     obj2=np.copy(prior_q_K)

##     for i in range(0,bins_q_H.size):
##         comb_obj[i]=comb_obj[i]*like_gender_same(data,bins_q_H[i])

##     for i in range(0,bins_q_K.size):
##         obj1[i]=obj1[i]*like_gender_diff(data[0],bins_q_K[i])
##         obj2[i]=obj2[i]*like_gender_diff(data[1],bins_q_K[i])

##     return np.log(np.trapz(comb_obj,x=bins_q_H)/(np.trapz(obj1,x=bins_q_K)*np.trapz(obj2,x=bins_q_K)))





# ---------------height curves------
def save_male_height_np():
    import asciitable
    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/Male_height_2_20_years.txt', guess=False,
                        delimiter='\t', fill_values=[('', '-999')])
    age = []
    height = []
    for i in x:
        age.append(i[0])
        height_tmp = []
        for j in range(1, 12):
            height_tmp.append(i[j])

        height.append(height_tmp)

    age = np.array(age)
    height = np.array(height)

    age = np.append(age, np.arange(240, 960.0, 1))
    height_extra = np.empty((960.0 - 240.0, height.shape[1]))
    print
    height[-1, :], height.shape, height_extra.shape
    for i in np.arange(0, 960.0 - 240.0, 1):
        print
        i
        height_extra[i, :] = height[-1, :]
    height = np.concatenate((height, height_extra), axis=0)
    import pylab as plt
    for i in range(0, 11):
        plt.plot(age, height[:, i])
    plt.show()
    np.savez('Male_height', age, height)


def save_female_height_np():
    import asciitable
    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/Female_height_2_20_years.txt', guess=False,
                        delimiter='\t', fill_values=[('', '-999')])
    age = []
    height = []
    for i in x:
        age.append(i[0])
        height_tmp = []
        for j in range(1, 12):
            height_tmp.append(i[j])

        height.append(height_tmp)
    age = np.array(age)
    height = np.array(height)

    age = np.append(age, np.arange(240, 960.0, 1))
    height_extra = np.empty((960.0 - 240.0, height.shape[1]))
    print
    height[-1, :], height.shape, height_extra.shape
    for i in np.arange(0, 960.0 - 240.0, 1):
        print
        i
        height_extra[i, :] = height[-1, :]
    height = np.concatenate((height, height_extra), axis=0)
    import pylab as plt
    for i in range(0, 11):
        plt.plot(age, height[:, i])
    plt.show()
    np.savez('Female_height', age, height)


def save_BMI_data_Male():
    import asciitable

    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/BMI_Boelle2012.txt', guess=False, delimiter='\t',
                        fill_values=[('', '-999')])
    age_tab = x['age_m']
    BMI_tab = x['male']
    age = np.arange(3, 39, 1)
    ind_90 = range(0, 12)
    ind_75 = range(13, 30)
    ind_50 = range(31, 44)
    ind_25 = range(45, 56)
    ind_10 = range(57, 68)
    indices = [ind_10, ind_25, ind_50, ind_75, ind_90]
    BMI = np.empty((36.0, 7))
    ii = 0
    for i in indices:
        print
        age_tab[i], BMI_tab[i]
        f = interp1d(age_tab[i], BMI_tab[i], kind='slinear')
        BMI[:, ii + 1] = f(age)
        ii += 1
    BMI[:, 0] = BMI[:, 1] / 1.5
    BMI[:, -1] = BMI[:, -2] * 1.2

    age = np.append(age, np.arange(39, 80.0, 1))
    BMI_extra = np.empty((80 - 39, BMI.shape[1]))
    for i in np.arange(0, 80 - 39, 1):
        print
        i
        BMI_extra[i, :] = BMI[-1, :]
    BMI = np.concatenate((BMI, BMI_extra), axis=0)
    import pylab as plt
    for i in range(0, 7):
        plt.plot(age, BMI[:, i])
    plt.show()
    np.savez('Male_BMI', age, BMI)


def save_BMI_data_Female():
    import asciitable

    x = asciitable.read('/Users/pdh21/Documents/CFwork/Patient_link/BMI_Boelle2012.txt', guess=False, delimiter='\t',
                        fill_values=[('', '-999')])
    age_tab = x['age_fe']
    BMI_tab = x['female']
    age = np.arange(3, 38, 1)
    ind_90 = range(0, 10)
    ind_75 = range(11, 24)
    ind_50 = range(25, 39)
    ind_25 = range(40, 51)
    ind_10 = range(52, 64)
    indices = [ind_10, ind_25, ind_50, ind_75, ind_90]
    BMI = np.empty((35.0, 7))
    ii = 0
    for i in indices:
        print
        age_tab[i], BMI_tab[i]
        f = interp1d(age_tab[i], BMI_tab[i], kind='slinear')
        BMI[:, ii + 1] = f(age)
        ii += 1
    BMI[:, 0] = BMI[:, 1] / 1.5
    BMI[:, -1] = BMI[:, -2] * 1.2

    age = np.append(age, np.arange(38, 80.0, 1))
    BMI_extra = np.empty((80 - 38, BMI.shape[1]))
    for i in np.arange(0, 80 - 38, 1):
        print
        i
        BMI_extra[i, :] = BMI[-1, :]
    BMI = np.concatenate((BMI, BMI_extra), axis=0)
    import pylab as plt
    for i in range(0, 7):
        plt.plot(age, BMI[:, i])
    plt.show()
    np.savez('Female_BMI', age, BMI)






