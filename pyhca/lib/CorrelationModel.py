import numpy as np

def baker_jayaram_correlation_2008(im1, im2, flag_orth = False):
    """
    Computing inter-event correlation coeffcieint between Sa of two periods
    Reference:
        Baker and Jayaram (2008) Correlation of Spectral Acceleration
        Values from NGA Ground Motion Models
    Input:
        im1: 1st intensity measure name
        im2: 2nd intensity measure name
        flag_orth: if the correlation coefficient is computed for the two
                   orthogonal components
    Output:
        rho: correlation coefficient
    Note:
        The valid range of T1 and T2 is 0.01s ~ 10.0s
    """

    # Parse periods from im1 and im2
    if im1.startswith('SA'):
        T1 = float(im1[3:-1])
    elif im1.startswith('PGA'):
        T1 = 0.0
    else:
        return 0.0
    if im2.startswith('SA'):
        T2 = float(im2[3:-1])
    elif im2.startswith('PGA'):
        T2 = 0.0
    else:
        return 0.0

    # Compute Tmin and Tmax (lower bounds 0.01 for T < 0.01)
    Tmin = max(min([T1, T2]), 0.01)
    Tmax = max(max([T1, T2]), 0.01)
    # Cofficient C1
    C1 = 1.0 - np.cos(np.pi / 2.0 - 0.366 * np.log(Tmax / max([Tmin, 0.109])))
    # Cofficient C2
    if Tmax < 0.2:
        C2 = 1.0 - 0.105 * (1.0 - 1.0 / (1.0 + np.exp(100.0 * Tmax - 5.0))) * \
            (Tmax - Tmin) / (Tmax - 0.0099)
    else:
        C2 = 0.0
    # Cofficient C3
    if Tmax < 0.109:
        C3 = C2
    else:
        C3 = C1
    # Cofficient C4
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1.0 + np.cos(np.pi * Tmin / 0.109))
    # rho for a singe component
    if Tmax <= 0.109:
        rho = C2
    elif Tmin > 0.109:
        rho = C1
    elif Tmax < 0.2:
        rho = min([C2, C4])
    else:
        rho = C4
    # rho for orthogonal components Cofficient C1
    if flag_orth:
        rho = rho * (0.79 - 0.023 * np.log(np.sqrt(Tmin * Tmax)))

    return rho


def bradley_correlation_2011(IM, T = None, flag_Ds = True):
    """
    Computing inter-event correlation coeffcieint between Sa(T) and Ds575/D595
    Reference:
        Bradley (2011) Correlation of Significant Duration with Amplitude and
        Cumulative Intensity Measures and Its Use in Ground Motion Selection
    Input:
        IM: string of intensity measure from options as follows
            'Sa', 'PGA', 'PGV', 'ASI', 'SI', 'DSI', 'CAV', 'Ds595'
        T: Sa period
        flag_Ds: true - Ds575, false = Ds595
    Output:
        rho: correlation coefficient
    Note:
        The valid range of T is 0.01s ~ 10.0s
    """
    # PGA
    if IM == 'PGA':
        if flag_Ds:
            return -0.442
        else:
            return -0.305
    elif IM == 'PGV':
        if flag_Ds:
            return -0.259
        else:
            return -0.211
    elif IM == 'ASI':
        if flag_Ds:
            return -0.411
        else:
            return -0.370
    elif IM == 'SI':
        if flag_Ds:
            return -0.131
        else:
            return -0.079
    elif IM == 'DSI':
        if flag_Ds:
            return 0.074
        else:
            return 0.163
    elif IM == 'CAV':
        if flag_Ds:
            return 0.077
        else:
            return 0.122
    elif IM == 'Ds595':
        if flag_Ds:
            return 0.843
        else:
            return None
    elif IM == 'Sa':
        if flag_Ds:
            if T < 0.09:
                a_p = -0.45; a_c = -0.39; b_p = 0.01; b_c = 0.09
            elif T < 0.30:
                a_p = -0.39; a_c = -0.39; b_p = 0.09; b_c = 0.30
            elif T < 1.40:
                a_p = -0.39; a_c = -0.06; b_p = 0.30; b_c = 1.40
            elif T < 6.50:
                a_p = -0.06; a_c = 0.16; b_p = 1.40; b_c = 6.50
            elif T <= 10.0:
                a_p = 0.16; a_c = 0.00; b_p = 6.50; b_c = 10.00
        else:
            if T < 0.04:
                a_p = -0.41; a_c = -0.41; b_p = 0.01; b_c = 0.04
            elif T < 0.08:
                a_p = -0.41; a_c = -0.38; b_p = 0.04; b_c = 0.08
            elif T < 0.26:
                a_p = -0.38; a_c = -0.35; b_p = 0.08; b_c = 0.26
            elif T < 1.40:
                a_p = -0.35; a_c = -0.02; b_p = 0.26; b_c = 1.40
            elif T <= 6.00:
                a_p = -0.02; a_c = 0.23; b_p = 1.40; b_c = 6.00
            elif T <= 10.00:
                a_p = 0.23; a_c = 0.02; b_p = 6.00; b_c = 10.0
        rho = a_p + np.log(T / b_p) / np.log(b_c / b_p) * (a_c - a_p)
        return rho


def baker_bradley_correlation_2017(im1=None, im2=None):
    """
    Correlation between Sa and other IMs
    Baker, J. W., and Bradley, B. A. (2017). “Intensity measure correlations observed in
    the NGA-West2 database, and dependence of correlations on rupture and site parameters.”
    Based on the script: https://github.com/bakerjw/NGAW2_correlations/blob/master/corrPredictions.m
    Input:
        T: period of Sa
        im1: 1st intensity measure name
        im2: 2nd intensity measure name
    Output:
        rho: correlation coefficient
    """

    # im map:
    im_map = {'DS575H': 0, 'DS595H':1, 'PGA': 2, 'PGV': 3}

    period_list = []
    im_list = []
    if im1.startswith('SA'):
        im_list.append('SA')
        period_list.append(float(im1[3:-1]))
    else:
        tmp_tag = im_map.get(im1.upper(), None)
        if tmp_tag is None:
            print("CorrelationModel.baker_bradley_correlation_2017: warning - return 0.0 for unknown {}".format(im1))
            return 0.0
        im_list.append(tmp_tag)
        period_list.append(None)
    if im2.startswith('SA'):
        im_list.append('SA')
        period_list.append(float(im2[3:-1]))
    else:
        tmp_tag = im_map.get(im2.upper(), None)
        if tmp_tag is None:
            print("CorrelationModel.baker_bradley_correlation_2017: warning - return 0.0 for unknown {}".format(im2))
            return 0.0
        im_list.append(tmp_tag)

    if im1.startswith('SA') and im2.startswith('SA'):
        # two Sa intensities
        return baker_jayaram_correlation_2008(im1, im2)
    
    if 'SA' not in im_list:
        # two non-Sa intensities
        # rho matrix
        rho_mat = [[1.000, 0.843, -0.442, -0.259],
                   [0.843, 1.000, -0.405, -0.211],
                   [-0.442, -0.405, 1.000, 0.733],
                   [-0.259, -0.211, 0.733, 1.000]]
        # return
        return rho_mat[im_list[0]][im_list[1]]

    # one Sa + one non-Sa
    im_list.remove('SA')
    im_tag = im_list[0]
    T = [x for x in period_list if x is not None][0]
    # modeling coefficients
    a = [[0.00, -0.45, -0.39, -0.39, -0.06, 0.16],
         [0.00, -0.41, -0.41, -0.38, -0.35, 0.02, 0.23],
         [1.00, 0.97],
         [0.73, 0.54, 0.80, 0.76]]
    b = [[0.00, -0.39, -0.39, -0.06, 0.16, 0.00],
         [0.00, -0.41, -0.38, -0.35, -0.02, 0.23, 0.02],
         [0.895, 0.25],
         [0.54, 0.81, 0.76, 0.70]]
    c = [[],[],
         [0.06, 0.80],
         [0.045, 0.28, 1.10, 5.00]]
    d = [[],[],
         [1.6, 0.8],
         [1.8, 1.5, 3.0, 3.2]]
    e = [[0.01, 0.09, 0.30, 1.40, 6.50 ,10.00],
         [0.01, 0.04, 0.08, 0.26, 1.40, 6.00, 10.00],
         [0.20, 10.00],
         [0.10, 0.75, 2.50, 10.00]]

    # rho
    if im_tag < 2:
        for j in range(1,len(e[im_tag])):
            if T <= e[im_tag][j]:
                rho = a[im_tag][j]+(b[im_tag][j]-a[im_tag][j])/np.log(e[im_tag][j]/e[im_tag][j-1])*np.log(T/e[im_tag][j-1])
                break
    else:
        for j in range(len(e[im_tag])):
            if T <= e[im_tag][j]:
                rho = (a[im_tag][j]+b[im_tag][j])/2-(a[im_tag][j]-b[im_tag][j])/2*np.tanh(d[im_tag][j]*np.log(T/c[im_tag][j]))
                break
    
    # return
    return rho

def get_distance_from_lat_lon(site_loc1, site_loc2):

    # earth radius (km)
    earth_radius_avg = 6371.0
    # site lat and lon
    lat1, lon1 = site_loc1
    lat2, lon2 = site_loc2
    # covert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # calculate haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dist = 2.0*earth_radius_avg*np.arcsin(np.sqrt(np.sin(0.5*dlat)**2+np.cos(lat1)*np.cos(lat2)*np.sin(0.5*dlon)**2))
    # return
    return dist