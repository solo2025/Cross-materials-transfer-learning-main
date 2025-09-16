import pandas as pd
import numpy as np

def parameter_calculation(data, len_element):
    

    composition = ['x(Al)','x(Nb)','x(Ti)','x(V)','x(Zr)','x(Cr)','x(Mo)','x(Hf)','x(Ta)']
    #判断data是否为dataframe格式
    Vars = np.array(data)
    # Vars = data.values
    len_element = len_element
    ##delta-Smix
    Smix = []
    for n in range(len(Vars)):
        deltas = 0
        for m in range(len_element):
            x = Vars[n,m]
            if x > 0:
                deltas = deltas-8.314*(x/100)*np.log((x/100))
        # 保留两位小数
        Smix.append(np.round(deltas,3))
    # print(Smix)

    ##delta-Hmix
    Hmix = []
    p1 = [[0,-18,-30,-16,-44,-10,-5,-39,-19],
        [0,0,2,-1,4,-7,-6,4,0],
        [0,0,0,-2,0,-2,-4,0,1],
        [0,0,0,0,-4,-2,0,-2,-1],
        [0,0,0,0,0,-12,-6,0,9],
        [0,0,0,0,0,0,0,-9,-7],
        [0,0,0,0,0,0,0,-4,-5],
        [0,0,0,0,0,0,0,0,-3],
        [0,0,0,0,0,0,0,0,0]]

    for i in range(0,len(Vars)):
        deltah = 0
        for j in range(len_element):
            for k in range(j,len_element):
                deltah = deltah + Vars[i,j]*Vars[i,k]*p1[j][k]*4/10000
        Hmix.append(np.round(deltah,3))  
    # print(Hmix)

    ##Tm
    p2 = [933,2750,1941,2183,2128,2180,2896,2506,3290]
    Tm = []
    for i in range(0,len(Vars)):
        t = 0
        for j in range(len_element):
            t = t+Vars[i,j]*p2[j]/100
        Tm.append(np.round(t,3))
    # print(Tm)

    ##Omiga
    Omiga = []
    for i in range(0,len(Vars)):
        if Hmix[i] == 0:
            o = 0
        else:
            o =  Smix[i]*Tm[i]/np.abs(Hmix[i])/1000
        Omiga.append(np.round(o,3))
    # Omiga = np.nan_to_num(Omiga,nan=0)
    # print(Omiga)

    #delta-r
    delta_r = []
    p3 = [141,150,142,134,158,130,139,158,154]
    mean_r = []
    for i in range(0,len(Vars)):
        r = 0
        for j in range(len_element):
            r = r+Vars[i,j]*p3[j]/100
        mean_r.append(r)
        deltar = 0
        for k in range(len_element):
            deltar = deltar + Vars[i,k]/100*(1-p3[k]/mean_r[i])**2
        delta_r.append(np.round(np.sqrt(deltar)*100,3))
    # print(dr)

    #gama
    gama = []
    for i in range(0,len(Vars)):
        pr = []
        for j in range(len_element):
            if Vars[i,j] != 0:
                pr.append(p3[j])
        maxr, minr = max(pr), min(pr)
        up = 1 - np.sqrt(((minr + mean_r[i])**2-mean_r[i]**2)/(minr + mean_r[i])**2)
        low = 1 - np.sqrt(((maxr + mean_r[i])**2-mean_r[i]**2)/(maxr + mean_r[i])**2)
        gama.append(np.round(up/low,3))
    # print(gama)

    #Dr
    Dr = []
    for i in range(len(Vars)):
        dr_value = 0
        for j in range(len_element):
            for k in range(len_element):
                dr_value += Vars[i,j]*Vars[i,k]*np.abs(p3[j]-p3[k])/10000
        Dr.append(np.round(dr_value,3))             

    #VEC
    p4 = [3,5,4,5,4,6,6,4,5]
    vec = []
    for i in range(0,len(Vars)):
        v = 0
        for j in range(len_element):
            v = v+Vars[i,j]*p4[j]/100
        vec.append(np.round(v,3))
    # print(vec)

    #e/a
    p5 = [1,1,2,2,2,1,1,2,2]
    e = []
    for i in range(0,len(Vars)):
        a = 0
        for j in range(len_element):
            a = a+Vars[i,j]*p5[j]/100
        e.append(np.round(a,3))
    # print(e)

    #X_pauling
    p6 = [1.61,1.6,1.54,1.63,1.33,1.66,2.16,1.3,1.5]
    xpauling = []
    for i in range(0,len(Vars)):
        a, xp = 0, 0
        for j in range(len_element):
            a = a+Vars[i,j]*p6[j]/100
        mean_xp = a
        for k in range(len_element):
            xp = xp + Vars[i,k]/100*(p6[k]-mean_xp)**2
        xpauling.append(np.round(np.sqrt(xp)*100,3))
    # print(xpauling)

    #X_Allen
    p7 = [1.51,1.41,1.38,1.53,1.32,1.65,1.47,1.16,1.34]
    xallen = []
    for i in range(0,len(Vars)):
        a, xa = 0, 0
        for j in range(len_element):
            a = a+Vars[i,j]*p7[j]/100
        mean_xa = a
        for k in range(len_element):
            xa = xa + Vars[i,k]/100*(1 - p7[k]/mean_xa)**2
        xallen.append(np.round(np.sqrt(xa)*100,3))
    # print(xallen)

    ##density
    p8 = [2.7,8.57,4.507,6.11,6.511,7.19,10.28,13.31,16.68]
    p9 = [26.982,92.906,47.867,50.942,91.224,51.996,95.95,178.49,180.95]
    density = []
    for i in range(0,len(Vars)):
        d = 0
        d1 = 0
        d2 = 0
        for j in range(len_element):
            d1 = d1+Vars[i,j]*p9[j]
            d2 = d2+Vars[i,j]*p9[j]/p8[j]
        d = d1/d2    
        density.append(np.round(d,3))

    ##dr
    p10 = [4.28, 4.3, 4.33, 4.3, 4.05, 4.5, 4.6, 3.9, 4.25]
    W = []
    for i in range(0,len(Vars)):
        w_value = 0
        for j in range(len_element):
            w_value += Vars[i,j]*p10[j]/100
        W.append(np.round(w_value**6,3))

    # G and E
    p11 = [26, 38, 44, 47, 33, 115, 120, 30, 69]#G
    p12 = [70, 105, 116, 128, 68, 279, 329, 78, 186]#E
    p13 = [76, 170, 110, 160, 91.1, 160, 230, 110, 200]#K
    E = []
    G = []
    K = []
    for i in range(0,len(Vars)):
        g_value = 0
        e_value = 0
        k_value = 0
        for j in range(len_element):
            g_value += Vars[i,j]*p11[j]/100
            e_value += Vars[i,j]*p12[j]/100
            k_value += Vars[i,j]*p13[j]/100
        G.append(np.round(g_value,3))
        E.append(np.round(e_value,3))
        K.append(np.round(k_value,3))
    
    #delta_G and delta_E
    delta_G = []
    delta_E = []
    delta_K = []
    for i in range(0,len(Vars)):
        dg_value = 0
        de_value = 0
        dk_value = 0
        for j in range(len_element):
            dg_value += Vars[i,j]*(1-p11[j]/G[i])**2/100
            de_value += Vars[i,j]*(1-p12[j]/E[i])**2/100
            dk_value += Vars[i,j]*(1-p13[j]/K[i])**2/100
        delta_G.append(np.round(np.sqrt(dg_value),3))
        delta_E.append(np.round(np.sqrt(de_value),3))
        delta_K.append(np.round(np.sqrt(dk_value),3))

    #DG and DE
    DG = []
    DE = []
    DK = []
    for i in range(0,len(Vars)):
        dg = 0
        de = 0
        dk = 0
        for j in range(len_element):
            for k in range(len_element):
                dg += Vars[i,j]*Vars[i,k]*np.abs(p11[j]-p11[k])/10000
                de += Vars[i,j]*Vars[i,k]*np.abs(p12[j]-p12[k])/10000
                dk += Vars[i,j]*Vars[i,k]*np.abs(p13[j]-p13[k])/10000
    
        DG.append(np.round(dg,3))
        DE.append(np.round(de,3))
        DK.append(np.round(dk,3))
    
    #eta
    eta = []
    for i in range(0,len(Vars)):
        eta_value = 0
        eta1, eta2 = 0, 0
        for j in range(len_element):
            eta1 = Vars[i,j]*2*(p11[j]-G[i])/(p11[j]+G[i])/100
            eta2 = 1 + 0.5*np.abs(eta1)
            eta_value += eta1/eta2
 
        eta.append(np.round(eta_value*100,3))
    
    #E2/E0 固有弹性应变能
    E20 = []
    alpha2 = []
    p3 = [141,150,142,134,158,130,139,158,154]
    for i in range(0,len(Vars)):
        mean_r = 0
        for j in range(len_element):
            mean_r += Vars[i,j]*p3[j]/100
        e20_value = 0
        alpha2_value = 0
        for k in range(len_element):
            for t in range(k, len_element):
                e20_value += Vars[i,k]*Vars[i,t]*np.abs(p3[k]+p3[t]-2*mean_r)**2/(4*(mean_r**2))
                alpha2_value += Vars[i,k]*Vars[i,t]*np.abs(p3[k]+p3[t]-2*mean_r)/(2*mean_r)
        
        E20.append(np.round(e20_value/10000,6))
        alpha2.append(np.round(alpha2_value/10000,5))    

    # deltaH_el
    deltaH_el = []
    p14 = [9.99,10.84,10.62,8.34,14.01,7.23,9.33,13.41,10.85]
    for i in range(0,len(Vars)):
        v1, v2 = 0, 0
        for j in range(len_element):
            v1 += Vars[i,j]*p13[j]*p14[j]
            v2 += Vars[i,j]*p13[j]
        mean_v = v1/v2  
        Hel_value = 0
        for k in range(len_element):
            Hel_value += Vars[i,k]*p13[k]*(p14[k]-mean_v)**2/2/p14[k]
        
        deltaH_el.append(np.round(Hel_value/100,3))

    #mu
    mu = []
    for i in range(0,len(Vars)):
        mu.append( np.round(delta_r[i]/100*E[i]*0.5,3))





    input = {'Delta-Smix': Smix,'Delta-Hmix': Hmix,	'Tm': Tm, 'Omiga': Omiga, 
             'Delta-r': delta_r, 'Gama': gama, 
             'VEC': vec, 'e/a': e, 'Delta-X_Pauling':xpauling, 'Delta-X_Allen': xallen, 
             'Dr': Dr, 'W': W, 'G': G, 'Delta-G': delta_G, 'DG': DG, 'E': E, 'Delta-E': delta_E, 'DE': DE, 'K': K, 'Delta-K': delta_K, 'DK': DK, 'eta': eta, 'E2/E0': E20, 'alpha2': alpha2, 'mu': mu, 'deltaH_el': deltaH_el,
             'density': density}
    df_input = pd.DataFrame(input)

    return df_input