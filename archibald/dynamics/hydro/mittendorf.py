# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:53:29 2026

@author: jrich
"""

import archibald.numpy as np
import warnings

def mm_raw(
        wl,
        dr,
        Lpp,
        Bwl,
        Tf,
        Ta,
        Cb,
        Le,
        Lr,
        kyy,
        Fr,
    ):
    """
    Function for computing the R_AW transfer function and its 90% prediction interval
    for arbitrary heading angle following the proposed method by Mittendorf, M.; Nielsen,
    U.D.; Bingham, H.B.; Liu, S. (2021) "Towards the Uncertainty Quantification of Semi-Empirical
    Formulas Applied to the Added Resistance of Ships in Waves of Arbitrary Heading" Ocean Engineering Vol. 251 No. 111040. 

    >INPUT 
    wl - wavelength [m]
    dr - wave angle [deg]
    Lpp - length per perpendicular [m]
    Bwl - beam [m]
    Tf - draught at fp [m]
    Ta - draught at ap [m]
    Cb - block coefficient [-]
    Le - length of run (bow) [m]
    Lr - length of run (stern) [m]
    kyy - norm. radius of gyration (kyy/Lpp), in general 0.25 [-]
    Fr - froude number [-]

    >OUTPUT
    array[0] - added resistance coefficient (mean)
    array[1] - added resistance coefficient (lower)
    array[2] - added resistance coefficient (upper)
    array[3] - intrinsic wave frequency
    array[4] - normalized wave frequency(w/sqrt(g/Lpp))
    array[5] - lambda/Lpp
    """

    #initalization
    #heading in radians
    if dr > 180 or dr < 0:
        warnings.warn('Method is only valid in range [0,180]deg.')
        dr = np.nan
    alp = np.radians(dr)
    tmax = max(Tf, Ta)
    #trim
    delta = Ta-Tf
    g = 9.807
    raw_arr = []
    U = Fr*np.sqrt(g*Lpp)
    #angles of entrance
    E1 = np.atan(0.99*Bwl*0.5/Le)
    E2 = np.atan(0.99*Bwl*0.5/Lr)

    #derived parameters    
    ps1= [4.05946942e+00, 7.04999991e-01, 5.62274820e+01, 1.18458738e+00,
         4.05706220e+00, 6.50000001e-01, 2.25000391e+00, 1.33566557e-01,
         1.62572344e+00, 3.58496857e+00, 2.91066129e+01, 1.08675480e+01,
         9.87827505e+00, 4.80961585e+02, 2.62009924e+00, 1.23162591e+02,
         1.74527790e+00, 1.06966785e-01, 3.50000000e+00, 8.24976128e-01,
         1.25000000e-01, 1.03000141e+00, 1.20135494e+00, 7.22951370e-01,
         1.36307675e+01, 5.00000000e+00, 3.14999975e+00, 2.89714558e+00,
         5.55186351e+00]

    ps2 = [4.27240317e+00, 7.05000000e-01, 5.58810360e+01, 1.05849695e+00,
         5.00000000e+00, 6.50000000e-01, 2.53759425e+00, 2.10000000e-01,
         1.50221848e+00, 3.58270417e+00, 2.70758743e+01, 8.20000054e+00,
         6.37000001e+00, 4.25000004e+02, 3.50000000e+00, 1.21801494e+02,
         2.04089706e+00, 8.35524201e-02, 3.50000000e+00, 6.39449863e-01,
         1.25000223e-01, 1.03000050e+00, 1.13419070e+00, 6.03042936e-01,
         1.23638297e+01, 5.00000000e+00, 3.14999947e+00, 1.00000004e+00,
         7.49999988e+00]

    ps3 = [3.03271576e+00, 4.20000015e-01, 4.99673118e+01, 1.65946595e+00,
         3.46566078e+00, 1.08999872e+00, 3.53017296e+00, 1.20000000e-01,
         1.48347334e+00, 2.62015273e+00, 3.09243793e+01, 1.29737533e+01,
         1.07000000e+01, 7.07939677e+02, 1.95001682e+00, 9.35351064e+01,
         2.16620326e+00, 1.39999997e-01, 2.89553627e+00, 8.24992364e-01,
         1.30763658e-01, 1.03000000e+00, 1.42153913e+00, 6.31753054e-01,
         1.72855270e+01, 3.00000001e+00, 3.11414443e+00, 6.00000000e+00,
         2.00000001e+00]

    pb1 = [6.98574593e+00, 4.25032242e-01, 7.09177633e+01, 1.99960254e+00,
         3.01301221e+00, 8.99994676e-01, 3.72568388e+00, 2.81612167e-01,
         1.38750811e+00, 3.86896582e+00, 1.79628922e+01, 1.14985385e+01,
         5.58870161e+00, 4.73743004e+02, 2.21408408e+00, 1.53646073e+02,
         2.43280559e+00, 5.56757940e-02, 2.41039849e+00, 8.13139297e-01,
         2.99999998e-01, 1.00776665e+00, 1.07619998e+00, 7.78406778e-01,
         1.92414808e+01, 3.02938365e+00, 3.84472339e+00, 3.12300755e+00,
         4.52696031e+00]

    pb2 = [4.58710582e+00, 5.65022282e-01, 5.42786254e+01 ,1.23604172e+00,
         4.99998263e+00, 6.50259977e-01, 2.25000468e+00, 2.10000000e-01,
         1.54949896e+00, 3.95366005e+00, 3.37772648e+01, 1.39999578e+01,
         6.37000001e+00, 4.25000564e+02, 3.50000000e+00, 1.18022195e+02,
         2.04252813e+00, 1.04536091e-01, 2.26795773e+00, 7.01539307e-01,
         1.25034505e-01, 1.03001307e+00, 1.44950225e+00, 6.47384026e-01,
         1.44229162e+01, 3.00000134e+00, 3.14999949e+00, 1.00000000e+00,
         7.49999997e+00]

    pb3 = [3.00355039e+00, 4.20001279e-01, 5.58414825e+01, 1.32934752e+00,
         4.31245215e+00, 1.08998811e+00, 3.74999578e+00, 1.46428490e-01,
         1.79301192e+00, 3.56562762e+00, 2.76275045e+01, 1.22975415e+01,
         1.06999891e+01, 6.79271138e+02, 2.30875016e+00, 1.30001874e+02,
         2.23066085e+00, 8.32000024e-02, 2.63843128e+00, 7.83843682e-01,
         2.20011546e-01, 1.35089575e+00, 1.31843660e+00, 6.26778792e-01,
         1.52590637e+01, 3.00000000e+00, 2.40826045e+00, 5.99999925e+00,
         2.00000000e+00]

    raw_arr = []
    #split for blockcoefficient
    if Cb <0.7:
        pv = [ps1, ps2, ps3]
    else:
        pv = [pb1, pb2, pb3]

    #loop for all 3 estimates
    for i in range(3): 
        p = pv[i]
        raw_local = []
        if i ==0:
            add = 0
        elif i==1:
            add = 0.19967
        else:
            add =-0.19967
        if Fr<0.12:
            a2 = (0.0072+p[7]*Fr)
        else:
            a2 = (Fr**p[8]*np.exp(-p[9]*Fr))
        a3 = (1.+p[10]*np.atan((abs(delta))/Lpp))
        for lambd in wl: 
            w0 = np.sqrt((2*np.pi*g)/lambd)
            Vc = np.sqrt(g*lambd/2/np.pi)
            
            omega = (p[16]*(kyy**(1/3))*np.sqrt(Lpp/lambd)*(1-(p[17]/Cb)*(np.log(Bwl/tmax)-np.log(p[18])))*(Cb/p[19])**p[20]*((-p[21]*Fr**2+p[22]*Fr)*abs(np.cos(alp))+(p[23]*(13+np.cos(2*alp)))/p[24]))
            if omega < 1:
                b1 = p[11]
            else:
                b1 = -p[12]

            if omega < 1:
                d1 = p[13]*(Lpp*Cb/Bwl)**(-p[14])
            else:
                d1 = -p[13]*(Lpp/Bwl)**(-p[14])*(4-p[15]*np.atan2(abs(delta),Lpp))

        #f(alpha)
            if np.degrees(np.pi -E1) <= dr and dr<= np.degrees(np.pi):
                fofa = -np.cos(alp)
            elif alp < np.pi -E1:
                fofa = 0

            a1 = p[2]*Cb**p[3]*(p[4]*kyy)**2*(p[5]/Cb)**(-(1+Fr)*np.cos(alp))*np.log(Bwl/tmax)**(-1)*(1-2*np.cos(alp))/p[6]
            a90 = p[2]*Cb**p[3]*(p[4]*kyy)**2*(p[5]/Cb)**(-(1+Fr)*np.cos(0.5*np.pi))*np.log(Bwl/tmax)**(-1)*(1-2*np.cos(0.5*np.pi))/p[6]

        #for following waves
            if U<(Vc/2):
                a20 = (-0.0072+0.0072*4/Vc*U) *(0.87/Cb)
            elif (U-Vc/2)/np.sqrt(g*Lpp)<=0.12:
                a20 = (0.0072+p[7]*(U-Vc/2)/np.sqrt(g*Lpp)) *(0.87/Cb)
            else:
                a20 = ((U-Vc/2)/np.sqrt(g*Lpp))**1.5*np.exp(-3.5*(U-Vc/2)/np.sqrt(g*Lpp))*(0.87/Cb)**(1+(U-Vc/2)/np.sqrt(g*Lpp))
            fnr = (U-Vc/2)/np.sqrt(g*Lpp)
            a10 = p[2]*Cb**1.34*(4*kyy)**2*np.log(Bwl/tmax)**(-1)

            if dr == 0:
                rawm = p[0]*a10*a20*a3*omega**(b1)*np.exp((b1/d1)*(1-omega**(d1)))
            elif dr>0 and dr <90:
                rawm0 = p[0]*a90*a2*a3*omega**(b1)*np.exp((b1/d1)*(1-omega**(d1)))
                rawm1 = p[0]*a10*a20*a3*omega**(b1)*np.exp((b1/d1)*(1-omega**(d1)))
                rawm = rawm0+(rawm1-rawm0)*(1-dr/90)
            else:
                rawm = p[0]*a1*a2*a3*omega**(b1)*np.exp((b1/d1)*(1-omega**(d1)))
        #alpha & tstar

            tstar= tmax
            if lambd/Lpp <= 2.5:
                alpha12 = 1-np.exp(-p[25]*np.pi*(tstar/lambd - tstar/(p[26]*Lpp)))
            else:
                #alpha12 = 0 #adjustment compared to original paper Liu & Papanikolaou (2020)
                alpha12 = 1-np.exp(-p[25]*np.pi*(tstar/lambd - tstar/(p[26]*Lpp)))


            tstar= (tmax*(p[27]+np.sqrt(abs(np.cos(alp)))))/p[28]

            if lambd/Lpp <= 2.5:
                alpha34 = 1-np.exp(-p[25]*np.pi*(tstar/lambd - tstar/(p[26]*Lpp)))
            else:
                #alpha34 = 0 #adjustment compared to original paper Liu & Papanikolaou (2020)
                alpha34 = 1-np.exp(-p[25]*np.pi*(tstar/lambd - tstar/(p[26]*Lpp)))

        #segment 1
            if np.degrees(E1) <= dr and dr<= np.degrees(np.pi):
                rawr1= p[1]*(Lpp/Bwl)*alpha12*((np.sin(E1-alp)**2)+2*w0*U/g*(np.cos(E1)*np.cos(E1-alp)-np.cos(alp)))*(0.87/Cb)**((1+4*np.sqrt(Fr))*fofa)
            else:
                rawr1 = 0

        #segment 2
            if np.degrees(np.pi-E1) <= dr and dr<= np.degrees(np.pi):
                rawr2= p[1]*(Lpp/Bwl)*alpha12*((np.sin(E1+alp)**2)+2*w0*U/g*(np.cos(E1)*np.cos(E1+alp)-np.cos(alp)))*(0.87/Cb)**((1+4*np.sqrt(Fr))*fofa)
            else:
                rawr2 = 0

        #segment 3
            if 0 <= dr and dr<= np.degrees(np.pi-E2):
                rawr3= -p[1]*(Lpp/Bwl)*alpha34*((np.sin(E2+alp)**2)+2*w0*U/g*(np.cos(E2)*np.cos(E2+alp)-np.cos(alp)))
            else:
                rawr3 = 0

        #segment 4          
            if 0 <= dr and dr<=np.degrees(E2):  
                rawr4= -p[1]*(Lpp/Bwl)*alpha34*((np.sin(E2-alp)**2)+2*w0*U/g*(np.cos(E2)*np.cos(E2-alp)-np.cos(alp)))
            else:
                rawr4 = 0
            rawr = rawr1 + rawr2 +rawr3 +rawr4
            raw =rawr+rawm
            raw_local.append(raw+add)
        raw_arr.append(raw_local)

    #define output array
    w = np.sqrt(np.divide(2*np.pi*g,wl))
    wbar = np.divide(w, np.sqrt(g/Lpp))
    ll = np.divide(wl, Lpp)
    
    # exception for intersecting bounds in following seas
    if np.any(np.greater(raw_arr[2],raw_arr[1])):
        warnings.warn('Uncertainty bounds did intersect, fallback to simplified version')
        spread = max(raw_arr[1])-min(raw_arr[2])
        raw_arr1 = np.add(raw_arr[0], spread*0.5)
        raw_arr2 = np.add(raw_arr[0], -spread*0.5)
        ret_arr = [raw_arr[0],raw_arr1,raw_arr2,w, wbar, ll]
    else:
        ret_arr = [raw_arr[0],raw_arr[1],raw_arr[2],w, wbar, ll]
    return ret_arr

if __name__=="__main__":
    # from raw_formula import mm_raw
    from matplotlib import pyplot as plt
    # import numpy as np 
    
    # #kvlcc2
    # Lpp = 320
    # Bwl = 58
    # t = 20.80
    # Cb = 0.808
    # Le = 60
    # Lr = 60
    # kyy = 0.25
    # Fr = 0.142
    # lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    # for angle in [180,120,90,30,0]:
    #     arr = mm_raw(lambd, angle, Lpp, Bwl, t, t, Cb, Le, Lr, kyy, Fr)
    #     plt.plot(arr[5],arr[0], label= str(angle)+' deg.')
    #     plt.fill_between(arr[5], arr[1],arr[2], alpha=0.1)
    # plt.legend()
    # plt.grid()
    # plt.xlim((0,2.7))
    # plt.title('KVLCC-2; Fn=0.142')
    # plt.ylabel(r'$C_{AW}$ [-]')
    # plt.xlabel(r'$\lambda/Lpp$ [-]')
    # plt.show()
    
    #Scb-84
    Lpp = 178.
    t = 11.57
    Bwl= 32.26
    Cb = 0.84
    Le = 35
    Lr = 45
    kyy=0.25
    Fr = 0.099
    lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    for angle in [180,120,90,30,0]:
        arr = mm_raw(lambd, angle, Lpp, Bwl, t,t,Cb, Le,Lr, kyy,Fr)
        plt.plot(arr[4],arr[0], label= str(angle)+' deg.')
        plt.fill_between(arr[4], arr[1],arr[2], alpha=0.1)
    plt.legend()
    plt.grid()
    plt.xlim((1.5,4.5))
    plt.title('SCb-84; Fn=0.1')
    plt.ylabel(r'$C_{AW}$ [-]')
    plt.xlabel(r'$\omega/ \sqrt{g/L_{pp}}$ [-]')
    plt.show()
     
    # #dtc
    # Lpp = 355
    # dr = 60
    # t = 14.5
    # Bwl= 51
    # Cb = 0.661
    # Le= 112
    # Lr = 112
    # Fr = 0.139
    # kyy = 0.27
    # lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    # for angle in [180,120,90,30,0]:
    #     arr = mm_raw(lambd,angle, Lpp, Bwl, t,t,Cb, Le,Lr, kyy,Fr)
    #     plt.plot(arr[3],arr[0], label= str(angle)+' deg.')
    #     plt.fill_between(arr[3], arr[1],arr[2], alpha=0.1)
    # plt.legend()
    # plt.grid()
    # plt.xlim((0.2,1))
    # plt.title('DTC; Fn=0.139')
    # plt.ylabel(r'$C_{AW}$ [-]')
    # plt.xlabel(r'$\omega$ [-]')
    # plt.show()
    
    #%%
    
    #neoliner origin
    stw = 10.
    Lpp = 136
    t = 5.5
    Bwl= 24
    Cb = 0.613
    Le= Lpp/3.
    Lr = Lpp/3.
    Fr = stw*.5144 / np.sqrt(9.8066 * Lpp)
    kyy = 0.25
    lambd = np.linspace(0.11*Lpp, 3.*Lpp, 100)
    
    for angle in [45, 30, 15, 0]:
    # for angle in [180,120,90,30,0]:
        arr = mm_raw(lambd, angle, Lpp, Bwl, t, t, Cb, Le, Lr, kyy, Fr)
        plt.plot(arr[3], arr[0], label= str(angle)+' deg.')
        plt.fill_between(arr[3], arr[1], arr[2], alpha=0.1)
    plt.legend()
    # plt.grid()
    plt.title('Neoliner Origin; Fn=0.139')
    plt.ylabel(r'$C_{AW}$ [-]')
    plt.xlabel(r'$\omega$ [-]')
    plt.show()
