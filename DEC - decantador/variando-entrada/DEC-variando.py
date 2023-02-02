import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
import math as mt
#entrando com os dados .csv

file_2 = 'parametros_estoicometricos.csv'
estoic_file = pd.read_csv(file_2, delimiter='\t')
file_3 = 'valores_iniciais.csv'
vfeed_file = pd.read_csv(file_3, delimiter='\t')
file_5 = 'tempo reator.csv'
tempo_file = pd.read_csv(file_5, delimiter='\t')

file_999 = 'Q_IN.csv'
q_in = pd.read_csv(file_999, delimiter='\t')
Q_IN = q_in['Q']

estoic_pivot = pd.pivot_table(estoic_file, index='ESTOICOMETRICO', values='VALOR')
vfeed_pivot = pd.pivot_table(vfeed_file, index='VARIAVEIS', values='Feed')
tempo_pivot = pd.pivot_table(tempo_file, index='t', values='delta')

tempo = tempo_pivot['delta']

#criando nomeclatura e declarando variaveis das concentrações
vfeed = vfeed_pivot['Feed'].astype(float)

S_ALKfeed = vfeed['S_ALK']
S_Ifeed = vfeed['S_I']
S_NDfeed = vfeed['S_ND']
S_NHfeed = vfeed['S_NH']
S_NOfeed = vfeed['S_NO']
S_Ofeed = vfeed['S_O']
S_Sfeed = vfeed['S_S']
X_BAfeed = vfeed['X_BA']
X_BHfeed = vfeed['X_BH']
X_Ifeed = vfeed['X_I']
X_NDfeed = vfeed['X_ND']
X_Sfeed = vfeed['X_S']
X_Pfeed = vfeed['X_P']


#declarando dimensionamento do decantador
file_5 = 'dimens_decantador.csv'
dimen_file = pd.read_csv(file_5, delimiter='\t')
file_6 = 'parametros_decantador.csv'
param_file = pd.read_csv(file_6, delimiter='\t')

dimen_pivot = pd.pivot_table(dimen_file, index='DIMENSAO', values='VALOR')
param_pivot = pd.pivot_table(param_file, index='PARAMETRO', values='VALOR')

dimen = dimen_pivot['VALOR'].astype(float)

A = dimen['AREA']
H = dimen['ALTURA']
nfeed = dimen['FEED']
ncam = int(dimen['nCAM'])
Qup = dimen['Qup']
Qdn = dimen['Qdn']
Qr = dimen['Qr']
Qw = dimen['Qw']

param = param_pivot['VALOR'].astype(float)

v0 = param['v0']
v00 = param['v00']
rp = param['rp']
rh = param['rh']
Xmin = param['Xmin']

V = A*H
h=H/ncam
Vncam=A*h
#nfeed=int(ncam/2)-1
Ncam = H/ncam
nfeed = H-nfeed
nfeed = round(nfeed/Ncam)

v00 /= (24*60*60)
v0 /= (24*60*60)
e = mt.e 

xM, xm = 0,0
xG, xg = 0,0
fxM, fxm = 0,0
gxG, gxg = 0,0
xe,xu = 0,0

x=np.zeros(ncam)
s=x.copy()
xfut=x.copy()
sfut=x.copy()

tackas = lambda X: max(0, min(v00, v0* (e**(-rh*(X-Xmin)) - e**(-rp*(X-Xmin)))))

qup = lambda X: Qe*X/A
qdn = lambda X: Qu*X/A
S = lambda X: Qfeed*X/A
dfdx = lambda X: v0* (e**(-rh*(X-Xmin))*(1-X*rh) - e**(-rp*(X-Xmin))*(1-X*rp))+Qu/A
dgdx = lambda X: v0* (e**(-rh*(X-Xmin))*(1-X*rh) - e**(-rp*(X-Xmin))*(1-X*rp))-Qe/A

f = lambda x: x*tackas(x)+Qu*x/A
g = lambda x: x*tackas(x)-Qe*x/A



##################################################################
##################################################################
##################################################################



#obtendo dados da lista
def Obter(v):
    alk = v['S_ALK']
    si = v['S_I']
    snd = v['S_ND']
    nh = v['S_NH']
    no = v['S_NO']
    o = v['S_O']
    ss = v['S_S']
    ba = v['X_BA']
    bh = v['X_BH']
    xi = v['X_I']
    xnd = v['X_ND']
    xs = v['X_S']
    xp = v['X_P']
    return alk, si, snd, nh, no, o, ss, ba, bh, xi, xnd, xs, xp

#atualizador de dados
def Atualizar(v, alk, si, snd, nh, no, o, ss, ba, bh, xi, xnd, xs, xp):
    #print(alk, si, snd, nh, no, o, ss, ba, bh, xi, xnd, xs)
    v['S_ALK'] = alk
    v['S_I'] = si
    v['S_ND'] = snd
    v['S_NH'] = nh
    v['S_NO'] = no
    v['S_O'] = o
    v['S_S'] = ss
    v['X_BA'] = ba
    v['X_BH'] = bh
    v['X_I'] = xi
    v['X_ND'] = xnd
    v['X_S'] = xs
    v['X_P'] = xp
    #print(v)

#entrando com as hiperboles
def HiperboleAtt(ss, o, no, nh, xs, bh):
    h_S = ss / (K_S + ss)
    h_OH = o / (K_OH + o)
    h_OHan = K_OH / (o + K_OH)
    h_NO = no / (K_NO + no)
    h_NH = nh / (K_NH + nh)
    h_OA = o / (K_OA + o)
    if not bh == 0:
        h_SBH = (xs / bh) / (K_X + (xs / bh))
    else:
        h_SBH = 0
    return h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH

#taxa de processos
def ProcessoAtt(bh, ba, nd, xs):
    H_grow  = u_H *h_S *h_OH *bh
    Han_grow= u_H *h_S *h_OHan *h_NO *n_g *bh
    A_grow  = u_A *h_NH *h_OA *ba
    decay_H = b_H *bh
    decay_A = b_A *ba
    Ammonia = k_a *nd *bh
    hydro_org = k_h *h_SBH *(h_OH + n_h *h_OHan *h_NO) *bh
    hydro_N = hydro_org *(nd / xs)
    return H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N

#calculando produção e consumo
def ProcessoTotal():
    alk = -i_XB/14*H_grow+((1-Y_H)/(14*2.86*Y_H)-i_XB/14)*Han_grow+(-i_XB/14-1/(7*Y_A))*A_grow+1/14*Ammonia 
    si = 0 
    snd = -Ammonia+hydro_N 
    nh = -i_XB *(H_grow + Han_grow) + (-i_XB-1/Y_A)*A_grow + Ammonia 
    no = -1*((1-Y_H)/(2.86*Y_H))*Han_grow + 1/Y_A*A_grow 
    o = -1*((1-Y_H)/Y_H)*H_grow + (-1)*((4.57-Y_A)/Y_A)*A_grow 
    ss = (-1/Y_H) *(H_grow + Han_grow) + hydro_org 
    ba = A_grow - decay_A 
    bh = H_grow + Han_grow - decay_H 
    xi = 0 
    xnd = (i_XB-f_P*i_XB)*(decay_H+decay_A) - hydro_N
    xs = (1-f_P)*(decay_H+decay_A) - hydro_org
    xp = f_P*(decay_H+decay_A)
    return alk, si, snd, nh, no, o, ss, ba, bh, xi, xnd, xs, xp
    #print(alk, si, snd, nh, no, o, ss, ba, bh, xi, xnd, xs, xp)

#calculando o tempo de passo mais largo possível
def TimeStep(vi, dv): 
    tO, tS, tX = 0,0,0
    tSi_list = np.array([vi['S_ALK'], vi['S_I'], vi['S_ND'], vi['S_NH'], vi['S_NO'], vi['S_S']])
    tXi_list = np.array([vi['X_BA'], vi['X_BH'], vi['X_I'], vi['X_ND'], vi['X_S'], vi['X_P']]) 

    tSd_list = np.array([dv['S_ALK'], dv['S_I'], dv['S_ND'], dv['S_NH'], dv['S_NO'], dv['S_S']])
    tXd_list = np.array([dv['X_BA'], dv['X_BH'], dv['X_I'], dv['X_ND'], dv['X_S'], dv['X_P']])

    tS_list = np.array([])
    tX_list = np.array([])
    
    if dv['S_O'] != 0:
        tO = abs(-vi['S_O']/dv['S_O'])
    
    for i in np.arange(0, tSi_list.size):
        if tSd_list[i] != 0:
            if not tSi_list[i] == 0 and not tSd_list[i] >= 0:
                tS_list = np.hstack(( tS_list, np.array([ abs(-tSi_list[i]/tSd_list[i]) ]) ))

    for i in np.arange(0, tXi_list.size):
        if tXd_list[i] != 0:
            if not tXi_list[i] == 0 and not tXd_list[i] >= 0:
                tX_list = np.hstack(( tX_list, np.array([ abs(-tXi_list[i]/tXd_list[i]) ]) ))

    #print(tX_list)
    #print(tS_list)
    if not tS_list.size == 0:
        tS = min(tS_list)
    else:
        tS = 0
    if not tX_list.size == 0:
        tX = min(tX_list)
    else:
        tX = 0
    '''if tO > 1: #evitando passos de tempo mais largos que o tempo de retenção
        tO = 1
    if tS > 60:
        tS = 60'''
    if tX > 600:
        tX = 600
    return tX, tS, tO

#calculando o menor tempo entre os 3 tempos do TimeStep
def Tmin(t_list):
    tmin=0
    for i in range(len(t_list)): #check up dos zeros 
        if t_list[i] == 0:
            t_list[i] = 9999999999 #valor alto arbitrario pra nao bugar o loop
    tmin = min(t_list)
    #print('t min:', tmin)
    return tmin

#calculando erro
def ErroCalculo(lista, n):
    errors = []
    if not lista['S_ALK']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_ALK']['t'+str(n)] - lista['S_ALK']['t'+str(n-1)])/lista['S_ALK']['t'+str(n-1)]) )
    if not lista['S_I']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_I']['t'+str(n)] - lista['S_I']['t'+str(n-1)])/lista['S_I']['t'+str(n-1)]) )
    if not lista['S_ND']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_ND']['t'+str(n)] - lista['S_ND']['t'+str(n-1)])/lista['S_ND']['t'+str(n-1)]) )
    if not lista['S_NH']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_NH']['t'+str(n)] - lista['S_NH']['t'+str(n-1)])/lista['S_NH']['t'+str(n-1)]) )
    if not lista['S_NO']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_NO']['t'+str(n)] - lista['S_NO']['t'+str(n-1)])/lista['S_NO']['t'+str(n-1)]) )
    if not lista['S_O']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_O']['t'+str(n)] - lista['S_O']['t'+str(n-1)])/lista['S_O']['t'+str(n-1)]) )
    if not lista['S_S']['t'+str(n-1)] == 0:
        errors.append( abs((lista['S_S']['t'+str(n)] - lista['S_S']['t'+str(n-1)])/lista['S_S']['t'+str(n-1)]) )
    if not lista['X_BA']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_BA']['t'+str(n)] - lista['X_BA']['t'+str(n-1)])/lista['X_BA']['t'+str(n-1)]) )
    if not lista['X_BH']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_BH']['t'+str(n)] - lista['X_BH']['t'+str(n-1)])/lista['X_BH']['t'+str(n-1)]) )
    if not lista['X_I']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_I']['t'+str(n)] - lista['X_I']['t'+str(n-1)])/lista['X_I']['t'+str(n-1)]) )
    if not lista['X_ND']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_ND']['t'+str(n)] - lista['X_ND']['t'+str(n-1)])/lista['X_ND']['t'+str(n-1)]) )
    if not lista['X_S']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_S']['t'+str(n)] - lista['X_S']['t'+str(n-1)])/lista['X_S']['t'+str(n-1)]) )
    if not lista['X_P']['t'+str(n-1)] == 0:
        errors.append( abs((lista['X_P']['t'+str(n)] - lista['X_P']['t'+str(n-1)])/lista['X_P']['t'+str(n-1)]) )
    the_error = max(errors)
    return the_error

def PrintHipProc():
    hip_index = ['h_S', 'h_OH', 'h_OHan', 'h_NO', 'h_NH', 'h_OA', 'h_SBH']
    hip_series = [h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH]
    hiper_series = pd.Series(hip_series, index=hip_index)
    dict_hip = {'HIPERBOLES':hiper_series}
    hiperboles = pd.DataFrame(dict_hip) #criando DataFrame das hiperboles
    print(hiperboles)


    p_index = ['H_grow', 'Han_grow', 'A_grow', 'decay_H', 'decay_A', 'Ammonia', 'hydro_org', 'hydro_N']
    p_series = [H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N]
    proc_series = pd.Series(p_series, index=p_index)
    dict_proc = {'PROCESSOS':proc_series}
    processos = pd.DataFrame(dict_proc) #criando DataFrame dos processos com as hiperboles
    print(processos)

def ListHipProc(hip, proc, n):
    hip_columns = ['h_S', 'h_OH', 'h_OHan', 'h_NO', 'h_NH', 'h_OA', 'h_SBH']
    hip_index = {'t'+str(n)}
    hip_values = np.array([ [h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH] ])
    hiper_df = pd.DataFrame(hip_values, index = hip_index, columns = hip_columns)

    p_columns = ['H_grow', 'Han_grow', 'A_grow', 'decay_H', 'decay_A', 'Ammonia', 'hydro_org', 'hydro_N']
    p_index = {'t'+str(n)}
    p_values = np.array([ [H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N] ])
    proc_df = pd.DataFrame(p_values, index = p_index, columns = p_columns)

    return pd.concat([hip, hiper_df]), pd.concat([proc, proc_df])

def Bissec():
    ulti=0
    prox=0        
    for i in range(int(Xmin),200000):
        ulti = dfdx(i)
        prox = dfdx(i+1)
        if (prox > 0 and ulti < 0):
            xM = i
            fxM = f(xM)
            for j in range(int(Xmin),20000):
                ulti = f(j)
                prox = f(j+1)
                if (prox > fxM and ulti < fxM) or ulti == fxM:
                    xm = j
                    fxm = f(xm)
                    break
            break

    ulti=0
    prox=0        
    for i in range(0,200000):
        ulti = dgdx(i/100)
        prox = dgdx((i+1)/100)
        if (prox > 0 and ulti < 0):
            xG = i/100
            gxG = g(xG)
            for j in range(100,20000):
                ulti = g(j)
                prox = g(j+1)
                if (prox < gxG and ulti > gxG) or ulti == gxG:
                    xg = j
                    gxg = g(xg)
                    break
            break
    return xM, xm, fxM, fxm, xG, xg, gxG, gxg

def G(i):
    if i-1 == 0:
        return g(x[0])
    if x[i-1] <= x[i]:
        return min(g(x[i-1]), g(x[i]))
    else:
        return max(g(x[i-1]), g(x[i]))
    
def F(i):
    if i == ncam-2:
        return f(x[ncam-1])
    if x[i] <= x[i+1]:
        return min(f(x[i+1]), f(x[i]))
    else:
        return max(f(x[i+1]), f(x[i]))

def AerDec(vaer, v):
    vaer['S_ALK'] = v['S_ALK']
    vaer['S_I'] = v['S_I']
    vaer['S_ND'] = v['S_ND']
    vaer['S_NH'] = v['S_NH']
    vaer['S_NO'] = v['S_NO']
    vaer['S_O'] = v['S_O']
    vaer['S_S'] = v['S_S']
    vaer['X_BA'] = v['X_BA']*0.9
    vaer['X_BH'] = v['X_BH']*0.9
    vaer['X_I'] = v['X_I']*0.75
    vaer['X_ND'] = v['X_ND']*0.75
    vaer['X_S'] = v['X_S']*0.75
    vaer['X_P'] = v['X_P']*0.75

def ConvDec(v):
    v['X_BA']*=0.9
    v['X_BH']*=0.9
    v['X_I']*=0.75
    v['X_ND']*=0.75
    v['X_S']*=0.75
    v['X_P']*=0.75

def ConvAer(v):
    v['X_BA']/=0.9
    v['X_BH']/=0.9
    v['X_I']/=0.75
    v['X_ND']/=0.75
    v['X_S']/=0.75
    v['X_P']/=0.75

def CSAtt (v, C, S):
    v['S_ALK'] = S['S_ALK']
    v['S_I'] = S['S_I']
    v['S_ND'] = S['S_ND']
    v['S_NH'] = S['S_NH']
    v['S_NO'] = S['S_NO']
    v['S_O'] = S['S_O']
    v['S_S'] = S['S_S']
    v['X_BA'] = C['X_BA']
    v['X_BH'] = C['X_BH']
    v['X_I'] = C['X_I']
    v['X_ND'] = C['X_ND']
    v['X_S'] = C['X_S']
    v['X_P'] = C['X_P']

def SAtt (S, x, v, X):
    S['S_ALK'] = v['S_ALK']/X*x
    S['S_I'] = v['S_I']/X*x
    S['S_ND'] = v['S_ND']/X*x
    S['S_NH'] = v['S_NH']/X*x
    S['S_NO'] = v['S_NO']/X*x
    S['S_O'] = v['S_O']/X*x
    S['S_S'] = v['S_S']/X*x

def CAtt (C, x, v, X):
    C['X_BA'] = v['X_BA']/X*x
    C['X_BH'] = v['X_BH']/X*x
    C['X_I'] = v['X_I']/X*x
    C['X_ND'] = v['X_ND']/X*x
    C['X_S'] = v['X_S']/X*x
    C['X_P'] = v['X_P']/X*x

def ToHour(Q):
    return str(Q*60*60)

def FloatHour(Q):
    return Q*60*60


##################################################################
##################################################################
##################################################################


tnum = 0
tend2 = tnum
t = tempo[0] # gerando valor arbitrário
#verificando o processo e sua duração
inicio = time()
#verificando o processo e sua duração
ss=0
vc=0
cc=0
sc=0
#criando lista para plot
x_start_dec_plot = tnum


ncounts = int(dimen['loops'])
count=0


malha = np.meshgrid(np.arange(ncam), np.arange(ncounts))
z_plot_list = np.arange(ncam*ncounts).reshape(ncounts,ncam).astype(float)
zs_plot_list = z_plot_list.copy()

xe_list = np.zeros(ncounts)
xu_list = xe_list.copy()
se_list = xe_list.copy()
su_list = xe_list.copy()
xfeed_list = xe_list.copy()
sfeed_list = xe_list.copy()
balanço = xe_list.copy()
balanço_s = xe_list.copy()
balanço_g = xe_list.copy()

f_list = np.arange(ncam*ncounts).reshape(ncounts,ncam).astype(float)
g_list = f_list.copy()
fs_list = f_list.copy()
gs_list = f_list.copy()

f_flux = np.zeros(ncam)
g_flux = f_flux.copy()
fs_flux = f_flux.copy()
gs_flux = f_flux.copy()
fast_flux = f_flux.copy()
fast_s_flux = f_flux.copy()

flux_list = np.arange(ncam*ncounts).reshape(ncounts,ncam).astype(float)
flux_s_list = flux_list.copy()

#Q_IN = 1000.0
#arranjo das proporções e unidades
Q_in = Q_IN[0]/(60*60)
QR = Qr*Qdn 
QW = Qw*Qdn
#balanço de massa com retorno

def Qatt(value, dn_percent):
    QVII = (1-dn_percent)*value #Qup*value#Q_in
    QVIII = dn_percent*Qr*value #QR*value#Q_in
    QIX = dn_percent*Qw*value #QW*value#Q_in
    Qfeed, Qe, Qu = value, QVII, QVIII+QIX #Q_in, QVII, QVIII+QIX
    return QVII, QVIII, QIX, Qfeed, Qe, Qu
QVII, QVIII, QIX, Qfeed, Qe, Qu  = Qatt(Q_in, Qdn)

xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()

flux_param = 'xM\t'+str(xM)+'\nxm\t'+str(xm)+'\nfxM\t'+str(fxM)+'\nfxm\t'+str(fxm)+\
             '\nxG\t'+str(xG)+'\nxg\t'+str(xg)+'\ngxG\t'+str(gxG)+'\ngxg\t'+str(gxg)

C_index = list(vfeed.loc['X_BA':'X_S'].index)
C_columns = {'X'}
C_values2 = np.zeros(len(C_index))
C_pivot = pd.DataFrame(C_values2.copy(), index=C_index, columns=C_columns)
Cup = C_pivot['X']
Cdn = Cup.copy()
#Cdecup = Cup.copy()
#Cdecdn = Cup.copy()
Cdecfeed = Cup.copy()

S_index = list(vfeed.loc['S_ALK':'S_S'].index)
S_columns = {'S'}
S_values2 = np.zeros(len(list(S_index)))
S_pivot = pd.DataFrame(S_values2.copy(), index=S_index, columns=S_columns)
Sup = S_pivot['S']
Sdn = Sup.copy()
#Sdecup = Sup.copy()
#Sdecdn = Sup.copy()
Sdecfeed = Sup.copy()

v_index = list(vfeed.index)
v_values2 = np.zeros(len(v_index)).astype(float)
v_values = np.array([np.zeros(vfeed.size).astype(float)])
vup_columns = ['DECup']
vdn_columns = ['DECdn']
v_columns = ['DEC']
vaer_columns = ['AERdec']
vdn_dec_pivot = pd.DataFrame(v_values2.copy(), index=v_index, columns=vdn_columns)
vdn_dec = vdn_dec_pivot['DECdn']
vup_dec_pivot = pd.DataFrame(v_values2.copy(), index=v_index, columns=vup_columns)
vup_dec = vup_dec_pivot['DECup']
v_dec_pivot = pd.DataFrame(v_values2.copy(), index=v_index, columns=v_columns)
v_dec = v_dec_pivot['DEC']
vaer_dec_pivot = pd.DataFrame(v_values2.copy(), index=v_index, columns=vaer_columns)
vaer_dec = vaer_dec_pivot['AERdec']
#vdec_up = vup_dec.copy()
#vdec_dn = vdn_dec.copy()
vdec_feed = v_dec.copy()

plot_list_vup = pd.DataFrame(columns = v_index)
plot_list_vdn = pd.DataFrame(columns = v_index)
plot_list_vdec = pd.DataFrame(columns = v_index)

#xu = 0
#xu_err = 1
x[ nfeed: ] = xm
nmid = ncam - nfeed
x[ nfeed+round(nmid/2): ] = xM

while count < ncounts: #and xu_err > 1e-4:
    #xu_i = xu

    #DECANTADOR
    '''
    S_ALKf_aer2, S_If_aer2, S_NDf_aer2, S_NHf_aer2, S_NOf_aer2, S_Of_aer2, S_Sf_aer2, \
    X_BAf_aer2, X_BHf_aer2, X_If_aer2, X_NDf_aer2, X_Sf_aer2, X_Pf_aer2 = Obter(vfeed)
    Xfeed = (X_BAf_aer2 + X_BHf_aer2)*0.9 + (X_If_aer2 + X_Pf_aer2 + X_Sf_aer2)*0.75
    xfeed_list[count] = Xfeed
    Sfeed = S_ALKf_aer2 + S_If_aer2 + S_NDf_aer2 + S_NHf_aer2 + S_NOf_aer2 + S_Of_aer2 + S_Sf_aer2
    sfeed_list[count] = Sfeed
    '''
    if count > ncounts*3/4:
        Q_in = Q_IN[3]/(60*60)
        QVII, QVIII, QIX, Qfeed, Qe, Qu  = Qatt(Q_in, Qdn)
        xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()
        Xfeed = (xm+xM)*0.5
        Sfeed = (xg+xG)*0.5
    elif count > ncounts/2:
        Q_in = Q_IN[2]/(60*60)
        QVII, QVIII, QIX, Qfeed, Qe, Qu  = Qatt(Q_in, 0.333)
        xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()
        #Xfeed = 5000
        #Sfeed = 5000
    elif count > ncounts/4:
        #Q_in = Q_IN[1]/(60*60)
        #QVII, QVIII, QIX, Qfeed, Qe, Qu  = Qatt(Q_in)
        #xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()
        Xfeed = (xm+xM)*0.77
        Sfeed = (xg+xG)*0.77
    else:
        Q_in = Q_IN[0]/(60*60)
        QVII, QVIII, QIX, Qfeed, Qe, Qu  = Qatt(Q_in, Qdn)
        xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()
        Xfeed = (xm+xM)*0.5
        Sfeed = (xg+xG)*0.5
    xfeed_list[count] = Xfeed
    sfeed_list[count] = Sfeed
    
    #X
    xfut[nfeed] = x[nfeed] + t/h * (G(nfeed) - F(nfeed) + S(Xfeed))
    fast_flux[nfeed] = G(nfeed) - F(nfeed) + S(Xfeed)
    f_flux[nfeed] = F(nfeed)
    g_flux[nfeed] = -G(nfeed)
    #S
    sfut[nfeed] = s[nfeed] + t/h * (- qup(s[nfeed]) - qdn(s[nfeed]) + S(Sfeed))
    fast_s_flux[nfeed] = - qup(s[nfeed]) - qdn(s[nfeed]) + S(Sfeed)
    fs_flux[nfeed] = -qdn(s[nfeed])
    gs_flux[nfeed] = -qup(s[nfeed])
    for i in range(1,nfeed): #clarificador
        xfut[i] = x[i] + t/h * (G(i)-G(i+1))
        fast_flux[i] = G(i)-G(i+1)
        g_flux[i] = -G(i)
        #
        sfut[i]=s[i] + t/h * (qup(s[i+1])-qup(s[i]))
        fast_s_flux[i] = qup(s[i+1])-qup(s[i])
        gs_flux[i] =-qup(s[i])
    for i in range(nfeed+1, ncam-1): #adensador
        xfut[i] = x[i] + t/h * (F(i-1)-F(i))
        fast_flux[i] = F(i-1)-F(i)
        f_flux[i] = F(i)
        #
        sfut[i]=s[i] + t/h * (qdn(s[i-1])-qdn(s[i]))
        fast_s_flux[i] = qdn(s[i-1])-qdn(s[i])
        fs_flux[i]=-qdn(s[i])
    #
    if xG < x[1] and x[1] < xg:
        x[0] = xG
    else:
        x[0] = x[1]
    if xm < x[ncam-2] and x[ncam-2] < xM:
        x[ncam-1] = xM
    else:
        x[ncam-1] = x[ncam-2]
    xfut[0] = x[0]
    xfut[ncam-1] = x[ncam-1]
    #
    sfut[0]=s[1]
    sfut[ncam-1]=s[ncam-2]

    #DATA
    '''
    xe = x[0] - x[0]*tackas(x[0])*A/Qe
    xu = x[ncam-1] + x[ncam-1]*tackas(x[ncam-1])*A/Qu
    f_flux[ncam-1] = F(ncam-2)
    g_flux[0] = -G(1)
    #
    se = s[0]
    su = s[ncam-1]
    fs_flux[ncam-1] = qdn(s[ncam-2])
    gs_flux[0] = qup(s[1])

    balanço[count] = S(Xfeed)-F(ncam-2)+G(1)
    balanço_s[count] = Sfeed*Qfeed - su*Qu - se*Qe
    balanço_g[count] = Xfeed*Qfeed - xu*Qu - xe*Qe
    
    f_list[count] = f_flux
    g_list[count] = g_flux
    flux_list[count] = fast_flux
    #
    fs_list[count] = fs_flux
    gs_list[count] = gs_flux
    flux_s_list[count] = fast_s_flux
    
    #V_DEC
    AerDec(vaer_dec, vfeed.copy())
    CAtt(Cdecfeed, Xfeed, vaer_dec, Xfeed)
    SAtt(Sdecfeed, Sfeed, vaer_dec, Sfeed)
    CSAtt(vdec_feed, Cdecfeed, Sdecfeed) #g/(m²*s)
    
    v_dec_pivot['DEC'] += vdec_feed*Qfeed*t
    ConvDec(vup_dec)
    ConvDec(vdn_dec)

    X_dec = v_dec['X_BA']+v_dec['X_BH']+v_dec['X_I']+v_dec['X_P']+v_dec['X_S']
    #
    S_dec = v_dec['S_ALK']+v_dec['S_I']+v_dec['S_ND']+v_dec['S_NH']+v_dec['S_NO']\
            +v_dec['S_O']+v_dec['S_S']
    
    CAtt(Cup, xe, v_dec.copy(), X_dec)
    CAtt(Cdn, xu, v_dec.copy(), X_dec)
    #
    SAtt(Sup, se, v_dec.copy(), S_dec)
    SAtt(Sdn, su, v_dec.copy(), S_dec)
    
    CSAtt(vup_dec_pivot['DECup'], Cup, Sup)
    CSAtt(vdn_dec_pivot['DECdn'], Cdn, Sdn)
    vup_dec = vup_dec_pivot['DECup']
    vdn_dec = vdn_dec_pivot['DECdn']
    v_dec_pivot['DEC'] -= (vup_dec*Qe + vdn_dec*Qu)*t

    v_dec = v_dec_pivot['DEC']
    ConvAer(vup_dec)
    ConvAer(vdn_dec)
    '''
    #Att
    x = xfut
    xe_list[count] = xe
    xu_list[count] = xu
    z_plot_list[count] = np.array([x.copy()])
    #z_plot_list[count][0] = xe
    #z_plot_list[count][ncam-1] = xu
    #
    s = sfut
    se_list[count] = s[0]
    su_list[count] = s[ncam-1]
    zs_plot_list[count] = np.array([s.copy()])

    
    if count >= 0.75*ncounts and sc == 0:
        sc=1
        print('75%')
    elif count >= 0.5*ncounts and cc == 0:
        cc=1
        print('50%')
    elif count >= 0.25*ncounts and vc == 0:
        vc=1
        print('25%')
    elif count >= 0*ncounts and ss == 0:
        ss=1
        print('Start')

    #preparando lista do plot
    tnum+=1
    count+=1
    '''
    plot_list_vup = pd.concat([plot_list_vup, \
                                pd.DataFrame({'t'+str(tnum): vup_dec_pivot['DECup'].copy()}).T])
    plot_list_vdn = pd.concat([plot_list_vdn, \
                                pd.DataFrame({'t'+str(tnum): vdn_dec_pivot['DECdn'].copy()}).T])    
    plot_list_vdec = pd.concat([plot_list_vdec, \
                                pd.DataFrame({'t'+str(tnum): v_dec_pivot['DEC'].copy()}).T]) 

    '''
    #xu_err = np.abs((xu_i - xu)/xu)
    #if np.isnan(xu_err) == True:
    #    xu_err = 1


fim = time()
tempo_processo = fim - inicio
print('100%\nO tempo do loop foi de', tempo_processo, 'segundos...')
tend3 = tnum    

file_vup_dec_lista = open('vup_dec.txt', 'w')
file_vup_dec_lista.write(plot_list_vup.to_string())
file_vup_dec_lista.close()
file_vdn_dec_lista = open('vdn_dec.txt', 'w')
file_vdn_dec_lista.write(plot_list_vdn.to_string())
file_vdn_dec_lista.close()
file_v_dec_lista = open('v_dec.txt', 'w')
file_v_dec_lista.write(plot_list_vdec.to_string())
file_v_dec_lista.close()
file_flux_param = open('flux_param.txt', 'w')
file_flux_param.write(str(flux_param))
file_flux_param.close()


np.savetxt('xe.txt', xe_list, delimiter=',')
np.savetxt('xu.txt', xu_list, delimiter=',')
np.savetxt('z.txt', z_plot_list, delimiter=',')
np.savetxt('zs.txt', zs_plot_list, delimiter=',')
np.savetxt('xfeed.txt', xfeed_list, delimiter=',')
np.savetxt('sfeed.txt', sfeed_list, delimiter=',')
np.savetxt('flux_list.txt', flux_list, delimiter=',')
np.savetxt('f_list.txt', f_list, delimiter=',')
np.savetxt('g_list.txt', g_list, delimiter=',')
np.savetxt('balanço_solidos.txt', balanço, delimiter=',')
np.savetxt('balanço_liquidos.txt', balanço_s, delimiter=',')
np.savetxt('balanço_massa.txt', balanço_g, delimiter=',')
'''
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
x_axie = np.arange(0, tnum+1)
x_axie2 = np.arange(x_start_aer2_plot, tnum+1)
x_axie4 = np.arange(x_start_dec_plot, tnum)
ax1.plot(x_axie4, xu_list, label='xu', color = 'blue')
ax1.plot(x_axie4, xfeed_list, label='xfeed', color = 'orange')
ax1.set_title('DEC')
ax1.set(ylabel='X')
#ax2.set(ylabel='xe')
ax1.legend()
#ax2.legend()
'''

#primeiras configurações pro plot

#PLOT DECANTADOR 2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
#ax.plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
ax.set_xlabel('nº de camada')
ax.set_ylabel('Passo de Tempo')
ax.set_zlabel('Concentração SS [g/m³]')
plt.tight_layout()
plt.show
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], flux_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], f_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], g_list, cmap=cm.coolwarm)

#PLOT DECANTADOR 1
fig, ax = plt.subplots()
#im = ax.imshow(z_plot_list.T, aspect ='auto', cmap = 'seismic')
Cs = ax.contourf(malha[1], malha[0], z_plot_list, levels=8, cmap = 'seismic',\
                 origin='upper')
ax.set_aspect('auto')
fig.colorbar(Cs, ax=ax, label='Concentração SS [g/m³]')
ax.set(ylabel='nº de camada')
ax.set(xlabel='Passo de Tempo')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

balanço_plot = np.arange(ncounts)
plt.plot(balanço_plot, balanço_g, label='Balanço de fluxo')
plt.plot(balanço_plot, xu_list*Qu, label='xu*Qu')
plt.plot(balanço_plot, xe_list*Qe, label='xe*Qe')
plt.plot(balanço_plot, xfeed_list*Qfeed, label='xf*Qfeed')
plt.legend(framealpha=1)
plt.show()

