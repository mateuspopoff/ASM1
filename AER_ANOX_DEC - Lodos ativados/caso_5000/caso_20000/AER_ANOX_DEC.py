import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
import math as mt
#entrando com os dados .csv

file = 'parametros_cineticos.csv'
cinet_file = pd.read_csv(file, delimiter='\t')
file_2 = 'parametros_estoicometricos.csv'
estoic_file = pd.read_csv(file_2, delimiter='\t')
file_3 = 'valores_iniciais.csv'
vfeed_file = pd.read_csv(file_3, delimiter='\t')
file_4 = 'reatores.csv'
reator_file = pd.read_csv(file_4, delimiter='\t')
file_5 = 'tempo reator.csv'
tempo_file = pd.read_csv(file_5, delimiter='\t')

cinet_pivot = pd.pivot_table(cinet_file, index='CINETICO', values='VALOR')
estoic_pivot = pd.pivot_table(estoic_file, index='ESTOICOMETRICO', values='VALOR')
vfeed_pivot = pd.pivot_table(vfeed_file, index='VARIAVEIS', values='Feed')
reator_pivot = pd.pivot_table(reator_file, index='DIMEN.', values=['ANOX', 'AEROB'])
tempo_pivot = pd.pivot_table(tempo_file, index='t', values='delta')

tempo = tempo_pivot['delta']

#declarando nomeclatura do dimensionamento dos reatores
aerob = reator_pivot['AEROB']
aer_IN = aerob['INFLUEN']
aer_V = aerob['VOLUME']

anox = reator_pivot['ANOX']
an_IN = anox['INFLUEN']
an_V = anox['VOLUME']

#criando nomeclatura e declarando variaveis dos parametros
cinet = cinet_pivot['VALOR']
estoic = estoic_pivot['VALOR']

K_NH = cinet['K_NH']
K_NO = cinet['K_NO']
K_OA = cinet['K_OA']
K_OH = cinet['K_OH']
K_S = cinet['K_S']
K_X = cinet['K_X']
b_A = cinet['b_A']
b_H = cinet['b_H']
k_a = cinet['k_a']
k_h = cinet['k_h']
n_g = cinet['n_g']
n_h = cinet['n_h']
u_A = cinet['u_A']
u_H = cinet['u_H']
Y_A = estoic['Y_A']
Y_H = estoic['Y_H']
f_P = estoic['f_P']
i_XB = estoic['i_XB']
i_XE = estoic['i_XE']

#normalizando as grandezas dia pra segundo
u_H /= (24*60*60)
u_A /= (24*60*60)
b_H /= (24*60*60)
b_A /= (24*60*60)
k_a /= (24*60*60)
k_h /= (24*60*60)

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

#ANOXICO
vi_an_pivot = vfeed_pivot.copy()
vi_an_pivot['Feed'] = 0 #nao deve ser feito
vi_an_pivot['ANOXi'] = vi_an_pivot['Feed']
vi_an = vi_an_pivot['ANOXi'].astype(float)
vi_an_pivot.pop('Feed')

S_ALKi_an = vi_an['S_ALK']
S_Ii_an = vi_an['S_I']
S_NDi_an = vi_an['S_ND']
S_NHi_an = vi_an['S_NH']
S_NOi_an = vi_an['S_NO']
S_Oi_an = vi_an['S_O']
S_Si_an = vi_an['S_S']
X_BAi_an = vi_an['X_BA']
X_BHi_an = vi_an['X_BH']
X_Ii_an = vi_an['X_I']
X_NDi_an = vi_an['X_ND']
X_Si_an = vi_an['X_S']
X_Pi_an = vi_an['X_P']

dv_an_pivot = vfeed_pivot.copy() #copiando as linhas
dv_an_pivot['Feed'] = 0 # zerando valores
dv_an_pivot['dANOX'] = dv_an_pivot['Feed'] #criando nova coluna
dv_an = dv_an_pivot['dANOX'].astype(float) # simplificando o nome vf_pivot['VALORf']['S_I']
dv_an_pivot.pop('Feed') #excluindo a joluna VALORfeed da tabela vf

S_ALKd_an = dv_an['S_ALK']
S_Id_an = dv_an['S_I']
S_NDd_an = dv_an['S_ND']
S_NHd_an = dv_an['S_NH']
S_NOd_an = dv_an['S_NO']
S_Od_an = dv_an['S_O']
S_Sd_an = dv_an['S_S']
X_BAd_an = dv_an['X_BA']
X_BHd_an = dv_an['X_BH']
X_Id_an = dv_an['X_I']
X_NDd_an = dv_an['X_ND']
X_Sd_an = dv_an['X_S']
X_Pd_an = dv_an['X_P']

vf_an_pivot = vfeed_pivot.copy() #copiando as linhas
vf_an_pivot['Feed'] = 0 # zerando valores
vf_an_pivot['ANOXf'] = vf_an_pivot['Feed'] #criando nova coluna
vf_an = vf_an_pivot['ANOXf'].astype(float) # simplificando o nome vf_pivot['VALORf']['S_I']
vf_an_pivot.pop('Feed') #excluindo a joluna VALORfeed da tabela vf

S_ALKf_an = vf_an['S_ALK']
S_If_an = vf_an['S_I']
S_NDf_an = vf_an['S_ND']
S_NHf_an = vf_an['S_NH']
S_NOf_an = vf_an['S_NO']
S_Of_an = vf_an['S_O']
S_Sf_an = vf_an['S_S']
X_BAf_an = vf_an['X_BA']
X_BHf_an = vf_an['X_BH']
X_If_an = vf_an['X_I']
X_NDf_an = vf_an['X_ND']
X_Sf_an = vf_an['X_S']
X_Pf_an = vf_an['X_P']

#AERADOR
vi_aer_pivot = vfeed_pivot.copy()
vi_aer_pivot['Feed'] = 0 #nao deve ser feito
vi_aer_pivot['AERi'] = vi_aer_pivot['Feed']
vi_aer = vi_aer_pivot['AERi'].astype(float)
vi_aer_pivot.pop('Feed')

S_ALKi_aer = vi_aer['S_ALK']
S_Ii_aer = vi_aer['S_I']
S_NDi_aer = vi_aer['S_ND']
S_NHi_aer = vi_aer['S_NH']
S_NOi_aer = vi_aer['S_NO']
S_Oi_aer = vi_aer['S_O']
S_Si_aer = vi_aer['S_S']
X_BAi_aer = vi_aer['X_BA']
X_BHi_aer = vi_aer['X_BH']
X_Ii_aer = vi_aer['X_I']
X_NDi_aer = vi_aer['X_ND']
X_Si_aer = vi_aer['X_S']
X_Pi_aer = vi_aer['X_P']

dv_aer_pivot = vfeed_pivot.copy() #copiando as linhas
dv_aer_pivot['Feed'] = 0 # zerando valores
dv_aer_pivot['dAER'] = dv_aer_pivot['Feed'] #criando nova coluna
dv_aer = dv_aer_pivot['dAER'].astype(float) # simplificando o nome vf_pivot['VALORf']['S_I']
dv_aer_pivot.pop('Feed') #excluindo a joluna VALORfeed da tabela vf

S_ALKd_aer = dv_aer['S_ALK']
S_Id_aer = dv_aer['S_I']
S_NDd_aer = dv_aer['S_ND']
S_NHd_aer = dv_aer['S_NH']
S_NOd_aer = dv_aer['S_NO']
S_Od_aer = dv_aer['S_O']
S_Sd_aer = dv_aer['S_S']
X_BAd_aer = dv_aer['X_BA']
X_BHd_aer = dv_aer['X_BH']
X_Id_aer = dv_aer['X_I']
X_NDd_aer = dv_aer['X_ND']
X_Sd_aer = dv_aer['X_S']
X_Pd_aer = dv_aer['X_P']

vf_aer_pivot = vfeed_pivot.copy() #copiando as linhas
vf_aer_pivot['Feed'] = 0 # zerando valores
vf_aer_pivot['AERf'] = vf_aer_pivot['Feed'] #criando nova coluna
vf_aer = vf_aer_pivot['AERf'].astype(float) # simplificando o nome vf_pivot['VALORf']['S_I']
vf_aer_pivot.pop('Feed') #excluindo a joluna VALORfeed da tabela vf

S_ALKf_aer = vf_aer['S_ALK']
S_If_aer = vf_aer['S_I']
S_NDf_aer = vf_aer['S_ND']
S_NHf_aer = vf_aer['S_NH']
S_NOf_aer = vf_aer['S_NO']
S_Of_aer = vf_aer['S_O']
S_Sf_aer = vf_aer['S_S']
X_BAf_aer = vf_aer['X_BA']
X_BHf_aer = vf_aer['X_BH']
X_If_aer = vf_aer['X_I']
X_NDf_aer = vf_aer['X_ND']
X_Sf_aer = vf_aer['X_S']
X_Pf_aer = vf_aer['X_P']

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
def ProcessoAtt(bh, ba, nd, xs, xnd):
    H_grow  = u_H *h_S *h_OH *bh
    Han_grow= u_H *h_S *h_OHan *h_NO *n_g *bh
    A_grow  = u_A *h_NH *h_OA *ba
    decay_H = b_H *bh
    decay_A = b_A *ba
    Ammonia = k_a *nd *bh
    hydro_org = k_h *h_SBH *(h_OH + n_h *h_OHan *h_NO) *bh
    hydro_N = hydro_org *(xnd / xs)
    #print(nd/xs, nd, xs)
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

    tS_list = np.zeros(tSi_list.size) + 999
    tX_list = np.zeros(tXi_list.size) + 999
    
    #if dv['S_O'] != 0:
    #    tO = abs(-vi['S_O']/dv['S_O'])
    '''
    if vi['S_O'] == 0 and dv['S_O'] < 0:
        tO = 0
    elif dv['S_O'] != 0:
        tO = abs(-vi['S_O']/dv['S_O'])
    else:
        tO = 60
    
    for i in np.arange(0, tSi_list.size):
        if tSi_list[i] < 0:
            if tSd_list[i] < 0:
                tS_list[i] = 0
                #print('S', i)
            else:
                tS_list[i] = 60
            continue
        if tSd_list[i] >= 0:
            tS_list[i] = 60
        else:
            tS_list[i] = max(60, -tSi_list[i]/tSd_list[i])
        
        if tSd_list[i] >= 0 and tSd_list[i] < tSi_list[i]:
            #if not tSi_list[i] == 0 and not tSd_list[i] >= 0:
            tS_list = np.hstack(( tS_list, np.array([ abs(-tSi_list[i]/tSd_list[i]) ]) ))
        else:
            tS_list = np.hstack(( tS_list, np.array([0]) ))
        

    for i in np.arange(0, tXi_list.size):
        if tXi_list[i] < 0:
            if tXd_list[i] < 0:
                tX_list[i] = 0
                #print('X', i)
            else:
                tX_list[i] = 60
            continue
        if tXd_list[i] >= 0:
            tX_list[i] = 60
        else:
            tX_list[i] = max(60, -tXi_list[i]/tXd_list[i])
        
        if tXd_list[i] < 0 and abs(tXd_list[i]) > abs(tXi_list[i]):
            tX_list = np.hstack(( tX_list, np.array([0]) ))
        elif tXd_list[i] != 0:
            tX_list = np.hstack(( tX_list, np.array([ abs(-tXi_list[i]/tXd_list[i]) ]) ))
        
    '''
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
    #print(tX_list)
    #print(tS_list)
    #if not tS_list.size == 0:
    tS = min(tS_list)
    #else:
    #tS = 0
    #if not tX_list.size == 0:
    tX = min(tX_list)
    #else:
    #tX = 0
    if tO > 60: #evitando passos de tempo mais largos que o tempo de retenção
        tO = 60
    if tS > 60:
        tS = 60
    if tX > 60:
        tX = 60
    return [tX, tS, tO]

#calculando o menor tempo entre os 3 tempos do TimeStep
def Tmin(t_list):
    tmin=0
    #for i in range(len(t_list)): #check up dos zeros 
    #    if t_list[i] == 0:
    #        t_list[i] = 9999999999 #valor alto arbitrario pra nao bugar o loop
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



#passando de m³/dia para m³/s
Qin_an = an_IN/(60*60*24)
Qin_aer = aer_IN/(60*60*24) 

t = tempo[0]

plot_list_an = pd.DataFrame(np.array([vfeed_file['Feed'].values.copy()]), index= {'t0'}, \
                            columns= vfeed_file['VARIAVEIS'].values)
plot_list_aer = plot_list_an.copy()

#variavel erro
an_error = 1
aer_error = 1

#contador
tnum=0
#lista de tempos
t_list = []
a=0
b=0
ench_an =[0]
ench_aer =[0]

#lista do tempo
tempo_lista = pd.Series({'t0':t})

#lista de hiperboles e processos
hip_columns = ['h_S', 'h_OH', 'h_OHan', 'h_NO', 'h_NH', 'h_OA', 'h_SBH']
hip_index = {'t'}
hip_values = np.array([ [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN] ])
hiperbole_an_df = pd.DataFrame(hip_values.copy(), index= hip_index, columns = hip_columns)
hiperbole_aer_df = hiperbole_an_df.copy()
hiperbole_aer2_df = hiperbole_an_df.copy()

p_columns = ['H_grow', 'Han_grow', 'A_grow', 'decay_H', 'decay_A', 'Ammonia', 'hydro_org', 'hydro_N']
p_index = {'t'}
p_values = np.array([ [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN] ])
processo_an_df = pd.DataFrame(p_values.copy(), index = p_index, columns = p_columns)
processo_aer_df = processo_an_df.copy()
processo_aer2_df = processo_an_df.copy()

vf_aer = vfeed.copy()
vf_an = vfeed.copy()

##################################################################
##################################################################
##################################################################

tend1 = tnum

##################################################################
##################################################################
##################################################################



tend2 = tnum
t = tempo[2] # gerando valor arbitrário
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

#arranjo das proporções e unidades
Qin_aer = aer_IN/(60*60*24)
#Qreturn = an_IN/(60*60*24)
#QR = Qr*Qdn #Dont need Qr or Qw 
#QW = Qw*Qdn
#balanço de massa com retorno

QI = Qin_aer
QVI = Qr/(60*60*24) #QI*Qr
QII = QI + QVI
QIII = QII
QIV = QI * Qup
QV = QI * Qw
Qfeed, Qe, Qu = QIII, QIV, QV+QVI

xM, xm, fxM, fxm, xG, xg, gxG, gxg = Bissec()

flux_param = 'xM\t'+str(xM)+'\nxm\t'+str(xm)+'\nfxM\t'+str(fxM)+'\nfxm\t'+str(fxm)+\
             '\nxG\t'+str(xG)+'\nxg\t'+str(xg)+'\ngxG\t'+str(gxG)+'\ngxg\t'+str(gxg)
#q_vel = 'I\t'+str(QI)+'\nII\t'+str(QII)+'\nIII\t'+str(QIII)+'\nIV\t'+str(QIV)+'\nV\t'+\
#        str(QV)+'\nVI\t'+str(QVI)+'\nVII\t'+str(QVII)+'\nVIII\t'+str(QVIII)+'\nIX\t'+str(QIX)
q_vel = 'I\t'+ToHour(QI)+'\nII\t'+ToHour(QII)+'\nIII\t'+ToHour(QIII)+\
    '\nIV\t'+ToHour(QIV)+'\nV\t'+ToHour(QV)+'\nVI\t'+ToHour(QVI)

det_hid = 'ANOX\t'+str(an_V/FloatHour(QI+QVI))\
+'\nAER1\t'+str(aer_V/FloatHour(QIII))\
+'\nDEC\t'+str(V/(FloatHour(QIV)\
+FloatHour(QVI)+FloatHour(QV)))


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

while count < ncounts:

    #FEED AER
    vi_aer_pivot['AERi'] = (vf_aer*(aer_V-QII*t) +  vfeed*QI*t + vdn_dec*QVI*t)/aer_V
    vi_aer = vi_aer_pivot['AERi']
    S_ALKi_aer, S_Ii_aer, S_NDi_aer, S_NHi_aer, S_NOi_aer, S_Oi_aer, S_Si_aer, X_BAi_aer, X_BHi_aer, X_Ii_aer, X_NDi_aer, X_Si_aer, X_Pi_aer = Obter(vi_aer)
    S_Oi_aer = 2
    vi_aer['S_O'] = S_Oi_aer
    #calculando hiperboles, processos e diferencial aerobico
    h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH =\
         HiperboleAtt(S_Si_aer, S_Oi_aer, S_NOi_aer, S_NHi_aer, X_Si_aer, X_BHi_aer)
    #print('aer')
    H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N =\
            ProcessoAtt(X_BHi_aer, X_BAi_aer, S_NDi_aer, X_Si_aer, X_NDi_aer)
    S_ALKd_aer, S_Id_aer, S_NDd_aer, S_NHd_aer, S_NOd_aer, S_Od_aer, S_Sd_aer, X_BAd_aer, X_BHd_aer, X_Id_aer, X_NDd_aer, X_Sd_aer, X_Pd_aer = ProcessoTotal()
    Atualizar(dv_aer, S_ALKd_aer, S_Id_aer, S_NDd_aer, S_NHd_aer, S_NOd_aer, S_Od_aer, S_Sd_aer, X_BAd_aer, X_BHd_aer, X_Id_aer, X_NDd_aer, X_Sd_aer, X_Pd_aer)
    processo_aer_df = pd.concat([processo_aer_df, \
                                pd.DataFrame({'t'+str(tnum):np.array(\
                                [H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N])},\
                                index = p_columns).T ])

    #AER ANOX
    vi_an_pivot['ANOXi'] = (vf_an*(an_V-QIII*t) + vf_aer*QII*t)/an_V 
    vi_an = vi_an_pivot['ANOXi']
    S_ALKi_an, S_Ii_an, S_NDi_an, S_NHi_an, S_NOi_an, S_Oi_an, S_Si_an, X_BAi_an, X_BHi_an, X_Ii_an, X_NDi_an, X_Si_an, X_Pi_an = Obter(vi_an)
    #S_NOi_an = 2
    #vi_an['S_NO'] = S_NOi_an
    #S_Oi_an = 0
    #vi_aer['S_O'] = S_Oi_an
    #calculando hiperboles, processos e diferencial anoxico
    h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH =\
         HiperboleAtt(S_Si_an, S_Oi_an, S_NOi_an, S_NHi_an, X_Si_an, X_BHi_an)
    H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N =\
            ProcessoAtt(X_BHi_an, X_BAi_an, S_NDi_an, X_Si_an, X_NDi_an)
    S_ALKd_an, S_Id_an, S_NDd_an, S_NHd_an, S_NOd_an, S_Od_an, S_Sd_an, X_BAd_an, X_BHd_an, X_Id_an, X_NDd_an, X_Sd_an, X_Pd_an = ProcessoTotal()
    Atualizar(dv_an, S_ALKd_an, S_Id_an, S_NDd_an, S_NHd_an, S_NOd_an, S_Od_an, S_Sd_an, X_BAd_an, X_BHd_an, X_Id_an, X_NDd_an, X_Sd_an, X_Pd_an)
    processo_an_df = pd.concat([processo_an_df, \
                                 pd.DataFrame({'t'+str(tnum):np.array(\
                                 [H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N])},\
                                              index = p_columns).T ])
    
    #DECANTADOR
    S_ALKf_an, S_If_an, S_NDf_an, S_NHf_an, S_NOf_an, S_Of_an, S_Sf_an, \
    X_BAf_an, X_BHf_an, X_If_an, X_NDf_an, X_Sf_an, X_Pf_an = Obter(vf_an)
    Xfeed = (X_BAf_an + X_BHf_an)*0.9 + (X_If_an + X_Pf_an + X_Sf_an)*0.75
    xfeed_list[count] = Xfeed
    Sfeed = S_ALKf_an + S_If_an + S_NDf_an + S_NHf_an + S_NOf_an + S_Of_an + S_Sf_an
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
    AerDec(vaer_dec, vf_an.copy())
    CAtt(Cdecfeed, Xfeed, vaer_dec, Xfeed)
    SAtt(Sdecfeed, Sfeed, vaer_dec, Sfeed)
    CSAtt(vdec_feed, Cdecfeed, Sdecfeed) #g/(m²*s)
    
    v_dec_pivot['DEC'] += vdec_feed*QIII*t
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
    #print('antes:', vdn_dec)
    ConvAer(vdn_dec)
    #print('depois:', vdn_dec)
    #print('-------------------------------')
    #vdn_dec_pivot['DECdn'] = vdn_dec
    #vup_dec_pivot['DECup'] = vup_dec
    #v_dec_pivot['DEC'] = v_dec

    #Att
    x = xfut
    xe_list[count] = xe
    xu_list[count] = xu
    z_plot_list[count] = np.array([x.copy()])
    #
    s = sfut
    se_list[count] = s[0]
    su_list[count] = s[ncam-1]
    zs_plot_list[count] = np.array([s.copy()])


    #t = Tmin(t_list)
    tempo_lista['t'+str(tnum)] = t
    
    #t_an = TimeStep(vi_an, dv_an)
    #t_an = Tmin(t_an)
    #t_aer = TimeStep(vi_aer, dv_aer)
    #t_aer = Tmin(t_aer)
    #t = min(t_an, t_aer)
    #aplicando método de EULER AN
    vf_an_pivot['ANOXf'] = vi_an + dv_an * t 
    vf_an = vf_an_pivot['ANOXf']
    #aplicando método de EULER AER
    vf_aer_pivot['AERf'] = vi_aer + dv_aer * t 
    vf_aer = vf_aer_pivot['AERf']
    
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

    plot_list_an = pd.concat([plot_list_an, \
                              pd.DataFrame({'t'+str(tnum): vf_an_pivot['ANOXf'].copy()}).T])
    plot_list_aer = pd.concat([plot_list_aer, \
                               pd.DataFrame({'t'+str(tnum): vf_aer_pivot['AERf'].copy()}).T])
    plot_list_vup = pd.concat([plot_list_vup, \
                                pd.DataFrame({'t'+str(tnum): vup_dec_pivot['DECup'].copy()}).T])
    plot_list_vdn = pd.concat([plot_list_vdn, \
                                pd.DataFrame({'t'+str(tnum): vdn_dec_pivot['DECdn'].copy()}).T])    
    plot_list_vdec = pd.concat([plot_list_vdec, \
                                pd.DataFrame({'t'+str(tnum): v_dec_pivot['DEC'].copy()}).T])

fim = time()
tempo_processo = fim - inicio
print('100%\nO tempo do loop foi de', tempo_processo, 'segundos...')
tend3 = tnum    

file_an = open('an.txt', 'w')
file_an.write(plot_list_an.to_string())
file_an.close()
file_aer = open('aer.txt', 'w')
file_aer.write(plot_list_aer.to_string())
file_aer.close()
file_tempo_lista = open('tempo_lista.txt', 'w')
file_tempo_lista.write(tempo_lista.to_string())
file_tempo_lista.close()
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
file_q_vel = open('q_vel.txt', 'w')
file_q_vel.write(str(q_vel))
file_q_vel.close()
file_det_hid = open('det_hid.txt', 'w')
file_det_hid.write(str(det_hid))
file_det_hid.close()


file_hip_aer2 = open('hiperbole an.txt', 'w')
file_hip_aer2.write(hiperbole_an_df.to_string())
file_hip_aer2.close()
file_proc_aer2 = open('process an.txt', 'w')
file_proc_aer2.write(processo_an_df.to_string())
file_proc_aer2.close()
file_hip_aer2 = open('hiperbole aer1.txt', 'w')
file_hip_aer2.write(hiperbole_aer_df.to_string())
file_hip_aer2.close()
file_proc_aer2 = open('process aer1.txt', 'w')
file_proc_aer2.write(processo_aer_df.to_string())
file_proc_aer2.close()


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

plot_list_aer.to_excel('aer.xlsx', engine='xlsxwriter')
plot_list_an.to_excel('an.xlsx', engine='xlsxwriter')

'''
fig, ax1 = plt.subplots(1,2)

x_axie = np.arange(0, tnum+1)

#ax1[1].plot(x_axie, plot_list_an['S_ALK'], label='S_ALK')
ax1[1].plot(x_axie, plot_list_an['S_I'], label='S_I')
#ax1[1].plot(x_axie, plot_list_an['S_ND'], label='S_ND')
ax1[1].plot(x_axie, plot_list_an['S_NH'], label='S_NH')
ax1[1].plot(x_axie, plot_list_an['S_NO'], label='S_NO')
#ax1[1].plot(x_axie, plot_list_an['S_O'], label='S_O')
ax1[1].plot(x_axie, plot_list_an['S_S'], label='S_S')
#ax1[1].plot(x_axie, plot_list_an['X_BA'], label='X_BA')
ax1[1].plot(x_axie, plot_list_an['X_BH'], label='X_BH')
ax1[1].plot(x_axie, plot_list_an['X_I'], label='X_I')
#ax1[1].plot(x_axie, plot_list_an['X_ND'], label='X_ND')
ax1[1].plot(x_axie, plot_list_an['X_S'], label='X_S')
#ax1[1].plot(x_axie, plot_list_an['X_P'], label='X_P')
#ax1[1].twinx().plot(x_axie, ench_an, label='V')
ax1[1].set_title('ANOX')
#ax1[1].twinx().set_ylabel('S_NO')
ax1[1].legend(loc='center right', framealpha = 1)
ax1[1].set_ylabel('Concentração (g/m³)')
ax1[1].set_xlabel('Passo de Tempo')
#ax1[1].grid()

#ax1[0].plot(x_axie, plot_list_aer['S_ALK'], label='S_ALK')
ax1[0].plot(x_axie, plot_list_aer['S_I'], label='S_I')
#ax1[0].plot(x_axie, plot_list_aer['S_ND'], label='S_ND')
ax1[0].plot(x_axie, plot_list_aer['S_NH'], label='S_NH')
ax1[0].plot(x_axie, plot_list_aer['S_NO'], label='S_NO')
#ax1[0].plot(x_axie, plot_list_aer['S_O'], label='S_O')
ax1[0].plot(x_axie, plot_list_aer['S_S'], label='S_S')
#ax1[0].plot(x_axie, plot_list_aer['X_BA'], label='X_BA')
ax1[0].plot(x_axie, plot_list_aer['X_BH'], label='X_BH')
ax1[0].plot(x_axie, plot_list_aer['X_I'], label='X_I')
#ax1[0].plot(x_axie, plot_list_aer['X_ND'], label='X_ND')
ax1[0].plot(x_axie, plot_list_aer['X_S'], label='X_S')
#ax1[0].plot(x_axie, plot_list_aer['X_P'], label='X_P')
#ax1[0].twinx().plot(x_axie, ench_aer, label='V')
ax1[0].set_title('AER')
#ax1[0].twinx().set_ylabel('S_NO')
#ax1[0].set_ylabel('Concentração (g/m³)')
ax1[0].set_xlabel('Passo de Tempo')
ax1[0].legend(loc='center right', framealpha = 1)
#ax1[0].grid()

plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(1,2)
x_axie3 = np.arange(x_start_dec_plot, tnum)

#ax1[0].plot(x_axie3, plot_list_vdec['S_ALK'], label='S_ALK')
ax1[0].plot(x_axie3, plot_list_vdn['S_I'], label='S_I')#*A/QVI/aer_V
#ax1[0].plot(x_axie3, plot_list_vdec['S_ND'], label='S_ND')
ax1[0].plot(x_axie3, plot_list_vdn['S_NH'], label='S_NH')
ax1[0].plot(x_axie3, plot_list_vdn['S_NO'], label='S_NO')
#ax1[0].plot(x_axie3, plot_list_vdec['S_O'], label='S_O')
ax1[0].plot(x_axie3, plot_list_vdn['S_S'], label='S_S')
#ax1[0].plot(x_axie3, plot_list_vdec['X_BA'], label='X_BA')
ax1[0].plot(x_axie3, plot_list_vdn['X_BH'], label='X_BH')
ax1[0].plot(x_axie3, plot_list_vdn['X_I'], label='X_I')
#ax1[0].plot(x_axie3, plot_list_vdec['X_ND'], label='X_ND')
ax1[0].plot(x_axie3, plot_list_vdn['X_S'], label='X_S')
#ax1[0].plot(x_axie3, plot_list_vdec['X_P'], label='X_P')
ax1[0].set_title('DECdn')
#ax1[0].set_ylabel('Concentração (g/m³)')
ax1[0].set_xlabel('Passo de Tempo')
ax1[0].legend(loc='center right', framealpha = 1)
#ax1[0].grid()

balanço_plot = np.arange(balanço_g.size)
ax1[1].plot(balanço_plot, balanço_g, label='Balanço de fluxo')
ax1[1].plot(balanço_plot, xu_list*Qu, label='xu*Qu')
ax1[1].plot(balanço_plot, xe_list*Qe, label='xe*Qe')
ax1[1].plot(balanço_plot, xfeed_list*Qfeed, label='xf*Qfeed')
ax1[1].legend(framealpha=1)

#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
plt.tight_layout()
plt.show()
'''
'''
fig, ax1 = plt.subplots(1,2)
x_axie = np.arange(0, tnum+1)
#ax1[0].plot(x_axie, plot_list_aer['S_ALK'], label='S_ALK')
#ax1[0].plot(x_axie, plot_list_aer['S_I'], label='S_I')
ax1[0].plot(x_axie, plot_list_aer['S_ND'], label='S_ND')
ax1[0].plot(x_axie, plot_list_aer['S_NH'], label='S_NH')
ax1[0].plot(x_axie, plot_list_aer['S_NO'], label='S_NO')
ax1[0].plot(x_axie, plot_list_aer['S_O'], label='S_O')
ax1[0].plot(x_axie, plot_list_aer['S_S'], label='S_S')
#ax1[0].plot(x_axie, plot_list_aer['X_BA'], label='X_BA')
ax1[0].plot(x_axie, plot_list_aer['X_BH'], label='X_BH')
#ax1[0].plot(x_axie, plot_list_aer['X_I'], label='X_I')
ax1[0].plot(x_axie, plot_list_aer['X_ND'], label='X_ND')
ax1[0].plot(x_axie, plot_list_aer['X_S'], label='X_S')
#ax1[0].plot(x_axie, plot_list_aer['X_P'], label='X_P')
ax1[0].set_title('AER')
ax1[0].set_ylabel('g/m³')
ax1[0].set_xlabel('time')
ax1[0].legend()
ax1[0].grid()
ax1[1].plot(x_axie, processo_aer_df['H_grow'], label='H_grow')
ax1[1].plot(x_axie, processo_aer_df['Han_grow'], label='Han_grow')
#ax1[1].plot(x_axie, processo_aer_df['A_grow'], label='A_grow')
ax1[1].plot(x_axie, processo_aer_df['decay_H'], label='decay_H')
ax1[1].plot(x_axie, processo_aer_df['Ammonia'], label='Ammonia')
ax1[1].plot(x_axie, processo_aer_df['hydro_org'], label='hydro_org')
ax1[1].plot(x_axie, processo_aer_df['hydro_N'], label='hydro_N')
ax1[1].set_title('PROCESSES')
ax1[1].set_ylabel('process')
ax1[1].set_xlabel('time')
ax1[1].legend()
ax1[1].grid()
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(1,2)
x_axie = np.arange(0, tnum+1)
#ax1[0].plot(x_axie, plot_list_an['S_ALK'], label='S_ALK')
#ax1[0].plot(x_axie, plot_list_an['S_I'], label='S_I')
ax1[0].plot(x_axie, plot_list_an['S_ND'], label='S_ND')
ax1[0].plot(x_axie, plot_list_an['S_NH'], label='S_NH')
ax1[0].plot(x_axie, plot_list_an['S_NO'], label='S_NO')
#ax1[0].plot(x_axie, plot_list_an['S_O'], label='S_O')
ax1[0].plot(x_axie, plot_list_an['S_S'], label='S_S')
#ax1[0].plot(x_axie, plot_list_an['X_BA'], label='X_BA')
ax1[0].plot(x_axie, plot_list_an['X_BH'], label='X_BH')
#ax1[0].plot(x_axie, plot_list_an['X_I'], label='X_I')
ax1[0].plot(x_axie, plot_list_an['X_ND'], label='X_ND')
ax1[0].plot(x_axie, plot_list_an['X_S'], label='X_S')
#ax1[0].plot(x_axie, plot_list_an['X_P'], label='X_P')
ax1[0].set_title('ANOX')
ax1[0].set_ylabel('g/m³')
ax1[0].set_xlabel('time')
ax1[0].legend()
ax1[0].grid()
ax1[1].plot(x_axie, processo_an_df['H_grow'], label='H_grow')
ax1[1].plot(x_axie, processo_an_df['Han_grow'], label='Han_grow')
#ax1[1].plot(x_axie, processo_an_df['A_grow'], label='A_grow')
ax1[1].plot(x_axie, processo_an_df['decay_H'], label='decay_H')
ax1[1].plot(x_axie, processo_an_df['Ammonia'], label='Ammonia')
ax1[1].plot(x_axie, processo_an_df['hydro_org'], label='hydro_org')
ax1[1].plot(x_axie, processo_an_df['hydro_N'], label='hydro_N')
ax1[1].set_title('PROCESSES')
ax1[1].set_ylabel('process')
ax1[1].set_xlabel('time')
ax1[1].legend()
ax1[1].grid()
plt.tight_layout()
plt.show()


plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
'''

'''
#PLOT DECANTADOR 2
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
#ax.plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
ax.set_xlabel('Altura [nº da camada]')
ax.set_ylabel('Passo de Tempo')
ax.set_zlabel('Concentração SS [g/m³]')
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
ax.plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
ax.set_xlabel('Altura [nº da camada]')
ax.set_ylabel('Passo de Tempo')
ax.set_zlabel('Concentração subst. sol. [g/m³]')
plt.tight_layout()
plt.show()

#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], flux_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], f_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], g_list, cmap=cm.coolwarm)

#PLOT DECANTADOR 1
fig, ax = plt.subplots()
#im = ax.imshow(z_plot_list.T, aspect ='auto', cmap = 'seismic')
Cs = ax.contourf(malha[1], malha[0], z_plot_list, levels=8, cmap = 'seismic',\
                 origin='upper')
ax.set_aspect('auto')
fig.colorbar(Cs, ax=ax, label='Concentração SS  [g/m³]')
ax.set(ylabel='Altura [nº da camada]')
ax.set(xlabel='Passo de Tempo')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
'''


#plt.tight_layout()
#plt.show()

x_axie = np.arange(0, tnum+1)
#FIGURA 1 POPULAÇAO BACTERIAS + CONSUMO DE SS
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x_axie, plot_list_aer['X_BH'], label='X_BH', color='green', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['S_S'], label='S_S', color='blue', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['X_S'], label='X_S', color='purple', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
#ax1.legend(loc='center right', framealpha = 1)
#ax2 = ax1.twinx()
ax2.plot(x_axie, processo_aer_df['H_grow']*60, label='Crescimento_BH', color='green', linestyle=':', linewidth=3)
#ax2.plot(x_axie, processo_aer_df['Han_grow'], label='Crescimento_BHanox', color='red', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['decay_H']*60, label='Decaimento_BH', color='black', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['hydro_org']*60, label='Hidrólise-DQO', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='upper right', framealpha = 1)
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()


#FIGURA 2 AMONIFICAÇAO E NITRIFICAÇÃO
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x_axie, plot_list_aer['X_BA'], label='X_BA', color='green', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['X_ND'], label='X_ND', color='purple', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['S_ND'], label='S_ND', color='blue', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['S_NH'], label='S_NH', color='red', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_aer['S_NO'], label='S_NO', color='black', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
#ax1.legend(loc='upper right', framealpha = 1)
#ax2 = ax1.twinx()
ax2.plot(x_axie, processo_aer_df['A_grow']*60, label='Crescimento_BA', color='green', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['Ammonia']*60, label='Amonificação', color='red', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['hydro_N']*60, label='Hidrólise-N', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='upper right', framealpha = 1)
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()


#FIGURA 3 CICLO DQO ANOX
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x_axie, plot_list_an['X_BH'], label='X_BH', color='green', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_S'], label='S_S', color='blue', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['X_S'], label='X_S', color='purple', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
#ax1.legend(loc='center right', framealpha = 1)
#ax2 = ax1.twinx()
#ax2.plot(x_axie, processo_an_df['H_grow']*60, label='Crescimento_BH', color='green', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['Han_grow']*60, label='Crescimento_BHanox', color='green', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_an_df['decay_H']*60, label='Decaimento_BH', color='black', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_an_df['hydro_org']*60, label='Hidrólise-DQO', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='center right', framealpha = 1)
ax2.legend(loc='upper right', framealpha = 1)
plt.tight_layout()
plt.show()

#FIGURA 4 CICLO N ANOX
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#ax1.plot(x_axie, plot_list_an['X_BA'], label='X_BA', color='green', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['X_ND'], label='X_ND', color='purple', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_ND'], label='S_ND', color='blue', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_NH'], label='S_NH', color='red', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_NO'], label='S_NO', color='black', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
#ax1.legend(loc='upper right', framealpha = 1)
#ax2 = ax1.twinx()
#ax2.plot(x_axie, processo_aer_df['A_grow']*60, label='Crescimento_BA', color='green', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['Ammonia']*60, label='Amonificação', color='red', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_aer_df['hydro_N']*60, label='Hidrólise-N', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='upper right', framealpha = 1)
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()

#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], z_plot_list, cmap=cm.coolwarm)
#plt.figure().gca(projection='3d').plot_surface(malha[0], malha[1], zs_plot_list, cmap=cm.coolwarm)
#plt.show()

#FIGURA 5 DECANTADOR DN
fig, ax1 = plt.subplots()
x_axie3 = np.arange(x_start_dec_plot, tnum)
#ax1[0].plot(x_axie3, plot_list_vdec['S_ALK'], label='S_ALK')
ax1.plot(x_axie3, plot_list_vdn['S_I'], label='S_I', linewidth=3)#*A/QVI/aer_V
#ax1[0].plot(x_axie3, plot_list_vdec['S_ND'], label='S_ND')
ax1.plot(x_axie3, plot_list_vdn['S_NH'], label='S_NH', linewidth=3)
ax1.plot(x_axie3, plot_list_vdn['S_NO'], label='S_NO', linewidth=3)
#ax1[0].plot(x_axie3, plot_list_vdec['S_O'], label='S_O')
ax1.plot(x_axie3, plot_list_vdn['S_S'], label='S_S', linewidth=3)
#ax1[0].plot(x_axie3, plot_list_vdec['X_BA'], label='X_BA')
ax1.plot(x_axie3, plot_list_vdn['X_BH'], label='X_BH', linewidth=3)
ax1.plot(x_axie3, plot_list_vdn['X_I'], label='X_I', linewidth=3)
#ax1[0].plot(x_axie3, plot_list_vdec['X_ND'], label='X_ND')
ax1.plot(x_axie3, plot_list_vdn['X_S'], label='X_S', linewidth=3)
#ax1[0].plot(x_axie3, plot_list_vdec['X_P'], label='X_P')
#plt.set_title('DECdn')
#ax1[0].set_ylabel('Concentração (g/m³)')
ax1.set_xlabel('Passo de Tempo')
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.legend(loc='center right', framealpha = 1)
#ax1[0].grid()
plt.show()
