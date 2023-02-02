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
reator_pivot = pd.pivot_table(reator_file, index='DIMEN.', values=['ANOX'])
tempo_pivot = pd.pivot_table(tempo_file, index='t', values='delta')
tempo = tempo_pivot['delta']

#declarando nomeclatura do dimensionamento dos reatores
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

def FloatHour(Q):
    return Q*60*60


#contador universal
tnum=1
#lista de hiperboles e processos
hip_columns = ['h_S', 'h_OH', 'h_OHan', 'h_NO', 'h_NH', 'h_OA', 'h_SBH']
hip_index = {'t'}
hip_values = np.array([ [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN] ])
hiperbole_an_df = pd.DataFrame(hip_values.copy(), index= hip_index, columns = hip_columns)
p_columns = ['H_grow', 'Han_grow', 'A_grow', 'decay_H', 'decay_A', 'Ammonia', 'hydro_org', 'hydro_N']
p_index = {'t'}
p_values = np.array([ [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN] ])
processo_an_df = pd.DataFrame(p_values.copy(), index = p_index, columns = p_columns)

plot_list_an = pd.DataFrame(np.array([vfeed_file['Feed'].values.copy()]), index= {'t0'}, \
                            columns= vfeed_file['VARIAVEIS'].values)

t = tempo[0] # espaço de tempo
Qin_an = an_IN/(60*60*24)
det_hid = 'ANOX\t'+str(an_V/FloatHour(Qin_an))+'hours'

error = 1

vf_an = vfeed.copy() #condiçao inicial do reator = condição de entrada

counts = 0
ncounts = 2000

while counts < ncounts:
    vi_an_pivot['ANOXi'] = vf_an+((vfeed-vf_an)*Qin_an*t)/an_V
    vi_an = vi_an_pivot['ANOXi']
    
    S_ALKi_an, S_Ii_an, S_NDi_an, S_NHi_an, S_NOi_an, S_Oi_an, S_Si_an, X_BAi_an, X_BHi_an, X_Ii_an, X_NDi_an, X_Si_an, X_Pi_an = Obter(vi_an)
    
    #Adição externa de Nitrito e Nitrato
    #S_NOi_an = 2
    #vi_an['S_NO'] = S_NOi_an

    #calculando hiperboles, processos e diferencial anoxico
    h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH = HiperboleAtt(S_Si_an, S_Oi_an, S_NOi_an, S_NHi_an, X_Si_an, X_BHi_an) 
    H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N = ProcessoAtt(X_BHi_an, X_BAi_an, S_NDi_an, X_Si_an, X_NDi_an)
    S_ALKd_an, S_Id_an, S_NDd_an, S_NHd_an, S_NOd_an, S_Od_an, S_Sd_an, X_BAd_an, X_BHd_an, X_Id_an, X_NDd_an, X_Sd_an, X_Pd_an = ProcessoTotal()
    Atualizar(dv_an, S_ALKd_an, S_Id_an, S_NDd_an, S_NHd_an, S_NOd_an, S_Od_an, S_Sd_an, X_BAd_an, X_BHd_an, X_Id_an, X_NDd_an, X_Sd_an, X_Pd_an)

    #hiperbole_aer_df, processo_aer_df = ListHipProc(hiperbole_aer_df, processo_aer_df, tnum)

    e = vf_an
    vf_an_pivot['ANOXf'] = vi_an + dv_an * t 
    vf_an = vf_an_pivot['ANOXf']
    e -= vf_an
    error = e.max()

    plot_list_an = pd.concat([plot_list_an, \
                               pd.DataFrame({'t'+str(tnum):\
                                             vf_an_pivot['ANOXf'].copy()}).T])
    hiperbole_an_df = pd.concat([hiperbole_an_df, \
                                  pd.DataFrame({'t'+str(tnum):np.array(\
                                              [h_S, h_OH, h_OHan, h_NO, h_NH, h_OA, h_SBH])},\
                                              index = hip_columns).T ])
    processo_an_df = pd.concat([processo_an_df, \
                                 pd.DataFrame({'t'+str(tnum):np.array(\
                                 [H_grow, Han_grow, A_grow, decay_H, decay_A, Ammonia, hydro_org, hydro_N])},\
                                              index = p_columns).T ])
    tnum+=1
    counts+=1

file_an = open('an.txt', 'w')
file_an.write(plot_list_an.to_string())
file_an.close()
file_det_hid = open('det_hid.txt', 'w')
file_det_hid.write(str(det_hid))
file_det_hid.close()

file_hip_aer2 = open('hiperbole an.txt', 'w')
file_hip_aer2.write(hiperbole_an_df.to_string())
file_hip_aer2.close()
file_proc_aer2 = open('process an.txt', 'w')
file_proc_aer2.write(processo_an_df.to_string())
file_proc_aer2.close()

plot_list_an.to_excel('an.xlsx', engine='xlsxwriter')
x_axie = np.arange(0, tnum)
'''
#FIGURA 0 PADRAO
fig, ax1 = plt.subplots(1,2)
ax1[0].plot(x_axie, plot_list_an['S_ALK'], label='S_ALK')
ax1[0].plot(x_axie, plot_list_an['S_I'], label='S_I')
ax1[0].plot(x_axie, plot_list_an['S_ND'], label='S_ND')
ax1[0].plot(x_axie, plot_list_an['S_NH'], label='S_NH')
ax1[0].plot(x_axie, plot_list_an['S_NO'], label='S_NO')
ax1[0].plot(x_axie, plot_list_an['S_O'], label='S_O')
ax1[0].plot(x_axie, plot_list_an['S_S'], label='S_S')
ax1[0].plot(x_axie, plot_list_an['X_BA'], label='X_BA')
ax1[0].plot(x_axie, plot_list_an['X_BH'], label='X_BH')
ax1[0].plot(x_axie, plot_list_an['X_I'], label='X_I')
ax1[0].plot(x_axie, plot_list_an['X_ND'], label='X_ND')
ax1[0].plot(x_axie, plot_list_an['X_S'], label='X_S')
ax1[0].plot(x_axie, plot_list_an['X_P'], label='X_P')
ax1[0].set_title('ANOX')
ax1[0].set_ylabel('g/m³')
ax1[0].set_xlabel('time')
ax1[0].legend()
ax1[0].grid()
ax1[1].plot(x_axie, processo_an_df['H_grow'], label='H_grow')
ax1[1].plot(x_axie, processo_an_df['Han_grow'], label='Han_grow')
ax1[1].plot(x_axie, processo_an_df['A_grow'], label='A_grow')
ax1[1].plot(x_axie, processo_an_df['decay_H'], label='decay_H')
ax1[1].plot(x_axie, processo_an_df['Ammonia'], label='Ammonia')
ax1[1].plot(x_axie, processo_an_df['hydro_org'], label='hydro_org')
ax1[1].plot(x_axie, processo_an_df['hydro_N'], label='hydro_N')
ax1[1].set_title('PROCESSES')
ax1[1].set_ylabel('process')
ax1[1].set_xlabel('time')
ax1[1].legend()
ax1[1].grid()
'''

#FIGURA 1 POPULAÇAO BACTERIAS + CONSUMO DE SS
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
ax2.plot(x_axie, processo_an_df['Han_grow']*60, label='Crescimento_BHanox', color='green', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_an_df['decay_H']*60, label='Decaimento_BH', color='black', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_an_df['hydro_org']*60, label='Hidrólise-DQO', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='upper right', framealpha = 1)
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()
'''
fig, ax1 = plt.subplots()
ax1.plot(x_axie, plot_list_an['X_BH'], label='X_BH', color='green', linestyle='-', linewidth=3)
#ax1.plot(x_axie, plot_list_an['X_ND'], label='X_ND', color='purple', linestyle=':', linewidth=3)
#ax1.plot(x_axie, plot_list_an['S_ND'], label='S_ND', color='blue', linestyle=':', linewidth=3)
#ax1.plot(x_axie, plot_list_an['S_NH'], label='S_NH', color='red', linestyle=':', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_NO'], label='S_NO', color='red', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
ax1.legend(loc='upper right', framealpha = 1)
ax2 = ax1.twinx()
ax2.plot(x_axie, processo_an_df['Han_grow']*60, label='Crescimento_BHanox', color='green', linestyle=':', linewidth=3)
#ax2.plot(x_axie, processo_an_df['Ammonia'], label='Amonificação', color='red')
#ax2.plot(x_axie, processo_an_df['hydro_N'], label='Hidrólise-N', color='purple')
ax2.plot(x_axie, processo_an_df['decay_H']*60, label='Decaimento_BH', color='black', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()
'''
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x_axie, plot_list_an['X_ND'], label='X_ND', color='purple', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_ND'], label='S_ND', color='blue', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_NH'], label='S_NH', color='red', linestyle='-', linewidth=3)
ax1.plot(x_axie, plot_list_an['S_NO'], label='S_NO', color='black', linestyle='-', linewidth=3)
ax1.set_ylabel('Concentração (g[DQO]/m³)')
ax1.set_xlabel('Passo de Tempo')
#ax1.legend(loc='upper right', framealpha = 1)
#ax2 = ax1.twinx()
ax2.plot(x_axie, processo_an_df['Ammonia']*60, label='Amonificação', color='red', linestyle=':', linewidth=3)
ax2.plot(x_axie, processo_an_df['hydro_N']*60, label='Hidrólise-N', color='purple', linestyle=':', linewidth=3)
ax2.set_ylabel('Processo (g/(m³*min))')
ax1.legend(loc='upper right', framealpha = 1)
ax2.legend(loc='center right', framealpha = 1)
plt.tight_layout()
plt.show()
#plt.tight_layout()
#plt.show()
