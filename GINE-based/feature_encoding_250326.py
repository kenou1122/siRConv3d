# DATA STRUCTURING (FEATURE ENCODING) METHODS

#-------------------------------------------------#

# SETTING ENVIRONMENT

import os
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
pd.set_option("display.max_columns", 50)

import Bio
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

import RNA

import khmer # python 3.9.16
from functools import reduce

#-------------------------------------------------#

# 0. DATA STRUCTURING

def seq_structuring(df_raw_data,df_raw_vocab):
    
    df_data = df_raw_data.copy(deep=True)
    df_vocab = df_raw_vocab.copy(deep=True)
    
    
    # remove null samples
    columns = ['siRNA_antisense_seq', 'modified_siRNA_antisense_seq_list']
    df_data.dropna(subset=columns, inplace=True)

    code = list(zip(df_vocab['Abbreviation'],df_vocab['single_abbr']))
    df_data['seq_sgchar_modi_sense'] = df_data['modified_siRNA_sense_seq']
    for ch in code:
        df_data['seq_sgchar_modi_sense'] = df_data['seq_sgchar_modi_sense'].str.replace(ch[0],ch[1],regex=False)
    df_data['seq_sgchar_modi_anti'] = df_data['modified_siRNA_antisense_seq']
    for ch in code:
        df_data['seq_sgchar_modi_anti'] = df_data['seq_sgchar_modi_anti'].str.replace(ch[0],ch[1],regex=False)
    df_data['seq_sgchar_modi_sense'] = df_data['seq_sgchar_modi_sense'].apply(lambda x: x[::-1])
    
    df_data['modi_sense_P+'] = df_data['seq_sgchar_modi_sense'].str.contains('P').astype('int')
    df_data['modi_anti_P+'] = df_data['seq_sgchar_modi_anti'].str.contains('P').astype('int')

    df_data['modi_sense_VP+'] = df_data['seq_sgchar_modi_sense'].str.contains('%').astype('int')
    df_data['modi_anti_VP+'] = df_data['seq_sgchar_modi_anti'].str.contains('%').astype('int')

    df_data['modi_sense_L96+'] = df_data['seq_sgchar_modi_sense'].str.contains('`').astype('int')
    df_data['modi_anti_L96+'] = df_data['seq_sgchar_modi_anti'].str.contains('`').astype('int')

    df_data['seq_sgchar_modi_sense'] = df_data['seq_sgchar_modi_sense'].str.replace('P','',regex=True)
    df_data['seq_sgchar_modi_sense'] = df_data['seq_sgchar_modi_sense'].str.replace('%','')
    df_data['seq_sgchar_modi_sense'] = df_data['seq_sgchar_modi_sense'].str.replace('`','')

    df_data['seq_sgchar_modi_anti'] = df_data['seq_sgchar_modi_anti'].str.replace('P','',regex=True)
    df_data['seq_sgchar_modi_anti'] = df_data['seq_sgchar_modi_anti'].str.replace('%','')
    df_data['seq_sgchar_modi_anti'] = df_data['seq_sgchar_modi_anti'].str.replace('`','')
    
    seq_code = list(zip(df_vocab['single_abbr'],df_vocab['deModi_sgabbr']))
    df_data['seq_agctus_sense'] = df_data['seq_sgchar_modi_sense']
    for ch in seq_code:
        df_data['seq_agctus_sense'] = df_data['seq_agctus_sense'].str.replace(ch[0],ch[1],regex=False)
    df_data['seq_agctus_anti'] = df_data['seq_sgchar_modi_anti']
    for ch in seq_code:
        df_data['seq_agctus_anti'] = df_data['seq_agctus_anti'].str.replace(ch[0],ch[1],regex=False)
    agct2int = str.maketrans('AUTGCS','012345')
    df_data['seq_agct_int_sense'] = df_data['seq_agctus_sense'].apply(lambda x: x.translate(agct2int))
    df_data['seq_agct_int_anti'] = df_data['seq_agctus_anti'].apply(lambda x: x.translate(agct2int))
    
    df_vocab['modil_label_int'] = df_vocab['modil_label_int'].fillna('_')
    modi_code = list(zip(df_vocab['single_abbr'],df_vocab['modil_label_int']))
    df_data['seq_modi_int_sense'] = df_data['seq_sgchar_modi_sense']
    for ch in modi_code:
        df_data['seq_modi_int_sense'] = df_data['seq_modi_int_sense'].str.replace(ch[0],ch[1],regex=False)
    df_data['seq_modi_int_anti'] = df_data['seq_sgchar_modi_anti']
    for ch in modi_code:
        df_data['seq_modi_int_anti'] = df_data['seq_modi_int_anti'].str.replace(ch[0],ch[1],regex=False)
    
    return df_data

#-------------------------------------------------#

# Ftr01. GC ratio

def RNAtoDNA(sequence):
    trantab = str.maketrans('Uu', 'TT')
    string = sequence.translate(trantab)
    return string

def GCcontent(sequence):
    sequence = sequence.upper()
    gcp = (sequence.count('G') + sequence.count('C')) / len(sequence)
    return gcp

def ft_encoding_gc(df_data):
    df_gc = df_data[[]].copy(deep=True)
    df_gc['!modiseq!_GC_ratio_antisense'] = df_data['seq_agctus_anti'].apply(lambda x: GCcontent(RNAtoDNA(x)))
    return df_gc

def ft_encoding_gc2(df_data):
    df_gc = df_data[[]].copy(deep=True)
    df_gc['!modiseq!_GC_ratio_antisense'] = df_data['seq_agctus_anti'].apply(lambda x: GCcontent(RNAtoDNA(x)))
    df_gc['!modiseq!_GC_ratio_antis_head'] = df_data['seq_agctus_anti'].str[:5].apply(lambda x: GCcontent(RNAtoDNA(x)))
    return df_gc

#-------------------------------------------------#

# Ftr02. Tfx time

def ft_encoding_hpt2(df_data,is_train):
    #df_data['!hpt_regu']
    df_hpt = df_data[[]].copy(deep=True)
    
    if is_train == True:
        mean = df_data['Duration_after_transfection_h'].mean()
        df_hpt['!tfx!_hpt_regu'] = df_data['Duration_after_transfection_h'].fillna(mean)
    else:
        df_hpt['!tfx!_hpt_regu'] = df_data['Duration_after_transfection_h']
    df_hpt['!tfx!_hpt_regu'] = df_hpt['!tfx!_hpt_regu']/72
    return df_hpt

#-------------------------------------------------#

# Ftr03. Concentration

def ft_encoding_conc3(df_data,is_train):
    #df_data['!conc_regu']
    df_conc_regu = df_data[[]].copy(deep=True)
    ###
    df_unit = df_data['concentration_unit'].str.lower()
    df_unit = df_unit.str.replace('μm','um')
    unit_map = {'nm':1,
                'um':1e3,
                'mm':1e6,
                'm':1e9,
                'pm':1e-3,
                'fm':1e-6}
    df_unit_regu = df_unit.map(unit_map).fillna(1)
    df_conc_same_unit = df_data['siRNA_concentration'] * df_unit_regu
    ###
    if is_train == True:
        m = df_conc_same_unit.median()
        df_conc_regu['!tfx!_conc_regu'] = df_conc_same_unit.fillna(m)
    else:
        df_conc_regu['!tfx!_conc_regu'] = df_conc_same_unit
    df_conc_regu['!tfx!_conc_regu'] = (np.log10(df_conc_regu['!tfx!_conc_regu']))/3
    return df_conc_regu

#-------------------------------------------------#

# Ftr04. Cell types

def ft_encoding_celltype_train(df_data):
    df_ct = df_data[['cell_line_donor']].copy(deep=True)
    df_ct.rename(columns={'cell_line_donor':'!tfx!_ct_cell_line_donor'},inplace=True)

    pcorcl_map = {'Hep3B Cells':'cell_line',
                  'Hepa1-6 Cells':'cell_line',
                  'HepG2 Cells':'cell_line',
                  'Primary Human Hepatocytes':'primary_cell',
                  'Primary Monkey Hepatocytes':'primary_cell',
                  'Primary Mouse Hepatocytes':'primary_cell',
                  'BE(2)-C Cells':'cell_line',
                  'Neuro2a Cells':'cell_line',
                  'Human Trabecular Meshwork Cells':'primary_cell',
                  'COS-7 Cells':'cell_line',
                  'A549 Cells':'cell_line',
                  'HeLa Cells':'cell_line',
                  'DU145 Cells':'cell_line',
                  'Panc-1 cells':'cell_line'}

    df_ct['!tfx!_ct_PCorCL'] = df_data['cell_line_donor'].map(pcorcl_map)
    df_ct['!tfx!_ct_PCorCL'] = df_ct['!tfx!_ct_PCorCL'].fillna('unknown_PCorCL')


    tissue_map = {'Hep3B Cells':'liver','Hepa1-6 Cells':'liver','HepG2 Cells':'liver',
                  'Primary Human Hepatocytes':'liver','Primary Monkey Hepatocytes':'liver','Primary Mouse Hepatocytes':'liver',
                  'BE(2)-C Cells':'Neural','Neuro2a Cells':'Neural',
                  'Human Trabecular Meshwork Cells':'eye',
                  'COS-7 Cells':'kidney',
                  'A549 Cells':'lung',
                  'HeLa Cells':'cervical',
                  'DU145 Cells':'prostate',
                  'Panc-1 cells':'pancreas'}

    df_ct['!tfx!_ct_tissue'] = df_data['cell_line_donor'].map(tissue_map)
    df_ct['!tfx!_ct_tissue'] = df_ct['!tfx!_ct_tissue'].fillna('unknown_tissue')
    
    return pd.get_dummies(df_ct).astype(int)

def ft_encoding_celltype_test(df_test,df_train_encoded):
    col_ref = df_train_encoded.columns
    df_test_encoded = ft_encoding_celltype_train(df_test)
    
    miss_features = set(df_train_encoded.columns)-set(df_test_encoded.columns)
    df_test_encoded[list(miss_features)] = 0

    df_test_encoded = df_test_encoded[df_train_encoded.columns]

    print('column order is same:',set(df_test_encoded.columns == df_train_encoded.columns))
    
    return df_test_encoded

#-------------------------------------------------#

# Ftr05. Tfx methods

def ft_encoding_tfx_method_2(df_data):
    df_tfx = df_data['Transfection_method'].copy(deep=True)
    df_tfx = '!tfx!_tfx_method_'+df_tfx
    dummies = pd.get_dummies(df_tfx).astype('int')
    if '!tfx!_tfx_method_Lipofectamine' not in dummies.columns:
        dummies['!tfx!_tfx_method_Lipofectamine']=0
    if '!tfx!_tfx_method_Free uptake' not in dummies.columns:
        dummies['!tfx!_tfx_method_Free uptake']=0
    return dummies[['!tfx!_tfx_method_Lipofectamine','!tfx!_tfx_method_Free uptake']]

#-------------------------------------------------#

# Ftr06. End-modi

def ft_encoding_endmodi2(df_data):
    df_endmodi = df_data[['modi_sense_L96+','modi_anti_L96+','modi_sense_P+','modi_anti_P+','modi_sense_VP+','modi_anti_VP+']]
    df_endmodi.columns = ['!modiseq!_modi_sense_L96+','!modiseq!_modi_anti_L96+',
                          '!modiseq!_modi_sense_P+','!modiseq!_modi_anti_P+',
                          '!modiseq!_modi_sense_VP+','!modiseq!_modi_anti_VP+']
    return df_endmodi

#-------------------------------------------------#

# Ftr07. Dist2ATG

def sirna2atg_dist(df):
    loc_atg_on_gene = df['gene_target_seq'].str.find('ATG')

    temp_series_sense_DNA_seq =  df['siRNA_sense_seq'].apply(lambda x: RNAtoDNA(x))
    temp_series_tarseq_senseq_concat = df['gene_target_seq'].fillna('_').apply(lambda x: [x]) + temp_series_sense_DNA_seq.apply(lambda x: [x])

    loc_sense_on_gene = temp_series_tarseq_senseq_concat.apply(lambda x: x[0].find(x[1][3:-3]))
    loc_sense_on_gene[loc_sense_on_gene<0]=None
    
    loc_sirna_to_gene_atg = loc_sense_on_gene-loc_atg_on_gene
    mean_dist_to_atg = loc_sirna_to_gene_atg.mean()
    
    return loc_sirna_to_gene_atg,mean_dist_to_atg

def dist_regular(loc_sirna_to_gene_atg,mean):
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def single_dist_regular(x,mean):
        return (x-mean)*3/mean
    series_regular_dist = loc_sirna_to_gene_atg.apply(lambda x: sigmoid(single_dist_regular(x,mean)))*2-1
    return series_regular_dist

def ft_encoding_dist2atg(df,DIST_MEAN):
    df_data = df[[]].copy(deep=True)
    loc_dist2atg = sirna2atg_dist(df)[0]
    loc_dist2atg = loc_dist2atg.fillna(DIST_MEAN)
    loc_dist2atg_regu = dist_regular(loc_dist2atg,DIST_MEAN)
    df_data['!mrna!_dist2atg_regu'] = loc_dist2atg_regu
    return df_data

#-------------------------------------------------#

# Ftr08. siRNA length

def ft_encoding_len(df_data):
    df_len = df_data[[]].copy(deep=True)
    df_len['!modiseq!_len_sense_regu'] = (df_data['siRNA_sense_seq'].str.len()-18)/10
    df_len['!modiseq!_len_anti_regu'] = (df_data['siRNA_antisense_seq'].str.len()-18)/10
    return df_len

#-------------------------------------------------#

# Ftr09. Second structure

def get_struct_ppdp(sequence): # probability of pairing matrix
    fc = RNA.fold_compound(sequence)
    (dp_mfe, val_mfe) = fc.mfe() #Minimum free energy
    (dp_pp, val_pf) = fc.pf()
    (dp_centroid, val_ctrdist) = fc.centroid()
    (dp_MEA, val_MEA) = fc.MEA()
    mtx_pp = np.array(fc.bpp())[1:,1:] 
    #adj_bpp = np.pad(adj_bpp,(0,SEQ_MAX_LEN-adj_bpp.shape[1]),'constant',constant_values=(0,0))
    return mtx_pp,dp_mfe,dp_pp,dp_centroid,dp_MEA,val_mfe,val_pf,val_ctrdist,val_MEA

def show_structure_map():
    structure_map = {
        's': 'stem',
        'h': 'hairpin loop',
        'i': 'interior loop',
        'm': 'multiloop',
        'f': 'fiveprime',
        't': 'threeprime',
        'P': 'no nt here (pad)'
    }
    return structure_map

def get_nt_strtype_mtx(dot_bracket,SEQ_MAX_LEN=28):
    import forgi.graph.bulge_graph as fgb
    bg = fgb.BulgeGraph.from_dotbracket(dot_bracket)
    elements_strcode = bg.to_element_string()
    elements_strcode += 'P'*(SEQ_MAX_LEN-len(elements_strcode))
    map_dict = {'P':0,'s':6,'h':1,'i':2,'m':3,'f':4,'t':5}
    elements_numcode = list(map(lambda x:map_dict[x],list(elements_strcode)))
    onehot_mtx = np.eye(SEQ_MAX_LEN,7)[elements_numcode]
    return torch.tensor(onehot_mtx).unsqueeze(0)

#-------------------------------------------------#

# Ftr10. tarSeq kmer

def ft_encoding_genekmer(df,is_train,df_kmer_freq_mean_train=None):
    def kmer_freq(seq,kmer_vocab):
        #kmer_counts = []
        counts = khmer.Counttable(3, 64, 1)
        counts.consume(seq)
        kmer_counts = list(map(lambda x: counts.get(x),kmer_vocab))
        kmer_counts = np.array(kmer_counts)/sum(kmer_counts)
        return kmer_counts
    df_data = df[['gene_target_seq']].copy(deep=True)
    df_data['gene_target_seq'] = df_data['gene_target_seq'].fillna('N'*6)
    gene_repseq_unique = pd.Series(df_data['gene_target_seq'].unique())
    
    K = 3
    kmer_vocab = reduce(lambda x,y: [i+j for i in x for j in y], [['A','T','C','G']] * K)
    
    df_kmer_freq = pd.DataFrame(list(map(lambda x:kmer_freq(x,kmer_vocab),gene_repseq_unique)))
    df_kmer_freq.columns = df_kmer_freq.columns.map(lambda x: '!mrna!_kmer_'+str(x)+'_freq')
    df_kmer_freq.index = gene_repseq_unique
    df_kmer_freq_mean = df_kmer_freq.mean()
    if is_train == False:
        df_kmer_freq.loc['NNNNNN'] = df_kmer_freq_mean_train
    else:
        df_kmer_freq.loc['NNNNNN'] = df_kmer_freq_mean
    df_kmer_freq.reset_index(inplace=True)
    return df_kmer_freq,df_kmer_freq_mean

#-------------------------------------------------#

# Ftr11. Species

species_columns = ['!mrna!_species_Homo sapiens','!mrna!_species_Macaca fascicularis','!mrna!_species_Mus musculus']

def ft_encoding_species(df):
    df_data = df['gene_target_species'].copy(deep=True)
    df_data = '!mrna!_species_'+df_data
    df_data = pd.get_dummies(df_data)
    
    miss_features = set(species_columns)-set(df_data.columns)
    df_data[list(miss_features)] = 0
    
    df_data = df_data[species_columns]
    return df_data.astype('int')

def species_fillna_based_on_celltype(df_data):
    cell_sp_sex = {('A549 Cells','Homo sapiens','M'),
         ('BE(2)-C Cells', 'Homo sapiens', 'M'),
         ('COS-7 Cells','Cercopithecus aethiops','M'),
         ('DU145 Cells', 'Homo sapiens','M'),
         ('HeLa Cells', 'Homo sapiens','F'),
         ('Hep3B Cells', 'Homo sapiens','M'),
         ('HepG2 Cells', 'Homo sapiens','M'),
         ('Hepa1-6 Cells', 'Mus musculus', 'F'),
         ('Human Trabecular Meshwork Cells','Homo sapiens',''),
         ('Neuro2a Cells', 'Mus musculus','M'),
         ('Panc-1 cells', 'Homo sapiens','M'),
         ('Primary Human Hepatocytes', 'Homo sapiens',''),
         ('Primary Monkey Hepatocytes', 'Macaca fascicularis',''),
         ('Primary Mouse Hepatocytes', 'Mus musculus','')}
    df_cell_sp_sex = pd.DataFrame(cell_sp_sex)
    df_cell_sp_sex.columns = ('cell_types','correct_species','gender')
    df_cell_sp_sex_merge = pd.merge(df_data[['cell_line_donor','gene_target_species']],
                                    df_cell_sp_sex,left_on='cell_line_donor',
                                    right_on='cell_types',how='left',sort=False)
    
    cell_line_donor_nan_index = df_cell_sp_sex_merge.index[df_cell_sp_sex_merge['cell_line_donor'].isna()]
    df_cell_sp_sex_merge.loc[cell_line_donor_nan_index,'correct_species'] = df_cell_sp_sex_merge.loc[cell_line_donor_nan_index,'gene_target_species']
    
    return df_cell_sp_sex_merge

#-------------------------------------------------#

# Ftr12. TM

def ft_encoding_TM(df):
    df_data = df[[]].copy(deep=True)
    df_data['!modiseq!_TM_sense'] = df['siRNA_sense_seq'].apply(lambda x: Bio.SeqUtils.MeltingTemp.Tm_NN(Seq(x),nn_table=mt.RNA_NN3))
    df_data['!modiseq!_TM_antis'] = df['siRNA_antisense_seq'].apply(lambda x: Bio.SeqUtils.MeltingTemp.Tm_NN(Seq(x),nn_table=mt.RNA_NN3))
    df_data = df_data/100
    return df_data

#-------------------------------------------------#
'''
# Ftr13. Prior Knowledge
dumped
'''
#-------------------------------------------------#

# Ftr14. ModiSeq OneHot 3D

def onehot_3d(seq_a='',seq_b='',shape_3d=[3,3,3]):
    space = np.zeros(shape_3d)
    seqls_a = list(map(int,list(seq_a)))
    seqls_b = list(map(int,list(seq_b)))
    seq_len = list(range(len(seq_a)))
    space[seq_len,seqls_a,seqls_b] = 1
    return space

def df2onehot3d_v2(df,space_shape=(28,6,7)):
    output = []
    skip_sample_index = []

    for i in range(df.shape[0]):


        seq_agct_int_sense = df.iloc[i]['seq_agct_int_sense']
        seq_modi_int_sense = df.iloc[i]['seq_modi_int_sense']

        seq_agct_int_anti = df.iloc[i]['seq_agct_int_anti']
        seq_modi_int_anti = df.iloc[i]['seq_modi_int_anti']

        if (len(seq_agct_int_sense) != len(seq_modi_int_sense)) or (len(seq_agct_int_anti) != len(seq_modi_int_anti)):
            print('!warning!\t',i,' sample agct-seq and modi-seq are not equal len.',sep='')
            skip_sample_index.append(i)
            continue

        if i%1000==999: print(i+1,end='\r')
        
        sense_onehot_3d = onehot_3d(seq_agct_int_sense,seq_modi_int_sense,space_shape)
        anti_onehot_3d = onehot_3d(seq_agct_int_anti,seq_modi_int_anti,space_shape)
        output.append([sense_onehot_3d,anti_onehot_3d])

    output = np.array(output)
    
    return output,skip_sample_index

#-------------------------------------------------#

#-------------------------------------------------#

# TFX FTR ENCODING

## for TRAINING set

def feature_encoding_train(df_train_data):
    df_train_encoded = df_train_data[[]].copy(deep=True)
    df_train_encoded = df_train_encoded.join(ft_encoding_gc(df_train_data)) ## 0904
    df_train_encoded = df_train_encoded.join(ft_encoding_hpt2(df_train_data,is_train=True))
    df_train_encoded = df_train_encoded.join(ft_encoding_conc3(df_train_data,is_train=True)) ## 0904

    df_train_celltype_encoded = ft_encoding_celltype_train(df_train_data)
    df_train_encoded = df_train_encoded.join(df_train_celltype_encoded)
    df_train_encoded = df_train_encoded.join(ft_encoding_tfx_method_2(df_train_data))
    
    df_train_encoded = df_train_encoded.join(ft_encoding_endmodi2(df_train_data))
    loc_train_dist2atg, DIST_MEAN = sirna2atg_dist(df_train_data)
    df_train_encoded = df_train_encoded.join(ft_encoding_dist2atg(df_train_data,DIST_MEAN))
    df_train_encoded = df_train_encoded.join(ft_encoding_len(df_train_data))
    df_train_encoded = df_train_encoded.join(ft_encoding_species(df_train_data))
    df_train_encoded = df_train_encoded.join(ft_encoding_TM(df_train_data))
    
    df_train_kmer_temp = df_train_data[['gene_target_seq','ID_in_model']].copy(deep=True)
    df_train_kmer_temp['gene_target_seq'] = df_train_kmer_temp['gene_target_seq'].fillna('N'*6)
    (kmer_freq_train,MEAN_kmer_freq_train) = ft_encoding_genekmer(df_train_kmer_temp,is_train=True)
    df_train_kmer = pd.merge(df_train_kmer_temp,kmer_freq_train,left_on='gene_target_seq',right_on='index',sort=False)
    df_train_kmer.set_index('ID_in_model',inplace=True)
    df_train_kmer.drop(columns=['gene_target_seq','index'],inplace=True)

    df_train_encoded = df_train_encoded.join(df_train_kmer)
    
    ftr_encoding_params = (DIST_MEAN,MEAN_kmer_freq_train)
    
    return df_train_encoded,ftr_encoding_params

## for TEST set

def feature_encoding_test(df_test_data,df_train_encoded,ftr_encoding_params):
    df_test_encoded = df_test_data[[]].copy(deep=True)
    
    DIST_MEAN = ftr_encoding_params[0]
    MEAN_kmer_freq_train = ftr_encoding_params[1]

    df_test_encoded = df_test_encoded.join(ft_encoding_gc(df_test_data))  ## 0904
    df_test_encoded = df_test_encoded.join(ft_encoding_hpt2(df_test_data,is_train=False))
    df_test_encoded = df_test_encoded.join(ft_encoding_conc3(df_test_data,is_train=False))  ## 0904
    col_train_ct_encoded = df_train_encoded.columns[df_train_encoded.columns.str.contains('!tfx!_ct_')]
    df_test_encoded = df_test_encoded.join(ft_encoding_celltype_test(df_test_data,df_train_encoded[col_train_ct_encoded]))
    
    df_test_encoded = df_test_encoded.join(ft_encoding_tfx_method_2(df_test_data))
    
    df_test_encoded = df_test_encoded.join(ft_encoding_endmodi2(df_test_data))
    df_test_encoded = df_test_encoded.join(ft_encoding_dist2atg(df_test_data,DIST_MEAN))
    df_test_encoded = df_test_encoded.join(ft_encoding_len(df_test_data))
    df_test_encoded = df_test_encoded.join(ft_encoding_species(df_test_data))
    df_test_encoded = df_test_encoded.join(ft_encoding_TM(df_test_data))
    
    df_test_kmer_temp = df_test_data[['gene_target_seq','ID_in_model']].copy(deep=True)
    df_test_kmer_temp['gene_target_seq'] = df_test_kmer_temp['gene_target_seq'].fillna('N'*6)
    kmer_freq_test = ft_encoding_genekmer(df_test_kmer_temp,is_train=False,df_kmer_freq_mean_train=MEAN_kmer_freq_train)[0]
    df_test_kmer = pd.merge(df_test_kmer_temp,kmer_freq_test,left_on='gene_target_seq',right_on='index',sort=False)
    df_test_kmer.set_index('ID_in_model',inplace=True)
    df_test_kmer.drop(columns=['gene_target_seq','index'],inplace=True)
    
    df_test_encoded = df_test_encoded.join(df_test_kmer)
    
    df_test_encoded['!tfx!_hpt_regu'] = df_test_encoded['!tfx!_hpt_regu'].fillna(df_train_encoded['!tfx!_hpt_regu'].mean())
    df_test_encoded['!tfx!_conc_regu'] = df_test_encoded['!tfx!_conc_regu'].fillna(df_train_encoded['!tfx!_conc_regu'].mean())
    
    return df_test_encoded

#-------------------------------------------------#