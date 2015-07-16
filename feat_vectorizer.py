'''
this script translate features to training table
'''
from collections import Counter
from operator import itemgetter

fi = open('features_norm.txt','r')
header = fi.readline().strip().split(',')
header_dict = dict((j,i) for i,j in enumerate(header))
feat_table = list()
for f in fi:
    #domain,cla,tld,entropy,f_len,norm_entropy,bigram,trigram=f.rstrip('\n').split('\t')
    ll = f.rstrip('\n').split(',')
    #dummy,domain,cla,tld,entropy,f_len,norm_entropy,vowel_ratio,uni_rank,bi_rank,tri_rank,uni_std,bi_std,tri_std=f.rstrip('\n').split(',')
    feat_dict = dict()
    feature_header = header
    for f in feature_header:
        if f in ['','ip','class','private_tld']: continue
        feat = ll[header_dict[f]]
        if not f=='tld': feat=float(feat)
        else: feat=feat.lower()

        feat_dict[f]=feat
    domain,cla = itemgetter(1,2)(ll)
    '''
    feat_dict['tld']=tld.lower()
    feat_dict['entropy']=float(entropy)
    feat_dict['len']=float(f_len)
    feat_dict['norm_entropy']=float(norm_entropy)
    feat_dict['uni_rank']=float(uni_rank)
    feat_dict['bi_rank']=float(bi_rank)
    feat_dict['tri_rank']=float(tri_rank)
    feat_dict['uni_std']=float(uni_std)
    feat_dict['bi_std']=float(bi_std)
    feat_dict['tri_std']=float(tri_std)
    feat_dict['vowel_ratio']=float(vowel_ratio)
    '''
    '''
    bigram=Counter(bigram.split(',')).most_common()
    trigram=Counter(trigram.split(',')).most_common()
    for b,freq in bigram:
        if not b =='':
            feat_dict['bigram_'+b]=float(freq)
    for t,freq in trigram:
        if not t =='':
            feat_dict['trigram_'+b]=float(freq)
    '''
    feat_table.append([domain,cla,feat_dict])

fi.close()

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

measurements = [feat_dict for domain,cla,feat_dict in feat_table]
feature_list = vec.fit_transform(measurements).toarray()
feature_header = vec.get_feature_names()
#print feature_header

fw_out = open('vectorized_feature_w_ranks_norm.txt','w')

fw_out.write('ip,%s,class\n'%(','.join(feature_header)))

ground_truth = [(domain,cla) for domain,cla,feat_dict in feat_table]
for (domain,cla),feats in zip(ground_truth,feature_list):
    joined_feats = ','.join('%.2f'%i for i in feats)
    fw_out.write('%s,%s,%s\n'%(domain,joined_feats,cla))

fw_out.close()
