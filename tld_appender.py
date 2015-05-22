'''
this script produces the max matching for each domain to it's longest mozilla TLD
since the TLD list is very short, the algorithm is naive and brut force
'''
def max_match(domain, tlds):#simple match and find the longest match
   match = [i for i in tlds if i in domain] 
   if len(match)>0:
       for i in sorted(match,key=lambda x: len(x), reverse = True):
           if i == domain[-(len(i)):]:#longest and matches the end of the domain
               return i
   else: return 'NONE'

tld_file = open('tld_list.txt','r')
tld_list = list('.'+t.strip().strip('.')+'.' for t in tld_file)#for domain match, add dot as prefix and postfix
tld_file.close()

#fi = open('1393459321.nps_malware_dga_training_set.txt','r')
#fi = open('expanded_training_set.txt','r')
fi = open('conficker_alexa_training.txt','r')

fw = open('training_w_tld.txt','w')

for f in fi:
    domain, cla = f.strip().split('\t')
    match = max_match(domain, tld_list)
    fw.write('%s\t%s\t%s\n'%(domain,cla,match))

fw.close()
fi.close()
