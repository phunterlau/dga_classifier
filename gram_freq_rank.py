'''
this script generate the reference rank list of unigram, bigram, trigram from alexa top 100k
'''

from collections import defaultdict
import tldextract #TLD

def bigrams(words):
    wprev = None
    for w in words:
        if not wprev==None:
            yield (wprev, w)
        wprev = w

def trigrams(words):
    wprev1 = None
    wprev2 = None
    for w in words:
        if not (wprev1==None or wprev2==None):
            yield (wprev1,wprev2, w)
        wprev1 = wprev2
        wprev2 = w

private_tld_file = open('private_tld.txt','r')
private_tld = set(f.strip() for f in private_tld_file)#black list for private tld
private_tld_file.close()

unigram_rank = defaultdict(int)
bigram_rank = defaultdict(int)
trigram_rank = defaultdict(int)

fi = open('top-100k.csv','r')
for f in fi:
    rank,domain = f.strip().split(',')
    ext = tldextract.extract(domain)
    tld = ext.suffix
    main_domain = '$'+ext.domain+'$'#add begin and end
    if tld in private_tld:
        tld_list = tld.split('.')
        tld = tld_list[-1]
        main_domain = '$'+tld_list[-2]+'$'
    for i in main_domain[1:-1]:
        unigram_rank[i]+=1
    for i in bigrams(main_domain):
        bigram_rank[''.join(i)]+=1
    for i in trigrams(main_domain):
        trigram_rank[''.join(i)]+=1

fi.close()

fw = open('n_gram_rank_freq.txt','w')
for rank,(i,freq)in enumerate(sorted(unigram_rank.iteritems(), key = lambda x:x[1], reverse = True)):
    try:
        fw.write('1,%s,%d,%d\n'%(i,freq,rank+1))
    except UnicodeEncodeError:
        continue
for rank,(i,freq) in enumerate(sorted(bigram_rank.iteritems(),key = lambda x:x[1], reverse = True)):
    try:
        fw.write('2,%s,%d,%d\n'%(i,freq,rank+1))
    except UnicodeEncodeError:
        continue
for rank,(i,freq) in enumerate(sorted(trigram_rank.iteritems(),key = lambda x:x[1], reverse = True)):
    try:
        fw.write('3,%s,%d,%d\n'%(i,freq,rank+1))
    except UnicodeEncodeError:
        continue

fw.close()
