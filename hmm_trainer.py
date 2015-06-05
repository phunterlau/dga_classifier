from collections	import defaultdict

# get a list of words with only ASCII characters
words = [w.strip().lower() for w in open("alexa_100k.txt").readlines()]
#words = [w for w in words if all([c in ascii_lowercase for c in w])]
words = ["^" + w.split('/')[0] + "$" for w in words if w != ""]

# construct a discrete-time markov chain of n-grams
n = 2 # this is the "n" in n-grams, try adjusting this for different results
transitions = defaultdict(lambda: defaultdict(float))
for word in words:
	if len(word) >= n:
		transitions[""][word[:n]] += 1.0
	for i in range(len(word) - n):
		gram = word[i : i + n]
		next = word[i + 1 : i + n + 1]
		transitions[gram][next] += 1.0
# normalize the probabilities
for gram in transitions:
	total = sum([transitions[gram][next] for next in transitions[gram]])
	for next in transitions[gram]:
		transitions[gram][next] /= total

fw = open('trans_matrix.csv','w')
for key1, dict1 in transitions.iteritems():
    for key2, value in dict1.iteritems():
        fw.write('%s\t%s\t%f\n'%(key1,key2,value))
        
fw.close()
