SHELL=bash

# 100:124  for a in `cat countries | tail -n +100 | head -n 25` ;

# get ICaRR  Implementation Completion and Result Report  (-t 3)
get1 ICaRR:
	# get ICaRR  Implementation Completion and Result Report  (-t 3)
	for a in `cat countries` ;\
	   do  \
	       get-WB-reports -d -t 3 -c $$a ;\
	   done 

# get ICRR  Implementation Completion Report Review (-t 2)
get3 ICRR:
	# get ICRR  Implementation Completion Report Review (-t 2)
	for a in `cat countries ` ; \
	   do  \
   	      get-WB-reports -d -t 2 -c $$a ;\
	   done 

# get PAD Project Appraisal Document (-t 1)
get2 PAD:
	# get PAD Project Appraisal Document (-t 1)
	for a in `cat countries ` ;\
	   do  \
   	      get-WB-reports -d -t 1 -c $$a ;\
	   done 

#Claculate TXT of ICaRR (GW)
pdf1:
	for a in GW/P[0-9]*pdf ;\
           do \
	      pdftotext -layout $$a $$a.txt ;\
	   done




## DEBUG experiments

U=https://search.worldbank.org/api/v2/wds?
#	rows=100&os=10
#C=Mozambique
C=Argentina
P=format=json&count_exact=$C&docty_exact=Implementation Status and Results Report&fl=count,volnb,totvolnb,docna,docty,repnme

p1:
#	wget 'https://search.worldbank.org/api/v2/wds?rows=100&os=10&format=json&count_exact=Brazil&docty_exact=Implementation Status and Results Report&fl=count,volnb,totvolnb,docna,docty,repnme'
	wget '$Urows=100&os=0&$P'
	wget '$Urows=100&os=100&$P'
	wget '$Urows=100&os=200&$P'
	wget '$Urows=100&os=300&$P'
	wget '$Urows=100&os=400&$P'
	wget '$Urows=100&os=500&$P'
	wget '$Urows=100&os=600&$P'

p2:
	jq .  'wds?rows=100'* | grep '"id"'| sort -u | wc
