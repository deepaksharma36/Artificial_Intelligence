
r=input()
c=input()
data={}
alpha=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

p={}
pep=input()
pep=pep.split(sep=",")

taken={}

def putlazy(pep):
                for i in range(1,int(c)+1):
                    for a in alpha:
                        if a in p.keys():

                            if i not in p[a] and a+str(i) not in taken.keys():
                               #print(i)
                               taken[a+str(i)]=[pep]
                               print(a+str(i),end=",")
                               return
                        else:
                            taken[a+str(1)]=[pep]


for item in pep:
    #print(item)
    if item[-1]=="*":
       print('Got laxx')
       #print(p)
       putlazy(item)
    else:
            if item in taken.keys():
                p[item[0]].append(int(item[1:]))
                putlazy(taken[item])
            elif item[0] not in p:
                p[item[0]]=[int(item[1:])]
            else:
                p[item[0]].append(int(item[1:]))
            #print(p)