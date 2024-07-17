Score=0

print("1-Sport")

print("2-geography")

print("3-History")

print("4-science")

print("5-math")
#sport
Q1="which player win the ballondor 2022? "
A1="messi"
Q2=":which national football team win the euro 2016? "
A2="portugal"
Q3="which national football team win the world cup 2014? "
A3="germany"
Q4="How many qualifed egypt to the world cup? "
A4="3"
Q5="which national football team win the world cup 2002? "
A5="brazil"
#geography
Q6="which country has the biggest area in the world ? "
A6="russia"
Q7="which country is in west egypt? "
A7="libya"
Q8="which country is the biggest in africa ? "
A8="algeria"
Q9="which country is from two contents? "
A9="russia"
Q10="which country has the biggest area in south america? "
A10="brazil"
#History
Q11="How many years did the Hundred Years War between France and England last? "
A11="116"
Q12="How many countries are there in the Arabian Peninsula? "
A12="7"
Q13="From the ancient state ruled by the Ptolemy dynasty? "
A13="Egypt"
Q14="In what year did World War I take place? "
A14="1914 "
Q15="What is the birthplace of Queen Cleopatra? "
A15="Greece"
#science
Q16="What is the thickest layer of the Earth? "
A16="plup"
Q17="What is the reason for the planet Mars becoming red? "
A17="it contains abundant iron oxide"
Q18="What part of plants is responsible for transporting food? "
A18="bark"
Q19="How many eyelids does a camel have? "
A19="3"
Q20="How many muscles are in the human body? "
A20="620"
#math
Q21="What is the area of ​​the circle? "
A21="pi*r*r"
Q22="What is the circumference of a circle? "
A22="2*pi*r"
Q23="is 0/0 undefined or indeterminate? "
A23="indeterminate"
Q24="negative * negative =? "
A24="positive"
Q25="72 × 5=? "
A25=360
####### MATH LVL 2
lvl2="yes"
Q1_lvl2="sin(30) "
a1_lvl2="1/2"
Q2_lvl2="plus 1for100= "
a2_lvl2="5050"
Q3_lvl2="i*i = "
a3_lvl2="-1"

subject=input("Enter you're subject ")
if subject=="1":
    answer=input(Q1)
    if answer==A1 :
        print("right")
        Score+=1        
    else :
        print("wrong,try again")
        answer=input(Q1)
        if answer==A1:
            print("right")
            Score+=1
            

    answer=input(Q2)
    if answer==A2:
        print("right")
        Score+=1
    else:   
        print("wrong,try again")
        answer=input(Q2)
        if answer==A2:
            print("right")
            Score+=1

    answer=input(Q3)
    if answer==A3:
        print("right")
        Score+=1
    else:
        print("wrong,try again")
        answer=input(Q3)
        if answer==A3:
            print("right")
            Score+=1     
	
    answer=input(Q4)
    if answer==A4:
        print("right")
        Score+=1
    else:
        print("wrong,try again")
        answer=input(Q4)
        if answer==A4:
            print("right")
            Score+=1
	    
    answer=input(Q5)
    if answer==A5:
        print("right")
        Score+=1
        print("your score is " +  str(Score))
        exit()
    else:
        print("wrong,try again")
        answer=input(Q5)
        if answer==A5:
            print("right")
            Score+=1
            print("your score is " +  str(Score))


if subject=="2":
    answer=input(Q6)
    if answer==A6:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q6)
        if answer==A6:
            print("right")
            Score+=1
			
    answer=input(Q7)
    if answer==A7:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q7)
        if answer==A7:
            print("right")
            Score+=1
	   
    answer=input(Q8)
    if answer==A8:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q8)
        if answer==A8:
            print("right")
            Score+=1
	
    aAnswer=input(Q9)
    if answer==A9:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q9)
        if answer==A9:
            print("right")
            Score+=1
	 
    answer=input(Q10)
    if answer==A10:
        Score+=1
        print("right")
        print("your score is "+str(Score))
        exit()
    else:
        print("wrong,try again")
        answer=input(Q10)
        if answer==A10:
            print("right")
            Score+=1
            print("your score is "+str(Score))
            exit()

if subject=="3":
    Answer=input(Q11)
    
    if Answer==A11:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q11)
        if answer==A11:
            print("right")
            Score+=1	
    
    Answer=input(Q12)
    if Answer==A12:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q12)
        if answer==A12:
            print("right")
            Score+=1
    
    Answer=input(Q13)
    if Answer==A13:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q13)
        if answer==A13:
            print("right")
            Score+=1
    
    Answer=input(Q14)
    if Answer==A14:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q14)
        if answer==A14:
            print("right")
            Score+=1
        
    Answer=input(Q15)
    if Answer==A15:
        Score+=1
        print("right")
        print("your score is "+str(Score))
        exit()
        
    else:
        print("wrong,try again")
        answer=input(Q15)
        if answer==A15:
            print("right")
            Score+=1
            print("your score is "+str(Score))
            exit()  


if subject=="4":
    
    Answer=input(Q16)
    if Answer==A16:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q16)
        if answer==A16:
            print("right")
            Score+=1	
    
    Answer=input(Q17)
    if Answer==A17:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q17)
        if answer==A17:
            print("right")
            Score+=1
    
    Answer=input(Q18)
    if Answer==A18:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q18)
        if answer==A18:
            print("right")
            Score+=1
    
    Answer=input(Q19)
    if Answer==A19:
        Score+=1
        print("right")
    else:
        print("wrong,try again")
        answer=input(Q19)
        if answer==A19:
            print("right")
            Score+=1
    
    Answer=input(Q20)
    if Answer==A20:
        Score+=1
        print("right")
        print("your score is "+str(Score))
        exit()
    else:
        print("wrong,try again")
        answer=input(Q20)
        if answer==A20:
            print("right")
            Score+=1
            print("your score is "+str(Score))
            exit()


if subject=="5":

    for z in range(1,4):
        answer=input(Q21)
        if answer==A21:
            Score+=1    
            print("right")
            break
        else:
            print("try again")
    for x in range(1,4):
        answer=input(Q22 )
        if answer==A22:
            Score+=1
            print("right")
            break
        else:
            print("try again")

    for c in range(1,4):
        answer=input(Q23)
        if answer==A23:
            Score+=1   
            print("right")
            break 
        else:
            print("try again")
    for v in range(1,4):
        answer=input(Q24)
        if answer==A24:
            Score+=1 
            print("right")
            break
        else:
            print("try again")
    for b in range(1,4):
        answer=int(input(Q25))
        if answer==A25:
            Score+=1    
            print("right")
            break
        else:
            print("try again")
    print("your score is "+str(Score))
    lvl_2=input("wanna lvl 2 ? ")#lvl 2
    if lvl2==lvl_2:
        for n in range(1,4):
            answer=input(Q1_lvl2)
            if answer==a1_lvl2:
                print("right")
                Score+=1
                break
            else:
                print("try agian")
        for m in range(1,4):
            answer=input(Q2_lvl2)
            if answer==a2_lvl2:
                print("right")
                Score+=1
                break
            else:
                print("try again")
        for a in range(1,4):
            answer=input(Q3_lvl2)
            if answer==a3_lvl2:
                print("Congratulations, you have passed the entire test")
                Score+=1
                print(Score)
                break
            else:
                print("try again")