def search_solution (goal_state, source_state):
    solution=[]
    while(goal_state>source_state):
        if goal_state%2>0:
            solution.insert(0,'Right')
        else:
            solution.insert(0,'left')
        goal_state=goal_state//2
    if goal_state==source_state:
        return solution
    else:
        return None
	
print(search_solution(13,2))
