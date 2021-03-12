import numpy as np

'''==================================================
Initial set up
=================================================='''
#Task defining
task = "Policy Iteration"

#Parameters
theta = 0.0005
gamma = 0.9

#States
state=[]
for i in range(4):
    for j in range(4):
        state.append((i,j))

		
#Rewards
rewards = {}
for i in state:
    if i == (1,3):
        rewards[i] = -1
    elif i == (3,2):
        rewards[i] = -1
    elif i == (3,3):
        rewards[i] = 1
    else:
        rewards[i] = 0

#Actions in each Sate
actions = {
    (0,0):('D', 'R'),
    (0,1):('D', 'R', 'L'),
    (0,2):('D', 'L', 'R'),
    (0,3):('D', 'L'),
    (1,0):('D', 'U', 'R'),
    (1,1):('D', 'R', 'L', 'U'),
    (1,2):('D', 'L', 'U', 'R'),
	#(1,3):('D', 'L', 'U'),
    (2,0):('U', 'R', 'D'),
    (2,1):('D', 'L', 'U', 'R'),
    (2,2):('D', 'L', 'U', 'R'),
    (2,3):('D', 'L', 'U'),
    (3,0):('U', 'R'),
    (3,1):('U', 'L', 'R'),
	#(3,2):('U', 'L', 'R'),
	#(3,3):('U', 'L'),
}

#Initial Policy
policy={}
for s in state:
	if s in actions.keys():
		policy[s] = np.random.choice(actions[s])
	else:
		policy[s] = "-"
policy[(3,3)] = "G"	


#Initial Value Function 
V={}
for s in state:
    if s in actions.keys():
        V[s] = 0
    if s ==(3,2):
        V[s]=-1
    if s == (1,3):
        V[s]=-1
    if s == (3,3):
        V[s]=1

#Printing Intial Value Function
print		
V_old = V.copy()
result_old = list(V_old.values())
for x in range(len(result_old)):
	if  x!= 0 and x%4 == 0:
		print()
	print(f"{result_old[x]}", end=" ")
print()
print()


'''==================================================
Value Iteration
=================================================='''
def ValueIteration(policy, V, state, actions, rewards, theta, gamma):
	iteration = 0
	while 1:
		delta = 0
		for s in state:
			if s in policy:
				if s != (1, 3) and s != (3, 2) and s != (3, 3):

					old_v = V[s]
					new_v = 0

					for a in actions[s]:
						if a == 'U':
							nxt = [s[0]-1, s[1]]
						if a == 'D':
							nxt = [s[0]+1, s[1]]
						if a == 'L':
							nxt = [s[0], s[1]-1]
						if a == 'R':
							nxt = [s[0], s[1]+1]


						nxt = tuple(nxt)
						v = rewards[s] + (gamma * V[nxt])
						if v > new_v:
							new_v = v
							policy[s] = a
                              
					V[s] = new_v
					delta = max(delta, np.abs(old_v - V[s]))
    
		if delta < theta:
			break
		iteration += 1
		
	print("After", iteration, "iterations:")
	result = list(V.values())
	for x in range(len(result)):
		if  x!= 0 and x%4 == 0:
			print()
		print(f"{result[x]}", end=" ")
		
	print()
	print()
	print("Final Policy:")
	result_policy = list(policy.values())
	for x in range(len(result_policy)):
		if  x!= 0 and x%4 == 0:
			print()
		print(result_policy[x], end=" ")
			

'''==================================================
Policy Iteration
=================================================='''
# Policy Evaluation
def PolicyEvaluation(policy, V, state, actions, rewards, theta, gamma):
	V_old = V.copy()
	iteration = 0
	while 1:
		delta = 0
		for s in state:
			v = 0
			cnt = s[0] * 4 + s[1]
				
			if cnt != 15 and cnt != 14 and cnt != 7:
				old_v = V[s]
				new_v = 0

				for a in actions[s]:
					if a == 'U':
						a_n = 0
						nxt = [s[0]-1, s[1]]
					if a == 'D':
						nxt = [s[0]+1, s[1]]
						a_n = 1
					if a == 'L':
						nxt = [s[0], s[1]-1]
						a_n = 2
					if a == 'R':
						nxt = [s[0], s[1]+1]
						a_n = 3

					nxt = tuple(nxt)
					v += policy[cnt][a_n] * (rewards[s] + (gamma * V[nxt]))
					new_v = v
                              
				V[s] = new_v
				delta = max(delta, np.abs(old_v - V[s]))
      
		if delta < theta:
			break
		iteration += 1
	
	return V
		
# Policy Improvement
def PolicyImprovement(V, state, actions, rewards, theta, gamma):
	

	policy = np.ones([len(state), int(len(state)**(1/len(state[0])))]) / int(len(state)**(1/len(state[0])))
	iteration = 0
	while 1:
		V = PolicyEvaluation(policy, V, state, actions, rewards, theta, gamma)
		
		policy_stable = True
		old_policy = policy
		for s in state:
			cnt = s[0] * 4 + s[1]
			if cnt != 15 and cnt != 14 and cnt != 7:
				old_action = np.argmax(policy[cnt])
				
				pi_s = -10

				for a in actions[s]:
					if a == 'U':
						a_n = 0
						nxt = [s[0]-1, s[1]]
					if a == 'D':
						nxt = [s[0]+1, s[1]]
						a_n = 1
					if a == 'L':
						nxt = [s[0], s[1]-1]
						a_n = 2
					if a == 'R':
						nxt = [s[0], s[1]+1]
						a_n = 3

					nxt = tuple(nxt)
					pi = rewards[s] + (gamma * V[nxt])

					
					if pi > pi_s:
						pi_s = pi
						policy[cnt] = np.eye(4)[a_n]
				pi_s = np.argmax(policy[cnt])
					
				if old_action != pi_s:
					policy_stable = False

		iteration = iteration + 1
		
		if policy_stable:
			break


	result = list(V.values())
	print("Final value Function:")
	for x in range(len(result)):
		if  x!= 0 and x%4 == 0:
			print()
		print(result[x], end=" ")
		
	print()
	print()
	X = np.array([[0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0]])
	pol_arr = []
	print("Final Policy:")
	#print(iteration)
	pol_arr = np.reshape(np.argmax(policy, axis=1), X.shape)
	pol_arr[0][15] = 10
	pol_arr[0][7] = -1
	pol_arr[0][14] = -1
	for x in range(len(pol_arr[0])):
		if  x!= 0 and x%4 == 0:
			print()
		if pol_arr[0][x] == 0:
			print("U", end=" ")
		elif pol_arr[0][x] == 1:
			print("D", end=" ")
		elif pol_arr[0][x] == 2:
			print("L", end=" ")
		elif pol_arr[0][x] == 3:
			print("R", end=" ")
		elif pol_arr[0][x] == 10:
			print("G", end=" ")
		else:
			print("-", end=" ")
	

'''==================================================
Task Management
=================================================='''	
if task == "Value Iteration":
	ValueIteration(policy, V, state, actions, rewards, theta, gamma)
	
		
elif task == "Policy Iteration":
	PolicyImprovement(V, state, actions, rewards, theta, gamma)
	