import numpy as np

task = "Policy Iteration"
'''==================================================
Initial set up
=================================================='''


#Hyperparameters
theta = 0.005
gamma = 0.9
NOISE = 0.10

#Define all states
state=[]
for i in range(4):
    for j in range(4):
        state.append((i,j))

#Define rewards for all states
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

#Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
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

#Define an initial policy
policy={}
for s in actions.keys():
    policy[s] = np.random.choice(actions[s])
	


#Define initial value function 
V={}
for s in state:
    if s in actions.keys():
        V[s] = 0
    else:
        V[s] = -1

V[(3,3)] = 1

print("-----------V------------")
print(V)
print		
V_old = V.copy()
result_old = list(V_old.values())
for x in range(len(result_old)):
	if  x!= 0 and x%4 == 0:
		print()
	print(result_old[x], end=" ")
print()
print()
'''==================================================
Value Iteration
=================================================='''
def ValueIteration(policy, V, state, actions, rewards, theta, gamma, NOISE):
	
	iteration = 0
	while 1:
		delta = 0
		for s in state:
			if s in policy:

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


					#Calculate the value
					nxt = tuple(nxt)
					v = rewards[s] + (gamma * V[nxt])
					if v > new_v: #Is this the best action so far? If so, keep it
						new_v = v
						policy[s] = a

				#Save the best of all actions for the state                                
				V[s] = new_v
				delta = max(delta, np.abs(old_v - V[s]))


		#See if the loop should stop now         
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
			
			
def PolicyEvaluation(V, state, actions, rewards, theta, gamma, NOISE):
	V_old = V.copy()
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


						#Calculate the value
						nxt = tuple(nxt)
						v = rewards[s] + (gamma * V[nxt])
						new_v = v

					#Save the best of all actions for the state                                
					V[s] = new_v
					delta = max(delta, np.abs(old_v - V[s]))


		#See if the loop should stop now         
		if delta < theta:
			break
		iteration += 1
	return V
		
		
		
		
def PolicyImprovement(policy, V, state, actions, rewards, theta, gamma, NOISE):
	for s in V.keys():
		policy[s] = "-"
	print("-------------------")
	print(policy)
	for s in actions.keys():
		policy[s] = 0
	print("-------------------")
	print(policy)
	iteration = 0
	while 1:
		V = PolicyEvaluation(V, state, actions, rewards, theta, gamma, NOISE)
		
		policy_stable = True
		
		for s in state:
				# The best action we would take under the current policy
				if policy[s] != "-":
					old_action = policy[s]
					pi_s = 0
					
					# Find the best action by one-step lookahead
					# Ties are resolved arbitarily
					for a in actions[s]:
						if a == 'U':
							nxt = [s[0]-1, s[1]]
						if a == 'D':
							nxt = [s[0]+1, s[1]]
						if a == 'L':
							nxt = [s[0], s[1]-1]
						if a == 'R':
							nxt = [s[0], s[1]+1]

						#Calculate the value
						nxt = tuple(nxt)
						pi = rewards[s] + (gamma * V[nxt])

						
						if pi > pi_s: #Is this the best action so far? If so, keep it
							pi_s = pi
							policy[s] = a
						
					
					# Greedily update the policy
					if old_action != pi_s:
						policy_stable = False
			
			# If the policy is stable we've found an optimal policy. Return it
		iteration = iteration + 1
		
		if policy_stable:
			break
		if iteration >= 5000:
			break


	result = list(V.values())
	print("Final value Function:")
	for x in range(len(result)):
		if  x!= 0 and x%4 == 0:
			print()
		print(result[x], end=" ")
		
	print()
	print()
	print("Final Policy:")
	result_policy = list(policy.values())
	for x in range(len(result_policy)):
		if  x!= 0 and x%4 == 0:
			print()
		print(result_policy[x], end=" ")
	
	
	
	
	
	
	
	
	
	
	
	
if task == "Value Iteration":
	ValueIteration(policy, V, state, actions, rewards, theta, gamma, NOISE)
	
		
elif task == "Policy Iteration":
	PolicyImprovement(policy, V, state, actions, rewards, theta, gamma, NOISE)
	
