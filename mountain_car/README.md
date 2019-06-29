# Project: Mountain-Car
## Add solution for DQN 19-6-29
You can run the DQN train by running the file: train_agent_dqn.py  
### new files are: 
train_agent_dqn.py  
batch_dqn_agent.py  
model_dqn.py

## File struct:
main: mountain_car_op.py  
agent: 	mocar_one_sarsa-agent.py  
	mocar_sarsa_agent.py  
sparse coding:	tile_coding.py  

## code environment:
1. Python 3.6  
2. numpy  
3. tqdm  
4. gym  
5. Pytorch = 1.0+

## Run method:
1. change the mountain_car_op.py parameter, if you want to run one-step sarsa, then change   
line 8 for "from mocar_one_sarsa_agent"  
or "from mocar_sarsa_agent" for n-step sarsa.  

2. run the code in folder "mountain_car/"  
	"python mountain_car_op.py"   
3. check the train result in "data/", the file name is structed by:"one" or "nstep" + data + time  
4. plot the data by matlab  
