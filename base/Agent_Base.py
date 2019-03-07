"""
This file is used to module the Agent Base, all agent should sub this module 
obey the rule of OPENAI agents
"""

class Agent_Base():
	"""
	THis module is used as the base module of the agent of all users defined agents
	It has three methods 
	"""
		
	def  __init__(self):
		"""
		init method
		"""
		pass

	def act(self):
		"""
		agent use this method to act with the env
		"""
		pass
	
	def update(self):
		"""
		agent use this method to update the action value
		"""
		pass
 
