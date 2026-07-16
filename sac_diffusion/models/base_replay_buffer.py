
class BaseReplayBuffer():
 def __init__(self):
  pass
 

 def save_transitions(self,transitions):
  
  raise NotImplementedError
 
 def load_transitions(self,critic):
  
  raise NotImplementedError