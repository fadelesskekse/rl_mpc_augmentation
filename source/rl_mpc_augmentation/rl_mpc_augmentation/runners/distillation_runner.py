import torch 
from torch.utils.data import DataLoader

class DistillationRunner:
    def __init__(self, env, teacher_policy, student_policy):
        self.env = env
        self.teacher = teacher_policy
        self.student = student_policy
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)

    def run(self, num_steps = 1000):
        obs, _ = self.env.reset()
        for _ in range(num_steps):
            # get teacher action (use scandots)
            with torch.no_grad():
                privileged_obs = obs["privileged"]
                teacher_action = self.teacher.get_action(privileged_obs)
            # get student action (use depth camera)
            depth_obs = obs["depth"]
            student_action = self.stduent.get_action(depth_obs)

            # loss: match the teacher's behavior
            loss = torch.nn.functional.mse_loss(student_action, teacher_action)

            # step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(teacher_action)
            
