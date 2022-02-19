class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        current_rate = self.start ** (current_step//self.decay)
        if current_rate < self.end:
            return self.end
        else:
            return current_rate