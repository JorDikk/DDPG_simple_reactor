import numpy as np

class Reactor():
    def __init__(self):
        self.reset()

    def ReactorStep(self, action, dt):

        # translate action into feed fraction of A and B
        self.Ain = 1-action
        self.Bin = action

        # computing changes in A, B & C over timestep dt. According to a model of A + 2B <-> C in a CISTR reactor.
        dA = (self.phi * self.Ain / self.V - self.A * self.B ** 2 * self.r1 - self.phi / self.V * self.A + self.C * self.r2) * dt
        dB = (self.phi * self.Bin / self.V - 2 * self.A * self.B ** 2 * self.r1 - self.phi / self.V * self.B + 2 * self.C * self.r2) * dt
        dC = (self.A * self.B ** 2 * self.r1 - self.phi * self.C / self.V - self.C * self.r2) * dt

        # updating concentrations of A, B & C in reactor
        self.A += dA
        self.B += dB
        self.C += dC

        return self.A,self.B,self.C, np.float64(dC*10)

    def reset(self):
        self.phi = 0.5                  # feed flow to reactor
        self.r1 = 5                     # forward reaction rate constant
        self.r2 = 0.05                  # backward reaction rate constant
        self.V = 10                     # reactor volume
        self.A = 0.1                    # initial concentration of A in reactor
        self.B = 0.9                    # initial concentration of B in reactor
        self.C = 0                      # initial concentration of C in reactor
        self.fraction = 0.5             # initial fraction of B in the feed
        self.Ain = 1-self.fraction      # initial amount of A in feed
        self.Bin = self.fraction        # initial amount of B in feed

        state = (self.A, self.B, self.C, self.Ain, self.Bin)

        return state

