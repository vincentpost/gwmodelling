import numpy as np

class Model:
    
    def __init__(self, T):
        self.T = T
        self.elements = [] # list of all elements
        
    def omega(self, x, y):
        omega = 0 + 0j
        for e in self.elements:
            omega += e.omega(x, y)
        return omega
    
    def potential(self, x, y):
        return self.omega(x, y).real
    
    def streamfunction(self, x, y):
        return self.omega(x, y).imag
        
    def head(self, x, y):
        return self.potential(x, y) / self.T
    
    def solve(self):
        esolve = [e for e in self.elements if e.nunknowns == 1]
        nunknowns = len(esolve)
        matrix = np.zeros((nunknowns, nunknowns))
        rhs = np.zeros(nunknowns)
        for irow in range(nunknowns):
            matrix[irow], rhs[irow] = esolve[irow].equation()
        solution = np.linalg.solve(matrix, rhs)
        for irow in range(nunknowns):
            esolve[irow].parameter = solution[irow]
            
class Element:
    
    def __init__(self, model, p):
        self.model = model
        self.parameter = p
        self.nunknowns = 0
        self.model.elements.append(self)
        
    def omega(self, x, y):
        return self.parameter * self.omegainf(x, y)
    
    def potential(self, x, y):
        return self.omega(x, y).real
    
    def potinf(self, x, y):
        return self.omegainf(x, y).real

class Well(Element):
    
    def __init__(self, model, xw=0, yw=0, Q=1, rw=0.3):
        Element.__init__(self, model, Q)
        self.zw = xw + 1j * yw
        self.rw = rw
            
    def omegainf(self, x, y):
        zminzw = x + 1j * y - self.zw
        zminzw = np.where(np.abs(zminzw) < self.rw, self.rw, zminzw)
        return 1 / (2 * np.pi) * np.log(zminzw)
    
class UniformFlow(Element):
    
    def __init__(self, model, gradient, angle):
        Element.__init__(self, model, model.T * gradient)
        self.udir = np.exp(-1j * np.deg2rad(angle))
        
    def omegainf(self, x, y):
        return -self.udir * (x + y * 1j)
    
class HeadEquation:

    def equation(self):
        row = []
        rhs = self.pc
        for e in self.model.elements:
            if e.nunknowns == 1:
                row.append(e.potinf(self.xc, self.yc))
            else:
                rhs -= e.potential(self.xc, self.yc)
        return row, rhs
    
class HeadWell(Well, HeadEquation):
    
    def __init__(self, model, xw, yw, rw, hw):
        Well.__init__(self, model, xw, yw, 0, rw)
        self.xc = xw + rw
        self.yc = yw
        self.pc = self.model.T * hw
        self.nunknowns = 1
        
class Constant(Element, HeadEquation):
    
    def __init__(self, model, xc, yc, hc):
        Element.__init__(self, model, 0)
        self.xc, self.yc = xc, yc
        self.pc = self.model.T * hc
        self.nunknowns = 1
        
    def omegainf(self, x, y):
        return np.ones_like(x, dtype='complex')
    
class LineSink(Element):
    
    def __init__(self, model, x0=0, y0=0, x1=1, y1=1, sigma=1):
        Element.__init__(self, model, sigma)
        self.z0 = x0 + y0 * 1j
        self.z1 = x1 + y1 * 1j
        self.L = np.abs(self.z1 - self.z0)
    
    def omegainf(self, x, y):
        zeta = x + y * 1j
        Z = (2 * zeta - (self.z0 + self.z1)) / (self.z1 - self.z0)
        Zp1 = np.where(np.abs(Z + 1) < 1e-12, 1e-12, Z + 1)
        Zm1 = np.where(np.abs(Z - 1) < 1e-12, 1e-12, Z - 1)
        return self.L / (4 * np.pi) * (Zp1 * np.log(Zp1) - Zm1 * np.log(Zm1))
    
class HeadLineSink(LineSink, HeadEquation):
    
    def __init__(self, model, x0=0, y0=0, x1=1, y1=1, hc=1):
        LineSink.__init__(self, model, x0, y0, x1, y1, 0)
        self.xc = 0.5 * (x0 + x1)
        self.yc = 0.5 * (y0 + y1)
        self.pc = self.model.T * hc
        self.nunknowns = 1