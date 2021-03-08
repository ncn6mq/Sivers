import lhapdf
import numpy as np


class Hadron(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo', ff_kaon='JAM19FF_kaon_nlo'):
        
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffData = lhapdf.mkPDF(ff_pion, 0)
        # needs to be extended to generalize for kaons
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        self.e = 1/137
    

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
    
    def ff(self, flavor, z, QQ):
        return np.array([self.ffData.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])
    
    
    def A0(self, z, pht, m1):
        ks2avg = self.kperp2avg*m1**2/(m1**2 + self.kperp2avg)
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * self.kperp2avg**2
        bottomfirst = (z**2 * ks2avg + self.pperp2avg) * self.kperp2avg**2
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg)
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg)
        last = np.sqrt(2*self.e) * z * pht / m1
        
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    
    
    def NN(self, x, n, a, b):
        return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

    def NNanti(self, x, n):
        return n
            
    
class PiPlus(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion)
        
    
    def sivers(self, kins, Nu, au, bu, Ndbar, adbar, bdbar, m1):
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        topleft = self.NN(x, Nu, au, bu) * self.eu**2 * self.pdf(2, x, QQ) * self.ff(2, z, QQ)
        topright = self.NNanti(x, Ndbar) * self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(-1, z, QQ)
        bottomleft = self.eu**2 * self.pdf(2, x, QQ) * self.ff(2, z, QQ)
        bottomright = self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(-1, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    
class PiMinus(Hadron):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo'):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion)
        
    
    def sivers(self, kins, Nd, ad, bd, Nubar, aubar, bubar, m1):
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        QQ = kins[:, 3]
        a0 = self.A0(z, pht, m1)
        topleft = self.NN(x, Nd, ad, bd) * self.ed**2 * self.pdf(1, x, QQ) * self.ff(1, z, QQ)
        topright = self.NNanti(x, Nubar) * self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(-2, z, QQ)
        bottomleft = self.ed**2 * self.pdf(1, x, QQ) * self.ff(1, z, QQ)
        bottomright = self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(-2, z, QQ)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
 