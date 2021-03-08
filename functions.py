import lhapdf
import numpy as np


class hadronFunc(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo', ff_kaon='JAM19FF_kaon_nlo', QQ=10):
        
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffData = lhapdf.mkPDF(ff_pion, 0)
        # needs to be extended to generalize for kaons
        self.QQ = QQ
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        self.e = 1/137
    

    def pdf(flavor, x):
        return np.array([self.pdfData.xfxQ2(flavor, ax, self.QQ) for ax in x])
    
    def ff(flavor, z):
        return np.array([self.ffData.xfxQ2(flavor, az, self.QQ) for az in z])
    
    
    def A0(z, pht, m1):
        ks2avg = self.kperp2avg*m1^2/(m1^2 + self.kperp2avg)
        topfirst = (np.square(z)*self.kperp2avg + self.pperp2avg)*kperp2avg^2
        bottomfirst = (np.square(z)*ks2avg + self.pperp2avg) * kperp2avg^2
        exptop = np.square(pht) * np.square(z) * (ks2avg - self.kperp2avg)
        expbottom = (np.square(z)*ks2avg + self.pperp2avg) * (np.square(z)*self.kperp2avg + self.pperp2avg)
        last = np.sqrt(2*self.e) * z * pht / m1
        
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    
    
    def NN(x, n, a, b):
        return n*np.power(x, a) * np.power(1 - x, b) * (((a + b)^(a + b))/(a^a * b^b))

    def NNanti(x, n):
        return n
            
    
class piplus(hadronFunc):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='JAM19PDF_proton_nlo',
                 ff_pion='JAM19FF_pion_nlo', QQ=10):
        
        super().__init__(kperp2avg=kperp2avg, pperp2avg=pperp2avg, pdfset=pdfset,
                 ff_pion=ff_pion, QQ=QQ)
        
    
    def sivers(self, kins, Nu, au, bu, Ndbar, adbar, bdbar, m1):
        x = kins[:, 0]
        z = kins[:, 1]
        pht = kins[:, 2]
        a0 = self.A0(z, pht, m1)
        topleft = self.NN(x, Nu, au, bu) * self.eu^2 * self.pdf(2, x) * self.ff(2, z)
        topright = self.NNanti(x, Ndbar) * self.edbar^2 * self.pdf(-1, x) * self.ff(-1, z)
        bottomleft = self.eu^2 * self.pdf(2, x) * self.ff(2, z)
        bottomright = self.edbar^2 * self.pdf(-1, x) * self.ff(-1, z)
        return a0*((topleft + topright)/(bottomleft + bottomright))
    
    