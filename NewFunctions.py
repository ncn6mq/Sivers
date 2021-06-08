import lhapdf
import numpy as np

class Hadron(object):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, pdfset='cteq61',
                 ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
                 ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
        '''
        Parent class of individual hadron functions as defined in Sivers Extraction with Neural Network (2021)
        '''
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffDataPIp = lhapdf.mkPDF(ff_PIp, 0)
        self.ffDataPIm = lhapdf.mkPDF(ff_PIm, 0)
        self.ffDataPIsum = lhapdf.mkPDF(ff_PIsum, 0)
        self.ffDataKAp = lhapdf.mkPDF(ff_KAp, 0)
        self.ffDataKAm = lhapdf.mkPDF(ff_KAm, 0)
        # needs to be extended to generalize for kaons
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3
        self.e = 1
        
        self.ffDict = {0: self.ffDataPIp,
                       1: self.ffDataPIm,
                       2: self.ffDataPIsum,
                       3: self.ffDataKAp,
                       4: self.ffDataKAm}
    

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])
    
    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])    
    
    
    def A0(self, z, pht, m1):
        ks2avg = (self.kperp2avg*m1**2)/(m1**2 + self.kperp2avg) #correct 
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2 #correct
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg #correct
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg) #correct
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg) #correct
        last = np.sqrt(2*self.e) * z * pht / m1 #correct
        
        return (topfirst/bottomfirst) * np.exp(-exptop/expbottom) * last
    
    
    def NN(self, x, n, a, b):
        return n * x**a * (1 - x)**b * (((a + b)**(a + b))/(a**a * b**b))

    
    def NNanti(self, n):
        return n
    
    def sivers(self, kinsandflag, Nu, Nd, Ns, Nubar, Ndbar, Nsbar, au, ad, as0, bu, bd, bs, m1):
        
        res = np.zeros(len(kinsandflag))
        for i in range(5):
            idxs = kinsandflag[:, 4] == i
                   
            x = kinsandflag[idxs, 0]
            z = kinsandflag[idxs, 1]
            pht = kinsandflag[idxs, 2]
            QQ = kinsandflag[idxs, 3]
            
            a0 = self.A0(z, pht, m1)
            
            nnu = self.NN(x, Nu, au, bu)
            nnubar = self.NNanti(Nubar)
            nnd = self.NN(x, Nd, ad, bd)
            nndbar = self.NNanti(Ndbar)
            nns = self.NN(x, Ns, as0, bs)
            nnsbar = self.NNanti(Nsbar)

            sexpr = self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ)
            sbarexpr = self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ)
            uexpr = self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ)
            ubarexpr = self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ)
            dexpr = self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ)
            dbarexpr = self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ)
                        
            numerator = nnu*uexpr + nnubar*ubarexpr + nnd*dexpr + nndbar*dbarexpr + nns*sexpr + nnsbar*sbarexpr
            
            denominator = uexpr + ubarexpr + dexpr + dbarexpr + sexpr + sbarexpr
            
            res[idxs] = a0*(numerator/denominator)
        
        return res

    
    def siversTest(self, kinsandflag, Nu, Nd, Ns, Nubar, Ndbar, Nsbar, au, ad, as0, bu, bd, bs, m1):
        
        res = []
        for j in range(len(kinsandflag)):
            i = kinsandflag[j, 4]
                   
            x = kinsandflag[[j], 0]
            z = kinsandflag[[j], 1]
            pht = kinsandflag[[j], 2]
            QQ = kinsandflag[[j], 3]
            
            a0 = self.A0(z, pht, m1)
            
            eu = self.NN(x, Nu, au, bu)
            nnubar = self.NNanti(Nubar)
            nnd = self.NN(x, Nd, ad, bd)
            nndbar = self.NNanti(Ndbar)
            nns = self.NN(x, Ns, as0, bs)
            nnsbar = self.NNanti(Nsbar)

            sexpr = self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ)
            sbarexpr = self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ)
            uexpr = self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ)
            ubarexpr = self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ)
            dexpr = self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ)
            dbarexpr = self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ)
                        
            numerator = nnu*uexpr + nnubar*ubarexpr + nnd*dexpr + nndbar*dbarexpr + nns*sexpr + nnsbar*sbarexpr
            
            denominator = uexpr + ubarexpr + dexpr + dbarexpr + sexpr + sbarexpr
            
            res += list(a0*(numerator/denominator)[0])
        
        return np.array(res)