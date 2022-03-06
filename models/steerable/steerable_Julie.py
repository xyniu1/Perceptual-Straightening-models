import torch
import torch.nn as nn

import models.steerable.steerableUtils as utils
from models.steerable.config import device, dtype

class SteerablePyramid(nn.Module):

    def __init__(self, imgSize, K=4, N=4, hilb=False, includeHF=True ):
        super(SteerablePyramid, self).__init__()

        size = [ imgSize, imgSize//2 + 1 ]
        self.hl0 = utils.HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ utils.freq_shift( size[0], True  ) ] 
        self.indB = [ utils.freq_shift( size[0], False ) ] 


        for n in range( self.N ):

            l = utils.L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            b = utils.B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            s = utils.S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( utils.freq_shift( size[0], True  ) )
            self.indB.append( utils.freq_shift( size[0], False ) )


    def forward(self, x):

#         fftfull = torch.rfft(x,2)
        fftfull = torch.fft.rfft2(x)
#         xreal = fftfull[... , 0]
#         xim = fftfull[... ,1]
        xreal = fftfull.real
        xim = fftfull.imag
        x = torch.cat((xreal.unsqueeze(1), xim.unsqueeze(1)), 1 ).unsqueeze( -3 ) # (1, 2, 1, 1, 256, 129)
        x = torch.index_select( x, -2, self.indF[0] )

        x   = self.hl0 * x # (1, 1, 1, 2, 256, 129)*(1, 2, 1, 1, 256, 129) -> (1, 2, 1, 2, 256, 129)
        h0f = x.select( -3, 0 ).unsqueeze( -3 ) # (1, 2, 1, 1, 256, 129)
        l0f = x.select( -3, 1 ).unsqueeze( -3 )
        lf  = l0f 

        output = []

        for n in range( self.N ):

            bf = self.b[n] *                     lf # (1, 1, 1, 8, 256, 129)*(1, 2, 1, 1, 256, 129) -> (1, 2, 1, 8, 256, 129)
            lf = self.l[n] * utils.central_crop( lf ) 
            if self.hilb:
                hbf = self.s[n] * torch.cat( (bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1 )
                bf  = torch.cat( ( bf , hbf ), -3 )
            if self.includeHF and n == 0:
                bf  = torch.cat( ( h0f,  bf ), -3 )

            output.append( bf )

        output.append( lf  ) 

        for n in range( len( output ) ):
            output[n] = torch.index_select( output[n], -2, self.indB[n] )
            sig_size = output[n].shape[-2]
#             output[n] = torch.stack((output[n].select(1,0), output[n].select(1,1)),-1)
            output[n] = torch.fft.irfft2(output[n], s=(sig_size, sig_size))

        if self.includeHF:
            output.insert( 0, output[0].narrow( -3, 0, 1                    ) )
            output[1]       = output[1].narrow( -3, 1, output[1].size(-3)-1 )

        for n in range( len( output ) ):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 = output[n].narrow( -3,     0, nfeat ).unsqueeze(1)
                    o2 = output[n].narrow( -3, nfeat, nfeat ).unsqueeze(1)
                    output[n] = torch.cat( (o1, o2), 1 ) 
                else:
                    output[n] = output[n].unsqueeze(1)

        return output
