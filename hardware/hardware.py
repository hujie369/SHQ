import math


# the number of per conv
con_wei_num = [27, 9,32, 9,64, 9,128, 9,128, 9,256, 9,256, 9,512, 9,512, 9,512, 9,512, 9,512, 9,512, 9,1024, 1024]

out_channel = [32, 32,64, 64,128, 128,128, 128,256, 256,256, 256,512, 512,512, 512,512, 512,512, 512,512, 512,512, 512,1024, 1024,1024, 1000]  #TODO:类别数传入

# the clk dilution
dilution = [4,4,4,16,16,16,16,64,64,64,64,256,256,256,256,256,256,256,256,256,256,256,256,1024,1024,1024,1024,1024]

###############################################################################################
# compute DSP
def reuse_dsp(pi, redsp):
    '''
    use act_bits and wei_bits to design per dsp computing how many 
    pi:qat policy
    redsp:the reuse of per dsp
    '''
    wei2act = {8:8, 6:6, 4:6, 2:4}  # find the act_bit to the wei_bit
    PW2reuse = {8:2, 6:2, 4:3, 2:5}  # find the reuse of dsp to PW wei_bit
    DW2reuse = {8:1, 6:2, 4:2, 2:3}  # find the reuse of dsp to DW wei_bit
    for i in range(len(pi)-1):
        if i%2 == 0:  # conv or PW
            redsp.append(PW2reuse[pi[i]])
        else:  # DW
            redsp.append(DW2reuse[pi[i]])
    redsp.append(PW2reuse[pi[-1]])  # Linear

def numdsp(DSP, redsp):
    '''
    compute dsp of each layer: the number of per conv * out_channel / dilution / redsp
    '''
    for i in range(len(redsp)):
        DSP.append(math.ceil(con_wei_num[i] * out_channel[i] / dilution[i] / redsp[i]))



def ComputeDsp(quant_policy):
    '''mix-qat about dsp, return total num of dsp'''
    # the reuse of per dsp
    redsp = []
    reuse_dsp(quant_policy, redsp)
    DSP = []
    numdsp(DSP, redsp)
    return sum(DSP)
#################################################################################

#################################################################################
# compute bram
def ComputeBram(quant_policy, bias=False):
    '''mix-qat about bram, return total num of bram.
    Brams = out_channel * the num of per conv * wei_bit /72 /dilution 
    '''
    weiBram = []
    for i in range(len(quant_policy)):
        if i < 23:  
            bramdata = out_channel[i] * con_wei_num[i] * quant_policy[i] / 72 / dilution[i]
            weiBram.append(math.ceil(bramdata) if bramdata%1 > 0.5 or bramdata%1 == 0 else int(bramdata)+0.5)
        elif i == 23:  # CL13 weight_bram=1
            weiBram.append(1)
        else:  # 72 -> 36
            # print(f'{i}:{out_channel[i]} * {con_wei_num[i]} * {quant_policy[i]} / 36 / {dilution[i]}')
            bramdata = out_channel[i] * con_wei_num[i] * quant_policy[i] / 36 / dilution[i]
            weiBram.append(math.ceil(bramdata) if bramdata%1 > 0.5 or bramdata%1 == 0 else int(bramdata)+0.5)

    if bias == False:
        return sum(weiBram)
    else:
        bias_bits = 16
        for i in range(len(quant_policy)):
            if i < 23:  
                bramdata = out_channel[i] * bias_bits / 72 / dilution[i]
                weiBram.append(math.ceil(bramdata) if bramdata%1 > 0.5 or bramdata%1 == 0 else int(bramdata)+0.5)
            elif i == 23:  # CL13 bias_bram=1
                weiBram.append(1)
            else:  # 72 -> 36
                bramdata = out_channel[i] * bias_bits / 36 / dilution[i]
                weiBram.append(math.ceil(bramdata) if bramdata%1 > 0.5 or bramdata%1 == 0 else int(bramdata)+0.5)
        print(weiBram)
        new = []
        for i in range(len(quant_policy)):
            new.append(weiBram[i]+weiBram[i+28])
        print(new)
        return sum(weiBram)





if __name__ == '__main__':
    split = 14
    quant_policy = tuple([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
    print(ComputeDsp(quant_policy))
    print(ComputeBram(quant_policy, True))