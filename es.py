import json
import os
import os.path as osp
from quantize_utils import *
from utils import *
import random
import conf
from hard import hardware
import pandas as pd
# import numpy as np


class GeneticAlgorithm:

    def __init__(
        self, qmodel, weight_path, arch, block_type,
        epoch, pop_size, sample_size, mutate_prob_split, mutate_prob_pi, 
        r_w=[2, 4, 6, 8], r_a=8, 
        device='cuda', 
        input_size=(3, 32, 32),
        log_dir='./log/',
        mem_dir='./mem/',
        batch_size=512, finetune_e=1, resume_epoch=0, dataset='cifar10', test=False, acc_module=False, predict=False, greedy=False) -> None:
        '''
        arch: 模型名 mobilenet\n
        qmodel: 模型
        '''
        print('ga init')
        print(dataset)
        self.epoch = epoch
        self.pop_size = pop_size
        self.sample_size = sample_size
        self.r_w = r_w
        self.r_a = r_a
        self.mutate_prob_split = mutate_prob_split
        self.mutate_prob_pi = mutate_prob_pi
        self.qmodel = qmodel
        self.gene_length = lenQmodel(qmodel)
        self.pop = {}
        self.arch =arch
        self.finetune_e = finetune_e
        self.predict = predict
        self.greedy = greedy
        self.a = 1
        self.b = 1
        self.c = 80
        print('predict:',predict,'//greedy:',greedy)

        self.device = device
        self.dataset = dataset
        if dataset == 'cifar10':
            self.input_size = input_size
            from githubtrain import valid_loader, train_loader
            self.testLoader = valid_loader
            self.trainLoader = train_loader
        else:
            self.input_size = (3, 224, 224)
            from imagenet_loader import data_loader
            # self.trainLoader, self.testLoader = data_loader(root='./imagenet10')
            self.trainLoader, self.testLoader = data_loader(root='../ImageNet')
        # self.trainLoader = cifar10DataLoader(train=True, shuffle=True, normalized=True, batch_size=batch_size)
        # self.testLoader = cifar10DataLoader(train=False, shuffle=False, normalized=True, batch_size=batch_size)
        
        self.sample, self.label = iter(self.trainLoader).__next__()
        
        self.weight_path = weight_path
        # print(acc_module == True)

        if test == True:
            from acc_module.models.cnn import CNN
            self.premodel = CNN(1,3)
            state_dict = torch.load(f'./acc_module/output/CNNcifar10.pth')
            self.premodel.load_state_dict(state_dict['model'])
            self.premodel = self.premodel.to(self.device)
            self.premodel.eval()
            pass
        elif acc_module == True:
            self.log_file = osp.join(log_dir, 'acc_module', conf.TIME_NOW)
            os.makedirs(self.log_file)
        elif predict == True:
            from acc_module.models.cnn import CNN
            self.premodel = CNN(1,3)
            state_dict = torch.load(f'./acc_module/output/CNNcifar10.pth')
            self.premodel.load_state_dict(state_dict['model'])
            self.premodel = self.premodel.to(self.device)
            self.premodel.eval()

            self.log_file = osp.join(log_dir, 'usepre',self.arch+self.dataset, conf.TIME_NOW)
            os.makedirs(self.log_file)
            self.mem_dir = mem_dir
            self.mem_dir = osp.join(self.mem_dir, 'usepre',self.arch+self.dataset, conf.TIME_NOW)
            os.makedirs(self.mem_dir)
            self.mem = []
            self.privacy_mem = {}
            self.energy_mem = {}
            self.accur_mem = {}
            # DSP BRAM
            self.hardware_mem = {}
            self.point = {}
            load_file(self.privacy_mem, osp.join(mem_dir, self.arch, 'privacy.log'))
            load_file(self.energy_mem, osp.join(mem_dir, self.arch, 'energy.log'))
            load_file(self.accur_mem, osp.join(mem_dir, self.arch, 'accur.log'))
            load_file(self.hardware_mem, osp.join(mem_dir, self.arch, 'hardware.log'))


        else:  # normal run
            self.log_file = osp.join(log_dir, self.arch+self.dataset, conf.TIME_NOW)
            os.makedirs(self.log_file)
            self.mem_dir = mem_dir
            self.mem_dir = osp.join(self.mem_dir, self.arch+self.dataset, conf.TIME_NOW)
            os.makedirs(self.mem_dir)
            self.mem = []
            self.privacy_mem = {}
            self.energy_mem = {}
            self.accur_mem = {}
            # DSP BRAM
            self.hardware_mem = {}
            self.point = {}
            load_file(self.privacy_mem, osp.join(mem_dir, self.arch, 'privacy.log'))
            load_file(self.energy_mem, osp.join(mem_dir, self.arch, 'energy.log'))
            load_file(self.accur_mem, osp.join(mem_dir, self.arch, 'accur.log'))
            load_file(self.hardware_mem, osp.join(mem_dir, self.arch, 'hardware.log'))


        self.block_type = block_type
        self.num_blocks = lenQmodel(self.qmodel, self.block_type)
        self.num_layers = lenQmodel(self.qmodel)
        self.top1 = 0.7200
        print(f'GeneticAlgorithm已经初始化,全精度模型准确度{self.top1}')

        self.resume_epoch = resume_epoch

    def start(self):
        self.pop = {}

    def init(self, pop):
        self.pop = pop

    def rand_init_pop(self, use_pre=False):
        while len(self.pop) < self.pop_size:
            strategy = self.__random_init()
            if strategy in self.pop.keys():
                continue
            else:
                self.pop[strategy] = self.fitness_func(strategy, len(self.pop)+1, use_pre)
    
    def __random_split(self):
        '''
        -1表示原始图片传输
        '''
        return random.randint(1, self.num_blocks-1) if self.num_blocks > 1 else 0
    
    def __greedy_split(self, epoch=0):
        '''
        贪婪渐进量化，先[1,2,3,4],在加入4-6,然后7-9,最后10-13.
        '''
        # if epoch <= (self.epoch/5):
        #     p = np.array([0.2, 0.2, 0.2, 0.4])
        #     return int(np.random.choice([1, 2, 3, 4], p = p.ravel()))
        # elif epoch <= (self.epoch*2 /5):
        #     p = np.array([0.4, 0.4, 0.2])
        #     return int(np.random.choice([5,6,7], p = p.ravel()))
        # elif epoch <= (self.epoch *3 /5):
        #     p = np.array([0.4, 0.4, 0.2])
        #     return int(np.random.choice([8,9,10], p = p.ravel()))
        # elif epoch <= (self.epoch *4 /5):
        #     p = np.array([0.2, 0.3, 0.4, 0.1])
        #     return int(np.random.choice([11,12,13,14], p = p.ravel()))
        # else:
        #     return 14
        
        if epoch <= 500:
            p = np.array([0.2, 0.2, 0.2, 0.4])
            return int(np.random.choice([1, 2, 3, 4], p = p.ravel()))
        elif epoch <= 1000:
            p = np.array([0.4, 0.4, 0.2])
            return int(np.random.choice([5,6,7], p = p.ravel()))
        elif epoch <= 2000:
            p = np.array([0.4, 0.4, 0.2])
            return int(np.random.choice([8,9,10], p = p.ravel()))
        elif epoch <= 3000:
            p = np.array([0.2, 0.3, 0.4, 0.1])
            return int(np.random.choice([11,12,13,14], p = p.ravel()))
        else:
            return 14


    def __random_bit(self):
        return random.sample(self.r_w, 1)[0]

    def __random_init(self):
        # split = self.__random_split()
        # split = self.__greedy_split()
        if self.greedy == False:
            split = 14
        else:
            split = self.__greedy_split()
        greedy_strategy = {0:1, 1:3, 2:5, 3:7, 4:9, 5:11, 6:13, 7:15, 8:17, 9:19, 10:21, 11:23, 12:25, 13:27, 14:28}
        quant_policy = tuple(self.__random_bit() if length+1 <= greedy_strategy[split] else 8 for length in range(self.gene_length))
        return (split, quant_policy)

    def test_init(self):
        return self.__random_init()

    def random_sample(self):
        keys = random.sample(list(self.pop), self.sample_size)
        return {k: self.pop[k] for k in keys}

    def mutate(self, parent, epoch):
        '''
        原版 无交叉 贪婪
        '''
        split = parent[0]
        quant_policy = parent[1]
        new_split = split if random.random() > self.mutate_prob_split else self.__greedy_split(epoch)
        greedy_strategy = {0:1, 1:3, 2:5, 3:7, 4:9, 5:11, 6:13, 7:15, 8:17, 9:19, 10:21, 11:23, 12:25, 13:27, 14:28}
        new_quant_policy = tuple([8 if length+1 > greedy_strategy[split] else quant_policy[length]
                                   if random.random() > self.mutate_prob_pi else self.__random_bit() for length in range(self.gene_length)])
        return (new_split, new_quant_policy)
    
    def cross_over(self, parent1, parent2, epoch):
        '''
        split选取两者最小的  每位编码在父代范围中选择或随机 自带变异
        '''
        split1, split2 = parent1[0], parent2[0]
        split = split1 if split1 > split2 else split2
        greedy_strategy = {0:1, 1:3, 2:5, 3:7, 4:9, 5:11, 6:13, 7:15, 8:17, 9:19, 10:21, 11:23, 12:25, 13:27, 14:28}
        # new_split = split1 if random.random() < 0.5 else split2
        # new_split = (split1 if random.random() < 0.5 else split2) if random.random() > self.mutate_prob_split else self.__greedy_split(epoch=epoch)
        if self.greedy == False:
            new_split = 14
        else:
            new_split = self.__greedy_split(epoch=epoch)
        # new_split = (split1 if random.random() < 0.5 else split2) if random.random() > self.mutate_prob_split else self.__random_split()
        policy = []
        for i in range(len(parent1[1])):
            if i < greedy_strategy[new_split]:
                if random.random() > self.mutate_prob_pi:
                    if parent1[1][i] == parent2[1][i]:
                        policy.append(parent1[1][i])
                    else:
                        max_bit = max(parent1[1][i], parent2[1][i])
                        min_bit = min(parent1[1][i], parent2[1][i])
                        policy.append(random.randrange(min_bit,max_bit+2,2))
                else:
                    policy.append(self.__random_bit())
            else:
                policy.append(8)

        new_quant_policy = tuple(policy)
        return (new_split, new_quant_policy)

    def selection(self, samples):
        '''
        best, worst, choose1, choose2
        '''
        # 大到小排序
        sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        best, worst = sorted_samples[0][0], sorted_samples[-1][0]
        choose1 = sorted_samples[np.random.choice([0,1,2,3,4,5,6,7,8], p = np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]).ravel())][0]
        choose2 = sorted_samples[np.random.choice([0,1,2,3,4,5,6,7,8], p = np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]).ravel())][0]
        return best, worst, choose1, choose2

    def __add(self, strategy, epoch, verbose, use_pre):
        if strategy in self.pop.keys():
            return
        else:
            self.pop[strategy] = self.fitness_func(strategy, epoch, use_pre)
            if verbose and (epoch -21)%100==0 :
                print(f'epoch: {epoch -21}, offspring: {strategy} : {self.pop[strategy]}\n')

    def run(self, verbose=False, init_pop=None, init_point=None): 
        self.start()
        if init_pop:
            self.pop = init_pop
            self.point = init_point
            for epoch in self.point.keys():
                qat, acc, fitness = self.point[epoch]
                pic_pop(acc, qat, fitness, epoch, resume=True)

        self.rand_init_pop()
        for i in range(self.resume_epoch, self.epoch):
            self.on_generation(idx=i)

            samples = self.random_sample()
            best, worst, choose1, choose2 = self.selection(samples)
            if verbose:
                print(f'iteration: {i}\nbest: {best} : {self.pop[best]}\nworst: {worst} : {self.pop[worst]}')
            del self.pop[worst]
            while len(self.pop) < self.pop_size:
                # children = self.mutate(best, i)
                children = self.cross_over(choose1, choose2, i)
                self.__add(children, i+21, verbose)
        # self.on_stop()
        print('all do well')
        # return self.best()

    def predict_run(self, verbose=True, init_pop=None, init_point=None): 
        self.start()
        if init_pop:
            self.pop = init_pop
            self.point = init_point
            for epoch in self.point.keys():
                qat, acc, fitness = self.point[epoch]
                pic_pop(acc, qat, fitness, epoch, resume=True)

        self.rand_init_pop(self.predict)
        for i in range(self.resume_epoch, self.epoch):
            self.on_generation(idx=i)

            samples = self.random_sample()
            best, worst, choose1, choose2 = self.selection(samples)
            # if verbose:
            #     print(f'iteration: {i}\nbest: {best} : {self.pop[best]}\nworst: {worst} : {self.pop[worst]}')
            del self.pop[worst]
            while len(self.pop) < self.pop_size:
                # if self.greedy:
                #     children = self.mutate(best, i)
                # else:
                children = self.cross_over(choose1, choose2, i)

                self.__add(children, i+21, verbose, use_pre=self.predict)
        # self.on_stop()
        print('predict do well, start finetune')

        if False:# write csv  last log for dsp - acc
            sortpop = sorted(self.pop.items(),  key=lambda d: d[1], reverse=True)

            pi = []
            fitness = []
            dsp = []
            bram = []
            a = [self.a] * len(sortpop[:-1])
            b = [self.b] * len(sortpop[:-1])
            c = [self.c] * len(sortpop[:-1])
            for i in sortpop[:-1]:
                pi.append(i[0][1]) 
                fitness.append(i[1])
                dsp.append(hardware.ComputeDsp(i[0][1]))
                bram.append(hardware.ComputeBram(i[0][1]))
            pi1 = torch.tensor(pi)
            # print(pi)
            # print(pi1)
            pi1 = pi1.reshape(-1,1,28).to('cuda')
            top1 = self.premodel(pi1)
            accuary = top1.reshape(-1).to('cpu').detach().numpy()
            # print(type(accuary),type(dsp[-1]))

            #字典中的key值即为csv中列名
            dataframe = pd.DataFrame({'pi':pi,'fitness':fitness,'acc':accuary,'dsp':dsp,'bram':bram,'a':a,'b':b,'c':c})

            #将DataFrame存储为csv,index表示是否显示行名，default=True

            if os.path.exists("./log/last_log.csv"):
                dataframe.to_csv("./log/last_log.csv",mode='a',index=False,sep=',',header=False)
            else:
                dataframe.to_csv("./log/last_log.csv",index=False,sep=',')

        if True:# write csv  last log for dsp - bit
            sortpop = sorted(self.pop.items(),  key=lambda d: d[1], reverse=True)

            dsp = []
            bram = []
            bitops = []
            size = []
            if self.c == 3:
                c = ['nano'] * len(sortpop[:-1])
            elif self.c == 20:
                c = ['tiny'] * len(sortpop[:-1])
            elif self.c == 80:
                c = ['medium'] * len(sortpop[:-1])
            elif self.c == 120:
                c = ['large'] * len(sortpop[:-1])
            for i in sortpop[:-1]:
                dsp.append(hardware.ComputeDsp(i[0][1])/6331)
                bram.append(hardware.ComputeBram(i[0][1])/1682)
                bitops.append(self.energy_mem[i[0]][1])
                size.append(self.energy_mem[i[0]][0])

            #字典中的key值即为csv中列名
            dataframe = pd.DataFrame({'DSP':dsp,'Bram':bram,'BOPs':bitops,'Size':size,'class':c})

            #将DataFrame存储为csv,index表示是否显示行名，default=True

            if os.path.exists("./log/dsp_bit_relat2.csv"):
                dataframe.to_csv("./log/dsp_bit_relat2.csv",mode='a',index=False,sep=',',header=False)
            else:
                dataframe.to_csv("./log/dsp_bit_relat2.csv",index=False,sep=',')

        if False:# write csv  last log for dsp - bram
            # c = 20
            # a = 1/2/3   more a less dsp
            sortpop = sorted(self.pop.items(),  key=lambda d: d[1], reverse=True)

            dsp = []
            bram = []
            pi = []
            if self.a == 1 and self.b == 1:
                c = ['balance'] * len(sortpop[:-1])
            elif self.a == 2 and self.b == 1:
                c = ['limited dsp'] * len(sortpop[:-1])
            elif self.a == 3 and self.b == 1:
                c = ['extremely limited dsp'] * len(sortpop[:-1])
            elif self.a == 1 and self.b == 2:
                c = ['limited bram'] * len(sortpop[:-1])
            elif self.a == 1 and self.b == 3:
                c = ['extremely limited bram'] * len(sortpop[:-1])
            for i in sortpop[:-1]:
                pi.append(i[0][1])
                dsp.append(hardware.ComputeDsp(i[0][1])/6331)
                bram.append(hardware.ComputeBram(i[0][1])/1682)
            pi1 = torch.tensor(pi)
            pi1 = pi1.reshape(-1,1,28).to('cuda')
            top1 = self.premodel(pi1)
            accuary = top1.reshape(-1).to('cpu').detach().numpy()

            #字典中的key值即为csv中列名
            dataframe = pd.DataFrame({'dsp':dsp,'bram':bram,'class':c,'acc':accuary})

            #将DataFrame存储为csv,index表示是否显示行名，default=True

            if os.path.exists("./log/dsp_bram.csv"):
                dataframe.to_csv("./log/dsp_bram.csv",mode='a',index=False,sep=',',header=False)
            else:
                dataframe.to_csv("./log/dsp_bram.csv",index=False,sep=',')


    def test(self, verbose=True):
        self.start()
        

        # self.rand_init_pop()
        split = 14
        # quant_policy = tuple([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
        # split = 14
        quant_policy = tuple([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
        # quant_policy = tuple([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
        # quant_policy = tuple([6, 6, 6, 6, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 4, 6, 4, 4, 4, 4, 2, 8, 2, 8, 2, 8, 2, 2]) large
        # quant_policy = tuple([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        # quant_policy = tuple([6, 6, 4, 8, 4, 6, 4, 6, 6, 8, 4, 6, 4, 8, 4, 6, 4, 4, 4, 6, 2, 6, 2, 4, 2, 6, 2, 2]) medium
        # quant_policy = tuple([6, 6, 4, 6, 4, 6, 4, 4, 4, 6, 4, 6, 4, 6, 4, 4, 4, 4, 2, 6, 2, 4, 2, 2, 2, 2, 2, 2]) 20
        # quant_policy = tuple([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        print(quant_policy)
        # split = 14
        # quant_policy = tuple([2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 2, 8, 2, 8])
        # split = 14
        # quant_policy = tuple([8, 6, 6, 2, 4, 6, 4, 4, 4, 8, 4, 6, 4, 2, 4, 6, 2, 8, 2, 4, 2, 4, 4, 8, 4, 8, 2, 6])

        
        strategy = (split, quant_policy)
        split, pi = strategy
        load_qnet(self.qmodel, self.weight_path)
        mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=split, block_type=self.block_type, a_bit=self.r_a)

        # 计算params
        total_params, total_MACs = energy_eval_bw(self.qmodel, self.input_size, device=self.device)    
        print(total_params)
        print(total_MACs)
        
        # top1 = finetune(self.qmodel, self.trainLoader, epochs=90, device=self.device, testloader=self.testLoader, verbose=verbose, dataset=self.dataset)
        # top1, _ = test(self.qmodel, self.testLoader, device=self.device)
        # top1 = top1.item()
        # if verbose:
        #     print(split,'  ',pi)
        #     print('top1:',top1)
        
        
        print('test done')

    def predict_test(self):
        '''测试premodel可用性'''
        self.start()
        

        # self.rand_init_pop()
        split = 14
        # quant_policy = tuple([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
        # quant_policy = tuple([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
        # quant_policy = tuple([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
        # quant_policy = tuple([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        # quant_policy = tuple([4, 6, 4, 4, 4, 4, 4, 4, 4, 6, 4, 6, 4, 8, 4, 4, 4, 6, 2, 2, 2, 6, 2, 8, 2, 8, 2, 2]) # tiny
        # quant_policy = tuple([6, 6, 6, 6, 6, 6, 4, 6, 4, 6, 4, 4, 6, 6, 4, 6, 4, 4, 4, 4, 2, 8, 2, 8, 2, 8, 2, 2]) # large
        # quant_policy = tuple([6, 6, 4, 6, 4, 4, 4, 4, 4, 6, 4, 6, 4, 8, 4, 6, 2, 6, 2, 6, 2, 2, 2, 4, 2, 4, 2, 2]) # nano
        # quant_policy = tuple([6, 6, 4, 8, 4, 6, 4, 6, 6, 8, 4, 6, 4, 8, 4, 6, 4, 4, 4, 6, 2, 6, 2, 4, 2, 6, 2, 2]) # medium

        quant_policy = tuple([8,6,6,6,4,4,6,8,4,6,8,6,6,8,6,6,6,6,6,8,6,8,8,6,6,6,6,8]) # releq
        
        strategy = (split, quant_policy)
        split, pi1 = strategy

        load_qnet(self.qmodel, self.weight_path)
        mixed_quant_with_partition_bw(self.qmodel, strategy=pi1, split=split, block_type=self.block_type, a_bit=self.r_a)
        # torch.save(self.qmodel.state_dict(), './saved_models/large.pth')
        # 计算params
        total_params, total_MACs = energy_eval_bw(self.qmodel, self.input_size, device=self.device)    
        print(total_params)
        print(total_MACs)
        print(hardware.ComputeDsp(pi1))
        print(hardware.ComputeBram(pi1))

        print('predict test done')


    def relat(self, pop_size =1000):
        '''this is for DSP and BOPs'''
        self.start()
        DSP = []
        Bram = []
        BOPs = []
        Size = []
        print('start com')
        while len(DSP) < pop_size:
            pi = tuple(self.__random_bit() for length in range(self.gene_length))

            load_qnet(self.qmodel, self.weight_path)
            mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=14, block_type=self.block_type, a_bit=self.r_a)

            # 计算params
            size, bops = energy_eval_bw(self.qmodel, self.input_size, device=self.device)
            Size.append(size/33672704.0)
            BOPs.append(bops/36350230528.0)
            DSP.append(hardware.ComputeDsp(pi)/6331)
            Bram.append(hardware.ComputeBram(pi)/1680.5)

        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'DSP':DSP,'Bram':Bram,'BOPs':BOPs,'Size':Size})

        #将DataFrame存储为csv,index表示是否显示行名，default=True

        if os.path.exists("./log/new_relat.csv"):
            dataframe.to_csv("./log/new_relat.csv",mode='a',index=False,sep=',',header=False)
        else:
            dataframe.to_csv("./log/new_relat.csv",index=False,sep=',')

        
        print('predict test done')




    # acc_module
    def acc_module(self, verbose=True):
        self.start()
        for i in range(self.epoch):
            strategy = self.__random_init()
            split, pi = strategy
            split = 14
            if pi in self.pop.keys():
                continue
            else:
                
                load_qnet(self.qmodel, self.weight_path)
                mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=split, block_type=self.block_type, a_bit=self.r_a, acc_module=True)
                top1 = finetune(self.qmodel, self.trainLoader, epochs=70, device=self.device, testloader=self.testLoader, verbose=False, dataset=self.dataset)
                top1 = top1.item()
                if verbose:
                    print(f'iteration: {i}\n{pi}:{top1}')
                self.pop[pi] = top1
                write_file(self.pop, osp.join(self.log_file, 'acc_module'))

    def on_generation(self, idx):
        '''
        每轮开始时动作,向文件中写入pop
        '''
        write_file(self.pop, osp.join(self.log_file, str(idx)))
        write_file(self.privacy_mem, osp.join(self.mem_dir, 'privacy'))
        write_file(self.energy_mem, osp.join(self.mem_dir, 'energy'))
        write_file(self.accur_mem, osp.join(self.mem_dir, 'accur'))
        write_file(self.hardware_mem, osp.join(self.mem_dir, 'hardware'))
        write_file(self.point, osp.join(self.mem_dir, 'point'))

    # def on_stop(self):
    #     '''
    #     搜素结束时动作，将所有mem写入文件
    #     '''
    #     write_file(self.privacy_mem, osp.join(self.mem_dir, 'privacyf'))
    #     write_file(self.energy_mem, osp.join(self.mem_dir, 'energyf'))
    #     write_file(self.accur_mem, osp.join(self.mem_dir, 'accurf'))

    # def best(self):
    #     '''
    #     strategy, accur, energy, privacy
    #     '''
    #     best_strategy, _ = self.selection(self.pop)
    #     return best_strategy, self.pop[best_strategy], self.accur_mem[best_strategy], self.energy_mem[best_strategy], self.privacy_mem[best_strategy]
    

    def fitness_func(self, strategy, epoch=10, use_pre=False):
        if use_pre == False:
            split, pi = strategy
            load_qnet(self.qmodel, self.weight_path)
            mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=split, block_type=self.block_type, a_bit=self.r_a)

            if strategy not in self.mem:
                top1 = finetune(self.qmodel, self.trainLoader, epochs=self.finetune_e, device=self.device, testloader=self.testLoader, dataset=self.dataset)
                top1 = top1.item()
                
                total_params, total_MACs = energy_eval_bw(self.qmodel, self.input_size, device=self.device)
                total_DSP = hardware.ComputeDsp(pi)/6331
                total_Bram = hardware.ComputeBram(pi)/1682
                
                self.mem.append(strategy)
                self.accur_mem[strategy] = top1 / self.top1

                self.energy_mem[strategy] = (total_params/28765696, total_MACs/741670912) #2966028288
                # 186535424.0/145336041472.0
                #  33672704.0/ 36350230528.0
                self.hardware_mem[strategy] = (total_DSP, total_Bram)
                


                
                privacy = 1 - ((self.a*total_DSP * self.b*total_Bram)**3 + self.c *(1 - (top1 / self.top1))**3)**(1/3)
                
                self.privacy_mem[strategy] = privacy
                self.point[epoch] = (total_DSP * total_Bram, top1 / self.top1, privacy)

                # draw pic
                pic_pop(top1 / self.top1, total_DSP * total_Bram, privacy, epoch)

            else:
                privacy, top1, _ = self.privacy_mem[strategy], self.accur_mem[strategy], self.energy_mem[strategy]

            fitness = privacy

        else:
            split, pi = strategy
            load_qnet(self.qmodel, self.weight_path)
            mixed_quant_with_partition_bw(self.qmodel, strategy=pi, split=split, block_type=self.block_type, a_bit=self.r_a)

            if strategy not in self.mem:
                split, pi= strategy
                pi1 = torch.tensor(pi)
                pi1 = pi1.reshape(1,1,28).to('cuda')
                top1 = self.premodel(pi1)

                top1 = top1.item()
                
                total_params, total_MACs = energy_eval_bw(self.qmodel, self.input_size, device=self.device)
                total_DSP = hardware.ComputeDsp(pi)/6331
                total_Bram = hardware.ComputeBram(pi)/1682
                
                self.mem.append(strategy)
                self.accur_mem[strategy] = top1 / self.top1
                self.energy_mem[strategy] = (total_params/28765696, total_MACs/2966028288)
                self.hardware_mem[strategy] = (total_DSP, total_Bram)
                
                privacy = 1 - ((self.a*total_DSP * self.b*total_Bram)**3 + self.c *(1 - (top1 / self.top1))**3)**(1/3)
                
                self.privacy_mem[strategy] = privacy
                self.point[epoch] = (total_DSP * total_Bram, top1 / self.top1, privacy)

                # draw pic
                pic_pop(top1 / self.top1, total_DSP * total_Bram, privacy, epoch, use_pre)

            else:
                privacy, top1, _ = self.privacy_mem[strategy], self.accur_mem[strategy], self.energy_mem[strategy]

            fitness = privacy

        

        return fitness


def load_file(dest, path):
    '''
    dest: dict k=(split, (*pi)), v=float\n
    path: path\n
    '''
    try:
        with open(path) as f:
            for line in f:
                (split, pi), value = json.loads(line)
                dest[(split, tuple(pi))] = value
        # print(f'{len(dest)} data loaded')
    except FileNotFoundError:
        # print('No file Found')
        pass

def load_point(dest, path):
    '''
    dest: dict epoch=(quant, acc, fitness)\n
    path: path\n
    '''
    try:
        with open(path) as f:
            for line in f:
                epoch, (quant, acc, fitness) = json.loads(line)
                dest[epoch] = (quant, acc, fitness)
        # print(f'{len(dest)} data loaded')
    except FileNotFoundError:
        # print('No file Found')
        pass 

def write_file(src, path):
    '''
    dest: dict k=(split, (*pi)), v=float\n
    path: path\n
    '''
    with open(path, 'w') as f:
        for k, v in src.items():
            print(json.dumps((k, v)), file=f)
    # print(f'wirte {len(src)} data')

def pic_pop(acc, quant, fitness, epoch, resume=False, use_pre=False):
    '''
    show the fitness power
    '''
    import matplotlib.pyplot as plt
    
    # x=[0.1, 0.2, 0.3]
    # y=[0.1, 0.2, 0.3]
    
    
    # 1. 首先是导入包，创建数据
    
    # x = np.random.rand(n) * 2# 随机产生10个0~2之间的x坐标
    # y = np.random.rand(n) * 2# 随机产生10个0~2之间的y坐标
    # 2.创建一张figure
    # fig = plt.figure(1)
    # 3. 设置颜色 color 值【可选参数，即可填可不填】，方式有几种
    # colors = np.random.rand(n) # 随机产生10个0~1之间的颜色值，或者
    ### TODO:color
    # colors = ['r', 'g', 'y', 'b', 'r', 'c', 'g', 'b', 'k', 'm']  # 可设置随机数取
    colors = ['k', 'r', 'orange', 'yellow', 'g', 'springgreen', 'b', 'purple']
    # colors = ['navy', 'b', 'blueviolet', 'mediumorchid', 'magenta', 'deeppink', 'palevioletred','pink']
    if fitness >= 0.8:
        color = colors[0]
    elif fitness >= 0.7:
        color = colors[1]
    elif fitness >= 0.6:
        color = colors[2]
    elif fitness >= 0.5:
        color = colors[3]
    elif fitness >= 0.4:
        color = colors[4]
    elif fitness >= 0.3:
        color = colors[5]
    elif fitness >= 0.2:
        color = colors[6]
    else:
        color = colors[7]

    if epoch == -1:
        pass
    elif epoch == 1:
        fig = plt.figure(1)
        plt.xlabel('quant_state')
        plt.ylabel('acc_state')
        plt.title('pop')
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)
        plt.scatter(quant, acc,s= 3,c=color)
        # plt.savefig('./pic/pic_{}.png'.format(1))
    elif (epoch)%20 == 0 and not resume:
        plt.scatter(quant, acc,s= 3,c=color)
        if use_pre == True:
            plt.savefig('./pre_pic/pic_{}.png'.format(epoch))
        else:
            plt.savefig('./pic/pic_{}.png'.format(epoch))
    
    else:
        plt.scatter(quant, acc,s= 3,c=color)
        # plt.savefig('./pic/pic_{}.png'.format(2))
        return plt



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # plt.imread('./pic/old/pic_280.png')
    # pic_pop(1, 0.1, 1, 281)
    pop = {}
    plt.scatter(0.1, 1, 3)
    plt.savefig('./pic/new.png')
    # pic_pop(0.6, 0.6, 0, 1)