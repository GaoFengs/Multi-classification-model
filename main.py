#coding=utf-8
"""
 主程序：主要完成四个功能
（1）训练：定义网络，损失函数，优化器，进行训练，生成模型
（2）验证：验证模型准确率
（3）测试：测试模型在测试集上的准确率
（4）help：打印log信息

"""
import torch

from config import opt
import os
#import models
import torch as t
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from torch.autograd import Variable
from torchvision import models
from torch import nn
import time
import csv
import numpy as np


"""模型训练：定义网络，定义数据，定义损失函数和优化器，训练并计算指标，计算在验证集上的准确率"""
def train(**kwargs):
	
	"""根据命令行参数更新配置"""
	opt.parse(kwargs)
	#vis = Visualizer(opt.env)


	"""(1)step1：加载网络，若有预训练模型也加载"""
	#model = getattr(models,opt.model)()
	model = models.alexnet(pretrained = True)
	#model.aux_logits = False
	#num_ftrs = model.fc.in_features
	print(model)
	# 自己构建的分类模型有8个类
	#model.classifier = nn.Linear(2208, 2)
	model.classifier = nn.Sequential(
		nn.Dropout(p=0.5, inplace=False),
		nn. Linear(in_features=9216, out_features=4096, bias=True),
		nn.ReLU(inplace=True),
		nn. Dropout(p=0.5, inplace=False),
		nn.Linear(in_features=4096, out_features=4096, bias=True),
		nn.ReLU(inplace=True),
		nn.Linear(in_features=4096, out_features=2, bias=True)
	)
	#model.fc = nn.Dropout(0.2)
	#model.fc = nn.Softmax(2)

	#if opt.load_model_path:
	#	model.load(opt.load_model_path)
	if opt.use_gpu: #GPU
		model.cuda()

	"""(2)step2：处理数据"""
	train_data = DogCat(opt.train_data_root,train=True) #训练集
	val_data = DogCat(opt.train_data_root,train=False) #验证集

	train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
	val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)

	"""(3)step3：定义损失函数和优化器"""
	criterion = t.nn.CrossEntropyLoss() #交叉熵损失
	lr = opt.lr #学习率
	optimizer = t.optim.SGD(model.parameters(),lr=opt.lr,weight_decay = opt.weight_decay)

	"""(4)step4：统计指标，平滑处理之后的损失，还有混淆矩阵"""
	loss_meter = meter.AverageValueMeter()
	confusion_matrix = meter.ConfusionMeter(2)
	previous_loss = 1e10

	"""(5)开始训练"""
	for epoch in range(opt.max_epoch):

		loss_meter.reset()
		confusion_matrix.reset()

		for ii,(data,label) in enumerate(train_dataloader):



			#训练模型参数
			input = Variable(data)

			target = Variable(label)
	
			if opt.use_gpu:
				input = input.cuda()
				target = target.cuda()
			
			#梯度清零
			optimizer.zero_grad()
			score = model(input)
			#result = np.round(score.cpu().detach().numpy(),3)
			#print(score.data)
			#print(result)
			#print(target.cpu().detach().numpy())
		
			loss = criterion(score,target)
			loss.backward() #反向传播
			
			#更新参数
			optimizer.step()

			#更新统计指标及可视化
			loss_meter.add(loss.item())
			#print score.shape,target.shape
			print(np.round(score.cpu().detach().numpy()).shape)
			print(np.round(target.cpu().detach().numpy()))
			confusion_matrix.add(score.detach(),target.detach())
			print("ii:", ii,'loss',loss_meter.value()[0])

			if ii % opt.print_freq == opt.print_freq-1:
				#vis.plot('loss',loss_meter.value()[0])
				#print('loss',loss_meter.value()[0])
				if os.path.exists(opt.debug_file):
					import ipdb;
					ipdb.set_trace()


		"""计算验证集上的指标及可视化"""
		val_cm,val_accuracy = val(model,val_dataloader)
		#vis.plot('val_accuracy',val_accuracy)
		#vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".
			#format(epoch=epoch,loss=loss_meter.value()[0],val_cm=str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))

		print("epoch:",epoch,"loss:",loss_meter.value()[0],"accuracy:",val_accuracy)

		if val_accuracy == 100.0:
			break

		"""如果损失不再下降，则降低学习率"""
		if loss_meter.value()[0] > previous_loss:
			lr = lr*opt.lr_decay
			for param_group in optimizer.param_groups:
				param_group["lr"] = lr

		previous_loss = loss_meter.value()[0]
		# model.save()
		name = time.strftime('model' + '%m%d_%H_%M_%S.pth')
		t.save(model.state_dict(), 'checkpoints/' + name)
	# model.save()
	name = time.strftime('model' + '%m%d_%H_%M_%S.pth')
	t.save(model.state_dict(), 'checkpoints/' + name)

"""计算模型在验证集上的准确率等信息"""
@t.no_grad()
def val(model,dataloader):

	model.eval() #将模型设置为验证模式

	confusion_matrix = meter.ConfusionMeter(2)
	for ii,data in enumerate(dataloader):
		input,label = data
		# val_input = Variable(input,volatile=True)
		# val_label = Variable(label.long(),volatile=True)
		with torch.no_grad():
			val_input = Variable(input)
		with torch.no_grad():
			val_label = Variable(label.long())
		if opt.use_gpu:
			val_input = val_input.cuda()
			val_label = val_label.cuda()

		score = model(val_input)
		confusion_matrix.add(score.detach().squeeze(),label.long())

	model.train() #模型恢复为训练模式
	cm_value = confusion_matrix.value()
	accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

	return confusion_matrix,accuracy

""""""
def test(**kwargs):
	opt.parse(kwargs)

	#data
	test_data = DogCat(opt.test_data_root,test=True)
	test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False,num_workers=opt.num_workers)
	results = []

	#model
	model = models.densenet161(pretrained = True)
	#model.fc = nn.Linear(2048,2)
	#model.fc = nn.Dropout(0.2)
	model.fc = nn.Linear(1000,2)
	model.load_state_dict(t.load('./model.pth'))
	if opt.use_gpu:
		model.cuda()
	model.eval()

	for ii,(data,path) in enumerate(test_dataloader):
		#input = Variable(data,volatile=True)
		with torch.no_grad():
			input = Variable(data)
		if opt.use_gpu:
			input = input.cuda()
		score = model(input)
		#print(score)
		path = path.numpy().tolist()
		#print path
		#print score.data,"+++++"
		_,predicted = t.max(score.data,1)
		print(predicted)
		#print "***************"
		#print predicted
		predicted = predicted.data.cpu().numpy().tolist()
		res = ""
		for (i,j) in zip(path,predicted):
			if j==1:
				res="CN"
			else:
				res="AD"
			results.append([i,"".join(res)])
		#print results

	write_csv(results,opt.result_file)
	print('完成预测！')
	return results

""""""
def write_csv(results,file_name):
	with open(file_name,"w") as f:
		writer = csv.writer(f)
		writer.writerow(['id','label'])
		writer.writerows(results)

if __name__ == '__main__':
	# import fire
	# fire.Fire()
	train()
	#test()



