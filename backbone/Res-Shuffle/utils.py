import os
import re
import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):

	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss


class AvgrageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0
		self.val = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res


def save_checkpoint(state, iters, tag=''):
	if not os.path.exists("./models"):
		os.makedirs("./models")
	filename = os.path.join("./models/{}checkpoint-{:06}.pth.tar".format(tag, iters))
	torch.save(state, filename)

def get_lastest_model():
	if not os.path.exists('./models'):
		os.mkdir('./models')
	model_list = os.listdir('./models/')
	if model_list == []:
		return None, 0
	model_list.sort()
	lastest_model = model_list[-1]
	iters = re.findall(r'\d+', lastest_model)
	return './models/' + lastest_model, int(iters[0])


def get_parameters(model):
	group_no_weight_decay = []
	group_weight_decay = []
	for pname, p in model.named_parameters():
		if pname.find('weight') >= 0 and len(p.size()) > 1:
			# print('include ', pname, p.size())
			group_weight_decay.append(p)
		else:
			# print('not include ', pname, p.size())
			group_no_weight_decay.append(p)
	assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
	groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
	return groups


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))