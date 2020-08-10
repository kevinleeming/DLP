import torch
import torchvision
import argparse
from dataloader import RetinopathyLoader
from matplotlib import pyplot as plt

def train(model, loader, Loss, optimizer, device):
	model.train()
	train_loss = 0
	correct = 0
	for batch_idx, (data, target) in enumerate(loader):
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = Loss(output, target)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		y_pred = output.data.max(1)[1]
		correct += y_pred.eq(target.data).cpu().sum()

	train_loss /= len(loader)
	accuracy = 100.0*correct/len(loader.dataset)
	print(f'\tTrain Data: Accuracy = {accuracy:.2f}%, Loss = {train_loss:.2f}')
	return accuracy

def evaluate(model, loader, Loss, device):
	model.eval()
	test_loss = 0
	correct = 0
	for batch_idx, (data, target) in enumerate(loader):
		data, target = data.to(device), target.to(device)
		with torch.no_grad():
			output = model(data)
		test_loss += Loss(output, target).item()
		y_pred = output.data.max(1)[1]
		correct += y_pred.eq(target.data).cpu().sum()

	test_loss /= len(loader)
	accuracy = 100.0*correct/len(loader.dataset)
	print(f'\tTest Data: Accuracy = {accuracy:.2f}%, Loss = {test_loss:.2f}')
	return accuracy

if __name__ == '__main__':

	# Data
	train_transform = torchvision.transforms.Compose([
			torchvision.transforms.RandomHorizontalFlip(p=0.5),
			torchvision.transforms.RandomVerticalFlip(p=0.5),
			torchvision.transforms.RandomRotation(degrees=45),
			torchvision.transforms.ColorJitter(0.05,0.05,0.05,0.05),
			torchvision.transforms.ToTensor(),	# Normalization + Transpose
		])
	test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		])

	device = torch.device('cuda:0')
	epochs = 5
	batch_size = 8
	train_dataset = RetinopathyLoader('./data/', 'train', train_transform)
	test_dataset = RetinopathyLoader('./data/', 'test', test_transform)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

	# model
	for _model in ['ResNet50']:
		for _pre in ['with pretraining', 'without pretraining']:
			print(f'==============={_model}_{_pre}===============')

			if _model == 'ResNet18' and _pre == 'with pretraining':
				model = torchvision.models.resnet18(pretrained=True)
			elif _model == 'ResNet18' and _pre == 'without pretraining':
				model = torchvision.models.resnet18()
			elif _model == 'ResNet50' and _pre == 'with pretraining':
				model = torchvision.models.resnet50(pretrained=True)
			elif _model == 'ResNet50' and _pre == 'without pretraining':
				model = torchvision.models.resnet50()

			in_features = model.fc.in_features
			model.fc = torch.nn.Linear(in_features, 5)
			model = model.to(device)

			Loss = torch.nn.CrossEntropyLoss()
			optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

			train_acc_list = []
			test_acc_list = []
			best_accuracy = 0
			for i in range(epochs):
				print(f'Train Epoch #{i}:')
				train_acc_list.append(train(model, train_loader, Loss, optimizer, device))
				test_acc = evaluate(model, test_loader, Loss, device)
				test_acc_list.append(test_acc)
				if test_acc >= best_accuracy:
					torch.save({
							'epoch': i,
							'state_dict': model.state_dict(),
						}, f'{_model}_{_pre}.tar')
					best_accuracy = test_acc
			x_list = [i for i in range(epochs)]
			plt.plot(x_list, train_acc_list, label=f'Train({_pre})')
			plt.plot(x_list, test_acc_list, label=f'Test({_pre})')

		plt.title(f'Result Comparsion({_model})')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(f'./{_model}.png')
		plt.clf()

		#plot_confusion_matrix(model, test_loader)
