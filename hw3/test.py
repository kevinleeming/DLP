import torch
import torchvision
from dataloader import RetinopathyLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(model, loader, name):
	model.eval()
	y_pred = np.array([])
	y_true = np.array([])
	correct = 0
	for batch_idx, (data, target) in enumerate(loader):
		data, target = data.to(device), target.to(device)
		with torch.no_grad():
			output = model(data)
		y = output.data.max(1)[1]
		correct += y.eq(target.data).cpu().sum()

		y_pred = np.append(y_pred, y.data.cpu().numpy())
		y_true = np.append(y_true, target.data.cpu().numpy())

	accuracy = 100.0*correct/len(loader.dataset)
	print(f'{name}: Accuracy = {accuracy:.2f}%')

	disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=[0,1,2,3,4])
	disp.plot()
	plt.savefig(f'./{name}_Matrix.png')
	plt.clf()

if __name__ == '__main__':

	device = torch.device('cuda:0')

	test_transform = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		])
	test_dataset = RetinopathyLoader('./data/', 'test', test_transform)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=4)

	for _model in ['ResNet18', 'ResNet50']:
		for _pre in ['with pretraining', 'without pretraining']:
			if _model == 'ResNet18':
				model = torchvision.models.resnet18()
			else:
				model = torchvision.models.resnet50()
			in_features = model.fc.in_features
			model.fc = torch.nn.Linear(in_features, 5)
			model = model.to(device)
			model.load_state_dict(torch.load(f'./{_model}_{_pre}.tar')['state_dict'])

			plot_confusion_matrix(model, test_loader, f'{_model}_{_pre}')
