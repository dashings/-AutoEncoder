import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    batch_size = 128
    epochs = 100
    torch.manual_seed(1)
    learning_rate = 1e-3


    input_dim= 28*28
    hidden_dim=400
    code_dim=20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(r'D:\data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(r'D:\data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)


    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            # encoder
            self.fc1 = nn.Linear(input_dim, hidden_dim)  # 784 , 400
            self.fc21 = nn.Linear(hidden_dim, code_dim)  # 400 , 20
            self.fc22 = nn.Linear(hidden_dim, code_dim)  # 400 , 20

            # decoder
            self.fc3 = nn.Linear(code_dim, hidden_dim)  # 20 , 400
            self.fc4 = nn.Linear(hidden_dim, input_dim)  # 400 , 784

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def loss_function(recon_x, x, mu, logvar):

      BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

      return BCE + KLD


    def train():
        model.train()
        train_loss = 0
        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        return train_loss

    def test():
      model.eval()
      test_loss = 0
      with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
          data = data.to(device)
          recon_batch, mu, logvar = model(data)
          test_loss += loss_function(recon_batch, data, mu, logvar).item()

      return test_loss


    with torch.no_grad():
        print('Before Trainig')
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        fig = plt.figure(figsize=(5, 5))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(sample[i].reshape(28, 28), cmap='gray')
            plt.axis('off')

    plt.show()

    tar = []
    ter = []

    for e in range(epochs):
        tr = train() / len(train_loader.dataset)
        te = test() / len(test_loader.dataset)
        tar.append(tr)
        ter.append(te)
        display.clear_output(wait=True)
        print('Epoch : ', e + 1, '/', epochs)

        with torch.no_grad():

            fig = plt.figure(figsize=(5, 5))
            fig.suptitle('Reconstruction')
            for i in range(25):
                sample = model(train_loader.dataset[i][0].to(device))[0].cpu()
                plt.subplot(5, 5, i + 1)
                plt.imshow(sample.reshape(28, 28), cmap='gray')
                plt.axis('off')

        plt.show()

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            fig = plt.figure(figsize=(5, 5))
            fig.suptitle('Generation')
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.imshow(sample[i].reshape(28, 28), cmap='gray')
                plt.axis('off')

        plt.show()

    plt.style.use('ggplot')
    plt.title('Loss Plot')
    plt.plot(tar,label='Training Loss')
    plt.plot(ter,label='Testing Loss')
    plt.legend()
    plt.show()

    from sklearn.manifold import TSNE
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    N=700
    lat=[]
    labels=[]
    for i in range(N):
      dat=train_loader.dataset[i][0]
      label=train_loader.dataset[i][1]
      ou=model.encode(dat.reshape(1,784).to(device))
      latent=model.reparameterize(ou[0],ou[1])
      lat.append(latent.cpu().detach().numpy())
      labels.append(label)

    lat=np.array(lat)
    lat=lat.reshape(N,20)


    lat2 = TSNE(n_components=2).fit_transform(lat)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    cmap = plt.cm.tab20b
    cmaplist = [cmap(i) for i in range(cmap.N)]
    bounds = np.linspace(0,10,11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scat = ax.scatter(lat2[:,0],lat[:,1],c=labels,cmap=cmap,norm=norm)
    cb = plt.colorbar(scat,ticks=[i for i in range(10)])
    cb.set_label('Labels')
    ax.set_title('TSNE plot for VAE Latent Space colour coded by Labels')
    plt.show()


    trainx=[]
    trainy=[]
    testx=[]
    testy=[]
    model.eval()
    for i in range(len(train_loader.dataset)):
      dat = train_loader.dataset[i][0]
      ou=model.encode(dat.reshape(1,784).to(device))
      latent=model.reparameterize(ou[0],ou[1])
      trainx.append(latent.cpu().detach().numpy().reshape(20))
      trainy.append(train_loader.dataset[i][1])

    for i in range(len(test_loader.dataset)):
      dat = test_loader.dataset[i][0]
      ou=model.encode(dat.reshape(1,784).to(device))
      latent=model.reparameterize(ou[0],ou[1])
      testx.append(latent.cpu().detach().numpy().reshape(20))
      testy.append(test_loader.dataset[i][1])


    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    import scikitplot as skplt

    clf=SVC()
    clf.fit(trainx,trainy)
    predy=clf.predict(testx)
    clf.score(testx,testy)


    print(classification_report(testy,predy))
    skplt.metrics.plot_confusion_matrix(testy, predy,figsize=(12,12))
    torch.save(model.state_dict(), 'model_weights.pt')