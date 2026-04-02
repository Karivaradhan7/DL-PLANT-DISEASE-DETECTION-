import os
import shutil
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from src.data.dataloader import make_dataloaders, set_seed
from src.models.classifiers import MLPClassifier, CNNClassifier
from src.models.temporal import PretrainedExtractor, SequenceModel
from src.models.autoencoder import ConvAutoencoder
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator
from src.utils.trainer import train_epoch, eval_model, plot_loss, plot_accuracy, plot_confusion, ensure_dir
from src.utils.misc import get_device, set_seed as set_seed_util


def get_dataloaders(cfg):
    # Get workspace root (parent of DL-Plant-Disease-System)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(current_dir)
    
    # Primary check: /workspaces/DL-PLANT-DISEASE-DETECTION-/data/PlantVillage
    workspace_data_path = os.path.join(workspace_root, 'data', 'PlantVillage')
    
    # Secondary check: relative to DL-Plant-Disease-System
    relative_data_path = os.path.join(current_dir, 'data', 'PlantVillage')
    
    # Handle PlantVillage at workspace root
    workspace_pv_path = os.path.join(workspace_root, 'PlantVillage')
    if os.path.isdir(workspace_pv_path) and not os.path.isdir(workspace_data_path):
        print(f"🔧 Moving {workspace_pv_path} to {workspace_data_path}")
        os.makedirs(os.path.dirname(workspace_data_path), exist_ok=True)
        shutil.move(workspace_pv_path, workspace_data_path)

    # Handle PlantVillage relative to DL-Plant-Disease-System
    relative_pv_path = os.path.join(current_dir, 'PlantVillage')
    if os.path.isdir(relative_pv_path) and not os.path.isdir(relative_data_path):
        print(f"🔧 Moving {relative_pv_path} to {relative_data_path}")
        os.makedirs(os.path.dirname(relative_data_path), exist_ok=True)
        shutil.move(relative_pv_path, relative_data_path)

    # Flatten if nested
    for data_path in [workspace_data_path, relative_data_path]:
        nested = os.path.join(data_path, 'PlantVillage')
        if os.path.isdir(nested):
            print(f"🔧 Flattening nested {nested}")
            for entry in os.listdir(nested):
                src = os.path.join(nested, entry)
                dst = os.path.join(data_path, entry)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            try:
                os.rmdir(nested)
            except:
                pass

    # Check paths in priority order
    possible_paths = [
        workspace_data_path,
        relative_data_path,
        "/kaggle/input/plantvillage-dataset/PlantVillage",
        "C:/Users/Shreenivasan/Downloads/archive/PlantVillage"
    ]

    DATA_PATH = None
    for path in possible_paths:
        if os.path.isdir(path):
            DATA_PATH = path
            break

    if DATA_PATH is None:
        print("⚠️ No valid dataset path found. Creating a minimal synthetic dataset.")
        os.makedirs(workspace_data_path, exist_ok=True)

        dummy_classes = ['class_a', 'class_b']
        for cls in dummy_classes:
            cls_dir = os.path.join(workspace_data_path, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(8):
                arr = (np.random.rand(cfg['image_size'], cfg['image_size'], 3) * 255).astype('uint8')
                img = Image.fromarray(arr)
                img.save(os.path.join(cls_dir, f'{cls}_{i}.png'))

        DATA_PATH = workspace_data_path
        print(f"✅ Synthetic dataset created at {DATA_PATH}")
    
    print(f"✅ Using dataset path: {DATA_PATH}")

    subdirs = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    if not subdirs:
        raise ValueError(f"No class subdirectories found in {DATA_PATH}. Expected structure: PlantVillage/<class_folders>")

    num_images = 0
    for cls in subdirs:
        cls_dir = os.path.join(DATA_PATH, cls)
        for root, _, files in os.walk(cls_dir):
            num_images += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    print(f"✅ Found {len(subdirs)} classes and {num_images} images")

    return make_dataloaders(DATA_PATH, cfg['image_size'], cfg['batch_size'], cfg['num_workers'], cfg['seed'])


def train_review1(train_loader, val_loader, test_loader, classes, cfg, device):
    # Define directories
    out_dir = os.path.join(cfg['paths']['results_dir'], 'review1')
    model_dir = os.path.join(cfg['paths']['model_dir'], 'review1')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    classifiers = {
        'mlp': MLPClassifier(num_features=3*cfg['image_size']*cfg['image_size'], hidden_size=cfg['experiments']['review1']['mlp_hidden'], num_classes=len(classes)),
        'cnn': CNNClassifier(num_classes=len(classes))
    }

    results = {}
    for name, model in classifiers.items():
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(cfg['num_epochs']):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = eval_model(model, val_loader, criterion, device)
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['val_acc'].append(val_metrics['acc'])
            print(f"[{name}] Epoch {epoch+1}/{cfg['num_epochs']} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        test_metrics = eval_model(model, test_loader, criterion, device)
        results[name] = test_metrics

        plot_loss(history, os.path.join(out_dir, f'{name}_loss.png'), title=f'{name} Loss')
        plot_accuracy(history, os.path.join(out_dir, f'{name}_acc.png'), title=f'{name} Accuracy')
        plot_confusion(test_metrics['confusion'], classes, os.path.join(out_dir, f'{name}_confusion.png'))

        # Save to both locations
        torch.save(model.state_dict(), os.path.join(out_dir, f'{name}_model.pt'))
        torch.save(model.state_dict(), os.path.join(model_dir, f'{name}.pth'))
        print(f"✅ Saved {name} model to {model_dir}/{name}.pth")

    print('✅ Review1 completed')


def extract_features(model, dataloader, device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            feats.append(out.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_review2(train_loader, val_loader, test_loader, classes, cfg, device):
    # Define directories
    out_dir = os.path.join(cfg['paths']['results_dir'], 'review2')
    model_dir = os.path.join(cfg['paths']['model_dir'], 'review2')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for base_name in cfg['experiments']['review2']['pretrained_models']:
        extractor = PretrainedExtractor(model_name=base_name, pretrained=True).to(device)
        feat_train, lab_train = extract_features(extractor, train_loader, device)

        # create sequence dataset from features
        from src.data.sequence_dataset import SequenceFromFeatures
        seq_ds = SequenceFromFeatures(feat_train, lab_train, seq_len=cfg['experiments']['review2']['sequence_len'])
        seq_loader = torch.utils.data.DataLoader(seq_ds, batch_size=cfg['batch_size'], shuffle=True)

        for rnn_type in ['LSTM', 'GRU', 'RNN']:
            for attn in [False]:
                model = SequenceModel(input_size=extractor.out_features, hidden_size=cfg['experiments']['review2']['hidden_size'], num_classes=len(classes), rnn_type=rnn_type, use_attention=attn).to(device)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
                for epoch in range(cfg['num_epochs']):
                    train_metrics = train_epoch(model, seq_loader, criterion, optimizer, device)
                    print(f"[rev2 {base_name} {rnn_type} attn={attn}] epoch {epoch+1} loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f}")

                # Save to both locations
                results_path = os.path.join(out_dir, f'{base_name}_{rnn_type}_attn_{attn}.pt')
                model_save_path = os.path.join(model_dir, f'{base_name.lower()}_{rnn_type.lower()}.pth')
                torch.save(model.state_dict(), results_path)
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Saved {rnn_type} model to {model_save_path}")

    print('✅ Review2 completed (feature + temporal model training).')


def train_review3(train_loader, val_loader, test_loader, classes, cfg, device):
    # Define directories
    out_dir = os.path.join(cfg['paths']['results_dir'], 'review3')
    model_dir = os.path.join(cfg['paths']['model_dir'], 'review3')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # autoencoder
    ae = ConvAutoencoder(latent_dim=cfg['experiments']['review3']['ae_latent_dim']).to(device)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(ae.parameters(), lr=cfg['learning_rate'])

    history = {'train_loss': []}
    for epoch in range(cfg['num_epochs']):
        ae.train(); losses=[]
        for x, _ in train_loader:
            x = x.to(device)
            optim.zero_grad()
            recon, _ = ae(x)
            loss = criterion(recon, x)
            loss.backward(); optim.step(); losses.append(loss.item())
        history['train_loss'].append(sum(losses)/len(losses))
        print(f'AE epoch {epoch+1} loss={history["train_loss"][-1]:.4f}')

    torch.save(ae.state_dict(), os.path.join(out_dir, 'autoencoder.pt'))
    torch.save(ae.state_dict(), os.path.join(model_dir, 'autoencoder.pth'))
    print(f"✅ Saved autoencoder to {model_dir}/autoencoder.pth")

    # save sample reconstructions
    ae.eval();
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon, _ = ae(x)
        import torchvision.utils as vutils
        vutils.save_image(x[:16], os.path.join(out_dir, 'ae_input.png'), normalize=True)
        vutils.save_image(recon[:16], os.path.join(out_dir, 'ae_recon.png'), normalize=True)

    # DCGAN
    gen = DCGANGenerator(z_dim=cfg['experiments']['review3']['gan_latent_dim']).to(device)
    disc = DCGANDiscriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    criterion_bce = torch.nn.BCELoss()

    for epoch in range(cfg['num_epochs']):
        for x, _ in train_loader:
            x = x.to(device)
            x_small = torch.nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            bs = x_small.size(0)
            real = torch.full((bs,), 0.9, device=device)
            fake = torch.full((bs,), 0.1, device=device)

            disc.zero_grad()
            out_real = disc(x_small)
            loss_real = criterion_bce(out_real, real)
            z = torch.randn(bs, cfg['experiments']['review3']['gan_latent_dim'], 1, 1, device=device)
            x_fake = gen(z)
            out_fake = disc(x_fake.detach())
            loss_fake = criterion_bce(out_fake, fake)
            loss_d = (loss_real + loss_fake) * 0.5
            loss_d.backward(); opt_d.step()

            gen.zero_grad()
            out_fake2 = disc(x_fake)
            loss_g = criterion_bce(out_fake2, real)
            loss_g.backward(); opt_g.step()

        print(f"DCGAN epoch {epoch+1}: d_loss={loss_d.item():.4f}, g_loss={loss_g.item():.4f}")

    torch.save(gen.state_dict(), os.path.join(out_dir, 'gan_generator.pt'))
    torch.save(gen.state_dict(), os.path.join(model_dir, 'gan_generator.pth'))
    torch.save(disc.state_dict(), os.path.join(out_dir, 'gan_discriminator.pt'))
    torch.save(disc.state_dict(), os.path.join(model_dir, 'gan_discriminator.pth'))
    print(f"✅ Saved GAN models to {model_dir}/")

    # generated samples
    with torch.no_grad():
        z = torch.randn(16, cfg['experiments']['review3']['gan_latent_dim'], 1, 1, device=device)
        fake_images = gen(z)
        import torchvision.utils as vutils
        vutils.save_image(fake_images, os.path.join(out_dir, 'gan_samples.png'), normalize=True)

    # latent visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    ae.eval();
    features = []
    labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, z = ae(x)
            features.append(z.cpu().numpy())
            labels.extend(y.numpy())
    features = np.vstack(features)

    pca_emb = PCA(n_components=2).fit_transform(features)
    n_samples = features.shape[0]
    tsne_perplexity = min(30, max(5, n_samples // 3))
    if tsne_perplexity >= n_samples:
        tsne_perplexity = max(2, n_samples - 1)

    tsne_emb = TSNE(n_components=2, random_state=cfg['seed'], perplexity=tsne_perplexity).fit_transform(features)

    import matplotlib.pyplot as plt
    for nm, emb in [('pca', pca_emb), ('tsne', tsne_emb)]:
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title='Classes', loc='best')
        plt.title(f'Latent {nm.upper()}')
        plt.savefig(os.path.join(out_dir, f'latent_{nm}.png'))
        plt.close()

    print('✅ Review3 completed.')


def train_review4(train_loader, val_loader, test_loader, classes, cfg, device):
    # Define directories
    out_dir = os.path.join(cfg['paths']['results_dir'], 'review4')
    model_dir = os.path.join(cfg['paths']['model_dir'], 'review4')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = CNNClassifier(num_classes=len(classes)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(cfg['num_epochs']):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = eval_model(model, val_loader, criterion, device)
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_metrics['acc'])
        print(f"[rev4] epoch {epoch+1} tr_loss={train_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

    test_metrics = eval_model(model, test_loader, criterion, device)
    print('✅ Final test metrics:', test_metrics)
    
    torch.save(model.state_dict(), os.path.join(out_dir, 'review4_model.pt'))
    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn.pth'))
    print(f"✅ Saved model to {model_dir}/cnn.pth")

    plot_loss(history, os.path.join(out_dir, 'review4_loss.png'), title='Review4 Loss')
    plot_accuracy(history, os.path.join(out_dir, 'review4_acc.png'), title='Review4 Accuracy')
    plot_confusion(test_metrics['confusion'], classes, os.path.join(out_dir, 'review4_confusion.png'))
    
    print('✅ Review4 completed')


def main(args):
    # Delete old models before training only if this is review 1
    if args.review == 1 and os.path.exists("outputs"):
        shutil.rmtree("outputs")
        print("🗑️ Deleted old outputs directory")

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['seed'] = cfg.get('seed', 42)
    cfg['batch_size'] = cfg.get('batch_size', 32)
    cfg['num_epochs'] = cfg.get('num_epochs', 10)
    cfg['learning_rate'] = cfg.get('learning_rate', 1e-4)
    cfg['use_gpu'] = cfg.get('use_gpu', True)

    set_seed_util(cfg['seed'])

    train_loader, val_loader, test_loader, classes = get_dataloaders(cfg)
    device = get_device()

    # Save classes to JSON for app to use (only if not exists)
    import json
    classes_path = os.path.join(cfg['paths']['model_dir'], 'classes.json')
    if not os.path.exists(classes_path):
        os.makedirs(os.path.dirname(classes_path), exist_ok=True)
        with open(classes_path, 'w') as f:
            json.dump(classes, f)
        print(f"💾 Saved classes to {classes_path}")

    print("Detected classes:", classes)
    print("Number of classes:", len(classes))

    print(f"\n{'='*60}")
    print(f"🚀 Starting Review {args.review} Training")
    print(f"📊 Classes: {classes}")
    print(f"💻 Device: {device}")
    print(f"{'='*60}\n")

    if args.review == 1:
        train_review1(train_loader, val_loader, test_loader, classes, cfg, device)
    elif args.review == 2:
        train_review2(train_loader, val_loader, test_loader, classes, cfg, device)
    elif args.review == 3:
        train_review3(train_loader, val_loader, test_loader, classes, cfg, device)
    elif args.review == 4:
        train_review4(train_loader, val_loader, test_loader, classes, cfg, device)
    else:
        print('Unknown review number')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DL Plant Disease System training entrypoint')
    parser.add_argument('--review', type=int, default=4, help='review number to run (1-4)')
    args = parser.parse_args()
    main(args)
