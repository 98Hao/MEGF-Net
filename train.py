

def main():
    os.makedirs(f'./results/{traindataname}', exist_ok=True)
    os.makedirs(f'./runs/{traindataname}', exist_ok=True)
    model_save_dir = f'./results/{traindataname}'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    cudnn.benchmark = True

    train_dataset = RealESRGANDataset(traindata_folder, crop_size, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    max_iter = len(train_loader)
    print(max_iter)

    model = mymodel.MYMODEL()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones, 0.5)

    save_writer_path = f'./runs/{traindataname}'
    if not os.path.exists(save_writer_path):
        os.mkdir(save_writer_path)
    writer = SummaryWriter(save_writer_path)
    epoch_adder = Adder()
    iter_adder = Adder()

    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    best_psnr= -1

    print("Start training...")
    epoch_idx=epoch
    while epoch_idx <= epoch_max:
        model.train()
        epoch_timer.tic()
        iter_timer.tic()

        for iter_idx, (input_img, label_img) in enumerate(train_loader):
            input_img, label_img = input_img, label_img
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)

            total_loss = 0.0
            if wei1 != 0:
                total_loss += wei1 * l1_loss(pred_img, label_img)
            if wei2 != 0:
                total_loss += wei2 * ssim_loss(pred_img, label_img)
            if wei3 != 0:
                total_loss += wei3 * lpips_loss(pred_img, label_img)
            if wei4 != 0:
                total_loss += wei4 * color_angle_loss(pred_img, label_img)
            loss = total_loss

            loss.backward()
            optimizer.step()

            iter_adder(loss.item())
            epoch_adder(loss.item())

            if (iter_idx + 1) % (max_iter/2) == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_adder.average()))
                iter_timer.tic()
                iter_adder.reset()


        print("EPOCH: %02d\t Elapsed time: %4.2f Epoch Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_adder.average()))
        writer.add_scalar('Epoch Loss', epoch_adder.average(), epoch_idx)
        aaa = epoch_adder.average()
        epoch_adder.reset()

        scheduler.step()

        if epoch_idx % savenum == 0:
            save_name = os.path.join(model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        if numpy.isnan(aaa):
            print(f"Detected NaN in loss at epoch {epoch_idx}, restarting training from previous checkpoint...")

            closest_checkpoint_epoch = epoch_idx - 2

            checkpoint_path = os.path.join(model_save_dir, f'model_{closest_checkpoint_epoch}.pkl')
            print(f"Reloading checkpoint from epoch {closest_checkpoint_epoch}...")

            modelpara = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(modelpara['model'])
            optimizer.load_state_dict(modelpara['optimizer'])
            scheduler.load_state_dict(modelpara['scheduler'])

            epoch_idx = closest_checkpoint_epoch + 1
            continue

        if epoch_idx % savenum == 0:
            psnr_set, ssim_set = _valid(model, testdata_folder)
            print(
                '%03d epoch \t Average PSNR %.2f dB; \t Average SSIM %.4f;'
                % (epoch_idx,  psnr_set, ssim_set))
            writer.add_scalar('PSNR_test', psnr_set, epoch_idx)
            writer.add_scalar('SSIM_test', ssim_set, epoch_idx)

        epoch_idx += 1

    writer.close()

    event_acc = event_accumulator.EventAccumulator(save_writer_path)
    event_acc.Reload()


    epoch_Loss = event_acc.Scalars('Epoch Loss')
    psnr_set = event_acc.Scalars('PSNR_test')
    ssim_set = event_acc.Scalars('SSIM_test')

    epoch_idx = [x.step for x in psnr_set]

    data = {
        'epoch_idx': epoch_idx,
        'PSNR_test5': [x.value for x in psnr_set],
        'SSIM_test5': [x.value for x in ssim_set],
    }
    df = pd.DataFrame(data)

    epoch_idx_Loss = [x.step for x in epoch_Loss]

    data_Loss = {
        'epoch_idx': epoch_idx_Loss,
        'Epoch Loss': [x.value for x in epoch_Loss],
    }
    df_Loss = pd.DataFrame(data_Loss)

    excel_save_name = os.path.join(save_writer_path, 'training_results.xlsx')
    df.to_excel(excel_save_name, index=False)

    excel_save_name_Loss = os.path.join(save_writer_path, 'training_results_Loss.xlsx')
    df_Loss.to_excel(excel_save_name_Loss, index=False)




import os
import pandas as pd
import torch
from tensorboard.backend.event_processing import event_accumulator
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy
from dataset import RealESRGANDataset
from utils import Adder, Timer, check_lr
from valid import _valid
import time
from metrics.loss_utils import l1_loss, l2_loss, ssim_loss, lpips_loss, color_angle_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    from model import ours as mymodel

    split = 'train'
    traindataname = 'LOLv2_Synthetic'
    testdataname = 'LOLv2_Synthetic'

    traindata_folder = f'./dataset/{traindataname}'
    testdata_folder = f'./dataset/{testdataname}'
    crop_size = 256
    batch_size = 2
    epoch = 1
    epoch_max = 500
    learning_rate = 3e-4
    milestones = [0.5*epoch_max, 0.7*epoch_max, 0.9*epoch_max,0.95*epoch_max]
    wei1, wei2, wei3, wei4 = 1.0 / 4.86, 2.13 / 4.86, 1.73 / 4.86, 0
    savenum = 1
    main()

