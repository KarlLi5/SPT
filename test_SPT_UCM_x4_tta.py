import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
from sewar.full_ref import scc, sam
from models.nets.SPT import SPT as net
from utils import utils_image as util


def tta(lr_images, net):
    "TTA based on _restoration_augment_tensor"

    hr_preds = []
    hr_preds.append(inference_onece(lr_images, net))
    hr_preds.append(inference_onece(lr_images.rot90(1, [2, 3]).flip([2]), net).flip([2]).rot90(3, [2, 3]))
    hr_preds.append(inference_onece(lr_images.flip([2]), net).flip([2]))
    hr_preds.append(inference_onece(lr_images.rot90(3, [2, 3]), net).rot90(1, [2, 3]))
    hr_preds.append(inference_onece(lr_images.rot90(2, [2, 3]).flip([2]), net).flip([2]).rot90(2, [2, 3]))
    hr_preds.append(inference_onece(lr_images.rot90(1, [2, 3]), net).rot90(3, [2, 3]))
    hr_preds.append(inference_onece(lr_images.rot90(2, [2, 3]), net).rot90(2, [2, 3]))
    hr_preds.append(inference_onece(lr_images.rot90(3, [2, 3]).flip([2]), net).flip([2]).rot90(1, [2, 3]))
    return torch.stack(hr_preds, dim=0).mean(dim=0)


def inference_onece(lr_images, net):
    outLF = net(lr_images)
    return outLF


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='classical_sr')

    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car

    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                                                            'Just used to differentiate two different '
                                                                            'settings in Table 2 of the paper. '
                                                                            'Images are NOT tested patch by patch.')

    parser.add_argument('--model_path', type=str,
                        default='/disk1/lwk/workspace/lwk_TGRS/weights/118236_E_500.pth')  # 测试权重地址

    parser.add_argument('--folder_lq', type=str, default='datasets/UCMerced/testsets/UCMerced/testLx4',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='datasets/UCMerced/testsets/UCMerced/testH',
                        help='input ground-truth test image folder')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        raise FileNotFoundError(f'Target file does not exist: {args.model_path}')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['scc'] = []
    test_results['sam'] = []
    psnr, ssim = 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()

            output = tta(img_lq, model)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            psnr = util.calculate_rgb_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            sig_scc = scc(output, img_gt)
            sig_sam = sam(output, img_gt)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            test_results['scc'].append(sig_scc)
            test_results['sam'].append(sig_sam)

            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}'.
                  format(idx, imgname, psnr, ssim))


        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_scc = sum(test_results['scc']) / len(test_results['scc'])
        ave_sam = sum(test_results['sam']) / len(test_results['sam'])
        print('\n{} \n-- Average PSNR/SSIM(RGB)/SCC/SAM: {:.2f} dB; {:.4f}; {:.4f}; {:.4f}'.format(save_dir, ave_psnr,
                                                                                                   ave_ssim, ave_scc,
                                                                                                   ave_sam))
        print(f'test: {args.model_path}')


def define_model(args):
    global model
    if args.task == 'classical_sr':
        model = net(upscale=4, img_size=(64, 64), patch_size=1, img_range=1., CPTB_num=6,
                    SPAL_num=3, embed_dim=96, num_head=6, mlp_ratio=2, upsampler='pixelshuffle')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    return model


def setup(args):
    # 001 classical image sr/ 002 lightwei ght image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'image_results/MBT_UCMx{args.scale}'
        folder = args.folder_gt
        border = args.scale

    return folder, save_dir, border


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        img_gt = cv2.cvtColor(img_gt, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        lq_path = f'{args.folder_lq}/{imgname}{imgext}'
        img_lq = cv2.imdecode(np.fromfile(lq_path, dtype=np.uint8), -1)
        img_lq = cv2.cvtColor(img_lq, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)

    return output


if __name__ == '__main__':
    cv2.setNumThreads(1)
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main()
