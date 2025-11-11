import torch
import torch.nn as nn
import numpy as np
from model.UnetModel import UNet
from model.UnetModel_small import UNet_small
from model.FC_model import FC
from model.FCN import FCN8s, VGGNet
from model.NO import NO
import torch.optim as optim
import random
import numpy.linalg as LA
from data_process import normalization


def loss_compress_fun(mask, compress_r, image_size):
    C = image_size
    loss_binary = torch.pow(torch.norm(mask, p=1) / C - compress_r, 2)
    # loss_binary = torch.pow(torch.sum(mask) / C - compress_r, 2)
    return loss_binary


def loss_compress_L1(mask):
    return torch.norm(mask, p=1)



def cur_compress(mask):
    C = mask.shape[0] * mask.shape[1] * mask.shape[2]
    final_mask = binary(mask)
    # print("cur ", torch.max(final_mask), " ", torch.min(final_mask), " ")
    one_nums = torch.sum(final_mask)   
    return one_nums / C



def cur_compress_noise(noise):
    tmp = noise.clone()
    tmp[torch.abs(tmp) > 1e-6] = 1
    tmp[torch.abs(tmp) <= 1e-6] = 0
    return torch.sum(tmp) / (tmp.shape[0] * tmp.shape[1] * tmp.shape[2])



def cur_compress_L1(noise, image_size):
    noise_np = noise.numpy()
    noise_np = LA.norm(noise_np, axis=0)
    noise_np[np.abs(noise_np) > 1e-6] = 1
    noise_np[np.abs(noise_np) <= 1e-6] = 0
    one_nums = np.sum(noise_np)
    return one_nums / image_size


def binary(mask):
    return (torch.sign(mask - 0.5) + 1.0) / 2.0



def binary_percentage(mask, low=0.01, high=0.99):
    total_nums = mask.shape[0] * mask.shape[1] * mask.shape[2]
    zero_nums = np.sum(mask.numpy() < low)
    one_nums = np.sum(mask.numpy() > high)
    return (zero_nums + one_nums) / total_nums



def log_decrement(x):
    if x.data >= -1:
        return x
    else:
        return -torch.log(-x)-1



def adjust_learning_rate(optimizer, init_lr, epoch, epoch_num):
    lr = init_lr * (0.1 ** (epoch // epoch_num))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def pruning(classify_model, noises, images, labels, conf, pruner_mode="ours", target=True, compress_list=None):
    if pruner_mode == "ours":
        return pruning_ours(classify_model, noises, images, labels, conf, target)



def get_new_mask(mask, mask_h, mask_w, image_h, image_w):
    # print("{}, {}, {}, {}".format(mask_h, mask_w, image_h, image_w))
    if(mask_h != mask_w or image_h != image_w):
        print("not implement!")
        exit(0)
    small_h, small_w = image_h // mask_h, image_w // mask_w
    new_mask = mask.flatten().reshape(mask_h*mask_w, 1, 1)
    new_mask = new_mask.expand(
        mask_h*mask_w, small_h, small_w).reshape(mask_h, mask_w, small_h, small_w)
    new_mask = new_mask.permute(0, 2, 1, 3)
    new_mask = new_mask.reshape(image_h, image_w)
    return new_mask



def pruning_ours(classify_model, noises, images, labels, conf, target, loss_classify_mod="ce"):
    loss_classify_fun = nn.CrossEntropyLoss()
    alpha_s, alpha_e, epochs, iterations = conf.alpha_s, conf.alpha_e, conf.epochs, conf.iterations

    for index, (noise_, image_, label_) in enumerate(zip(noises, images, labels)):
        is_change = 0
        loss_lambda_gamma = conf.loss_lambda_gamma
        gamma_cnt = 0   
        while is_change == 0 and gamma_cnt < 2:
            # print("loss_lambda_gamma = ", loss_lambda_gamma)
            gamma_cnt = gamma_cnt + 1
            min_L0 = conf.image_size * conf.mask_c
            for init_count in range(conf.max_init_count):
                # print("{}/{}, init_count: {}".format(index+1, len(noises), init_count))
                if conf.pruner == "fc":
                    pruner = FC(in_c=3, in_h=conf.image_h, in_w=conf.image_w,
                                out_c=conf.mask_c, out_h=conf.mask_h, out_w=conf.mask_w)
                elif conf.pruner == "unet":
                    pruner = UNet(in_ch=3, out_ch=conf.mask_c,
                                  is_group=conf.is_group)
                elif conf.pruner == "unet_small":
                    pruner = UNet_small(
                        in_ch=3, out_ch=conf.mask_c, is_group=conf.is_group)
                elif conf.pruner == "fcn":
                    vgg_model = VGGNet(requires_grad=False)
                    pruner = FCN8s(pretrained_net=vgg_model,
                                   n_class=conf.mask_c)

                if torch.cuda.device_count() > 1:
                    device_ids = conf.gpu
                    device_ids = [int(x) for x in device_ids.split(",")]
                    pruner = torch.nn.DataParallel(
                        pruner, device_ids=device_ids)
                pruner = pruner.to(conf.device)
                pruner.train()

                # init_noise = noise.clone().detach()
                noise, image, label = noise_.clone().detach().to(conf.device), image_.clone(
                ).detach().to(conf.device), label_.clone().detach().to(conf.device)
                # labels_onehot = torch.zeros(1, conf.num_classes).to(conf.device).scatter(1, label.unsqueeze(0).view(-1, 1), 1)

                if target == False:
                    loss_classify_init = loss_classify_fun(classify_model(normalization((noise.detach(
                    ) + image.detach()).unsqueeze(0), conf.normalized_ops)), label.unsqueeze(0)).data
                    # print("loss_classify_init = ", loss_classify_init)

                init_lr = 0.01
                optimizer = optim.SGD(
                    pruner.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005)

                noise.requires_grad = True
                # sigmoid中的alpha
                alpha_step = (alpha_e - alpha_s) / (epochs * iterations)
                alpha = alpha_s

                cnt = 1
                for epoch in range(epochs):
                    # adjust_learning_rate(optimizer, init_lr, epoch, epochs)
                    for iteration in range(iterations):

                        mask = pruner(noise.unsqueeze(0), alpha).squeeze(0)

                        if conf.is_group == "0":
                            new_noise = noise * mask
                        else:
                            new_mask = get_new_mask(
                                mask, conf.mask_h, conf.mask_w, conf.image_h, conf.image_w)
                            new_noise = noise * new_mask

                        new_image = image + new_noise
                        outputs = classify_model(normalization(
                            new_image.unsqueeze(0), conf.normalized_ops))

                        if target == False:
                            loss_classify = loss_classify_fun(
                                outputs, label.unsqueeze(0))
                            loss_classify = -1/loss_classify_init * loss_classify + 1
                            gamma = 0.05
                            # loss_classify = torch.max(loss_classify, -1 + gamma * (loss_classify + 1))  
                            loss_classify = torch.max(
                                loss_classify, gamma * loss_classify)   
                        elif target == True:
                            pass

                        if epoch == epochs - 1:
                            if binary_percentage(mask.detach().cpu()) < 0.99:
                                alpha = alpha + 10 * alpha_step
                        else:
                            alpha += alpha_step

                        loss_compress = loss_compress_L1(mask)
                        loss_compress = loss_compress / \
                            (conf.mask_size * conf.mask_c)
                        if conf.sparsity != 0:
                            loss_compress = torch.pow(
                                loss_compress - conf.sparsity, 2)

                        compress_lambda = conf.C + loss_lambda_gamma * \
                            cur_compress(mask.detach())

                        loss = loss_classify + compress_lambda * loss_compress

                        '''
                        loss_list.append(loss.item())
                        loss_classify_list.append(loss_classify.item())
                        loss_compress_list.append(loss_compress.item())
                        cur_compress_list.append(cur_compress(mask.detach()))
                        '''

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            grad_now = noise.grad / \
                                torch.sum(torch.abs(noise.grad))
                            if epoch == 0 and iteration == 0:
                                grad = grad_now
                            else:
                                grad = grad * 0.9 + grad_now
                            # grad = noise.grad / torch.max(torch.abs(noise.grad))
                            with torch.no_grad():
                                noise[:] = noise - (2/255) * torch.sign(grad)
                                # noise[:] = noise - conf.eps * grad
                                noise[:] = (
                                    noise + image).clamp(conf.pixel_min, conf.pixel_max) - image
                                noise[:] = torch.clamp(
                                    noise, -conf.eps, conf.eps)
                        noise.grad.data.zero_()

                mask = pruner(noise.unsqueeze(0), alpha).squeeze(
                    0).detach().cpu()   
                if conf.is_group == "0":
                    new_mask = mask
                else:
                    new_mask = get_new_mask(
                        mask, conf.mask_h, conf.mask_w, conf.image_h, conf.image_w)

                binary_mask = binary(new_mask)
                tmp_noise = (noise.detach().cpu() *
                             binary_mask).unsqueeze(0).detach().cpu()
                tmp_image = (noise.detach().cpu() * binary_mask +
                             image.detach().cpu()).unsqueeze(0)

                if conf.mask_c == 3:
                    L0 = torch.sum(
                        torch.abs(tmp_image - image.detach().cpu()) > 1e-6)
                elif conf.mask_c == 1:
                    L0 = (image.shape[1] * image.shape[2]) - np.sum(np.all(
                        np.abs(tmp_image.squeeze(0).numpy() - image.detach().cpu().numpy()) <= 1e-6, 0))

                # print('L0 = ', L0.item(), 'min_l0 = ', min_L0)
                if L0.item() < min_L0:
                    output = classify_model(normalization(
                        tmp_image.to(conf.device), conf.normalized_ops))
                    _, pred = torch.max(output, 1)
                    # print("pre_label: ", pred.item()," ori_label: ", label.item())
                    if (pred.item() == label.item() and target == True) or (pred.item() != label.item() and target == False):
                        is_change = 1
                        min_L0 = L0.item()
                        best_noise = tmp_noise.clone().detach()
                        best_image = tmp_image.clone().detach()
                        best_mask = binary_mask.clone().detach()
                        '''
                        best_loss_list = loss_list[:]
                        best_loss_classify_list = loss_classify_list[:]
                        best_loss_compress_list = loss_compress_list[:]
                        best_cur_compress_list = cur_compress_list[:]
                        '''

            loss_lambda_gamma = loss_lambda_gamma / 10  

        best_noise = torch.zeros_like(
            tmp_noise) if is_change == 0 else best_noise
        best_image = image.detach().cpu().unsqueeze(0) if is_change == 0 else best_image
        best_mask = torch.zeros_like(
            binary_mask) if is_change == 0 else best_mask

        new_noises = best_noise if index == 0 else torch.cat(
            (new_noises, best_noise), 0)
        new_adv_images = best_image if index == 0 else torch.cat(
            (new_adv_images, best_image), 0)
        binary_masks = best_mask.unsqueeze(0) if index == 0 else torch.cat(
            (binary_masks, best_mask.unsqueeze(0)), 0)
        # compress_r_list.append(cur_compress(best_mask))
        '''
        losses_list.append(best_loss_list)
        losses_classify_list.append(best_loss_classify_list)
        losses_compress_list.append(best_loss_compress_list)
        cur_compresses_list.append(best_cur_compress_list)
        '''

    # change = {"losses": losses_list, "losses_class": losses_classify_list, "losses_compress": losses_compress_list, "compresses": cur_compresses_list}
    return new_noises, new_adv_images, binary_masks, None, None



def random_mask(l0, h, w, c):
    numbers = np.arange(c*h*w)
    random.shuffle(numbers)
    numbers = numbers[0: int(l0)]
    mask = np.zeros(c*h*w)
    mask[numbers] = 1
    mask = mask.reshape(c, h, w)
    return torch.FloatTensor(mask)
