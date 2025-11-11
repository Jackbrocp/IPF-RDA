import torch
import os
import argparse
import cv2

from data_process import load_data, tensor_to_image, save_pkl, load_pkl, save_train_process, get_target_labels, normalization, get_mask, inv_normalization
from attack import attack_batch, filter_correct, filter_adv
from distance import compute_distance
from pruner import pruning
from config import Config
from model.classify_model import initialize_model

if __name__ == "__main__":
    print("main_attack.py")
    parser = argparse.ArgumentParser(description='auto_adv_main_attck')
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="inception_v3, vgg19, resnet18, resnet50, flower_resnet50, flower_resnet50_baseaug")
    parser.add_argument("--data_name", type=str, default="cifar10",
                        help="cifar10, imagenet, cifar100, tiny_imagenet, flower")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--attacker", type=str, default="ours",
                        help="ours")
    args = parser.parse_args()

    device = torch.device(
        "cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    conf = Config(gpu=args.gpu, device=device, model_name=args.model_name,
                  data_name=args.data_name, attacker=args.attacker)
    conf.setup_seed()   

    model = initialize_model(
        args.model_name, conf.num_classes, conf.model_path, args.gpu)
    model.eval()
    print("model loaded!")

    dataloader = load_data(
        conf.data_dir, args.batch_size, conf, args.data_name)
    print("data loaded!")

    correct_cnt_clean, success_cnt, total_cnt = 0, 0, 0
    save_id = 0
    L2_mean = 0.0
    L1_mean = 0.0
    L_inf_mean = 0.0
    L0_total_mean = 0.0
    L0_pixel_mean = 0.0
    s = ""
    attack_time = 0

    # 检测需要保存的
    images_pkl, adv_images_pkl, target_label_pkl = None, None, None

    for index, (images, labels) in enumerate(dataloader):
        print("Attack Batch: {}".format(index))

        '''
        if index < 1000:
            continue
        '''

        total_cnt += len(images)
        correct_cnt_clean += len(images)

        # 无目标攻击
        # 进行批量的攻击
        noises, adv_images, attack_time_batch = attack_batch(
            model, images, labels, conf, args.attacker, target=False)
        attack_time += attack_time_batch
        # 判断是否攻击成功，计算success rate
        adv_images_filter, labels_filter, idx = filter_adv(
            model, adv_images, labels, conf)
        success_cnt += len(adv_images_filter)
        target_labels = labels  # 方便后面统一处理

        # 计算距离并保存为图像
        for i, (image, adv_image, target_label) in enumerate(zip(images, adv_images, target_labels)):
            # print(type(image))
            # 计算距离指标
            L2, L1, L_inf, L0_total, L0_pixel = compute_distance(
                image, adv_image, ["L2", "L1", "L_inf", "L0_total", "L0_pixel"], args.data_name)
            if idx[i] == 0:
                # print("fail!")
                s = s + "{}: success = {}, L2 = {:.3f}, L1 = {:.3f}, L_inf = {:.3f}, L0_total = {}, L0_pixel = {}\n".format(
                    save_id, 0, L2, L1, L_inf, L0_total, L0_pixel)
                # save_id += 1
                # continue
            elif idx[i] == 1:
                s = s + "{}: success = {}, L2 = {:.3f}, L1 = {:.3f}, L_inf = {:.3f}, L0_total = {}, L0_pixel = {}\n".format(
                    save_id, 1, L2, L1, L_inf, L0_total, L0_pixel)
                L2_mean += L2
                L1_mean += L1
                L_inf_mean += L_inf
                L0_total_mean += L0_total
                L0_pixel_mean += L0_pixel

            # 保存数字结果，检测或者迁移性都需要使用
            images_pkl = image.unsqueeze(0) if images_pkl is None else torch.cat(
                (images_pkl, image.unsqueeze(0)), 0)
            adv_images_pkl = adv_image.unsqueeze(0) if adv_images_pkl is None else torch.cat(
                (adv_images_pkl, adv_image.unsqueeze(0)), 0)
            target_label_pkl = target_label.unsqueeze(0) if target_label_pkl is None else torch.cat(
                (target_label_pkl, target_label.unsqueeze(0)), 0)

            # 保存
            mask_image = get_mask(image, adv_image, args.data_name)
            cv2.imwrite(os.path.join(conf.result_dir, str(
                save_id) + "_mask.png"), mask_image)
            image = tensor_to_image(image, args.data_name)
            adv_image = tensor_to_image(adv_image, args.data_name)
            image.save(os.path.join(conf.result_dir, str(save_id) + ".png"))
            adv_image.save(os.path.join(conf.result_dir,
                           str(save_id) + "_attack.png"))
            save_id += 1

    # 攻击一个样本所花费的平均时间
    speed = attack_time / correct_cnt_clean
    # 计算指标
    correct_cnt_adv = correct_cnt_clean - success_cnt
    L2_mean, L1_mean, L_inf_mean, L0_total_mean, L0_pixel_mean = L2_mean / success_cnt, L1_mean / \
        success_cnt, L_inf_mean / success_cnt, L0_total_mean / \
        success_cnt, L0_pixel_mean / success_cnt
    s += "Acc_clean = {:.3f}%, Acc_adv = {:.3f}%, Success_rate = {:.3f}%, L2_mean = {:.3f}, L1_mean = {:.3f}, L_inf_mean = {:.3f}, L0_total = {:.3f}, L0_pixel = {:.3f}, speed = {:.3f}s/it\n".format(
        correct_cnt_clean * 100 / total_cnt, correct_cnt_adv * 100 / total_cnt, success_cnt * 100 / correct_cnt_clean, L2_mean, L1_mean, L_inf_mean, L0_total_mean, L0_pixel_mean, speed)

    # 记录指标
    with open(os.path.join(conf.process_result_dir, "..", "attack_result.txt"), "w") as f:
        f.writelines(s)

    # 保存检测需要用到的
    save_pkl([images_pkl, adv_images_pkl, target_label_pkl], [
             "images", "adv_images", "labels"], conf.process_result_dir)
    print("done.")
