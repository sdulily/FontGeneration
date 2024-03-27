"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss


def trainGAN_SUP(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_styconts = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    logger = additional['logger']

    # queue = additional['queue']

    # summary writer
    train_it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            source, reference, pos_imgs, neg_imgs, component_list, y_org = next(train_it)
        except:
            train_it = iter(data_loader)
            source, reference, pos_imgs, neg_imgs, component_list, y_org = next(train_it)

        comp_list = []
        for idx in range(len(component_list)):
            component_img = component_list[idx].cuda(args.gpu)
            comp_list.append(component_img)
        #
        # comp_batch = []
        # for idx in range(args.batch_size):
        #     comp_batch.append([comp[idx] for comp in comp_list][:num_comp[idx]])

        # comp_batch_1 = [i[0] for i in comp_list][:num_comp[0]]
        # comp_batch_2 = [i[1] for i in comp_list][:num_comp[1]]
        # print(len(lenth))
        x_org = source.cuda(args.gpu)
        # x_tf = reference[1]

        # x_ref_idx = torch.randperm(x_org.size(0))

        # x_org = x_org.cuda(args.gpu)
        positive = []
        for i in range(len(pos_imgs)):
            positive.append(pos_imgs[i].cuda(args.gpu))

        negative = []
        for i in range(len(neg_imgs)):
            negative.append(neg_imgs[i].cuda(args.gpu))
        y_org = y_org.cuda(args.gpu)
        # x_ref_idx = x_ref_idx.cuda(args.gpu)

        # x_ref = x_org.clone()
        x_ref = reference.cuda(args.gpu)

        #################
        # BEGIN Train C #
        #################
        training_mode = 'GANSUP'

        q_cont = C.moco(x_ref)

        B, _, _, _ = x_ref.shape
        queue = torch.zeros((B, 0, 512), dtype=torch.float).cuda(args.gpu)
        for i in range(len(negative)):
            n_cont = C_EMA.moco(negative[i]).unsqueeze(1)
            queue = torch.cat((queue, n_cont), dim=1)

        total_moco_loss = 0.0
        for i in range(len(positive)):
            k_cont = C_EMA.moco(positive[i])
            k_cont = k_cont.detach()

            moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
            total_moco_loss = total_moco_loss + moco_loss

        # k_cont = C_EMA.moco(x_tf)
        # k_cont = k_cont.detach()
        #
        # moco_loss = calc_contrastive_loss(args, q_cont, k_cont, queue)
        total_moco_loss = total_moco_loss / 5.0

        c_loss = 0.1 * total_moco_loss

        c_opt.zero_grad()
        c_loss.backward()
        if args.distributed:
            average_gradients(C)
        c_opt.step()
        ###############
        # END Train C #
        ###############

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            y_ref = y_org.clone()

            # y_ref = y_ref[x_ref_idx]
            s_ref = C.moco(x_ref)
            c_src = G.cnt_encoder(x_org, comp_list)
            x_fake = G.decode(c_src, s_ref)

        x_ref.requires_grad_()

        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        # 计算组件损失
        d_adv_com = 0.0
        # for index in range(len(com_list)):
        #     with torch.no_grad():
        #         com_src = G.cnt_encoder(com_list[index])
        #         com_fake = G.decode(com_src, s_ref)
        #     d_fake_com_logit, _ = D(com_fake.detach(), y_ref)
        #     d_adv_com_fake =calc_adv_loss(d_fake_com_logit, 'd_fake')
        #     d_adv_com = float(d_adv_com) + d_adv_com_fake / len(com_list)
        # d_adv_com_total = 0.0
        # for index in range(args.batch_size):
        #
        #     d_adv_com = 0.0
        #     for idx_1 in range(len(comp_batch[index])):
        #         # a = comp_batch[index]
        #         with torch.no_grad():
        #             com_src = G.cnt_encoder(comp_batch[index][idx_1].unsqueeze(0))
        #             com_ref = C.moco(x_ref[index].unsqueeze(0))
        #             com_fake = G.decode(com_src, com_ref)
        #         d_fake_com_logit, _ = D(com_fake.detach(), y_ref[index].unsqueeze(0))
        #         d_adv_com_fake = calc_adv_loss(d_fake_com_logit, 'd_fake')
        #         d_adv_com = float(d_adv_com) + d_adv_com_fake / len(comp_batch[index])
        #
        #     d_adv_com_total = float(d_adv_com_total) + d_adv_com / args.batch_size

        # d_adv_fake = d_adv_fake + d_adv_com_total

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        if args.distributed:
            average_gradients(D)
        d_opt.step()

        # Train G
        # s_src = C.moco(x_org)
        s_ref = C.moco(x_ref)

        c_src = G.cnt_encoder(x_org, comp_list)
        x_fake = G.decode(c_src, s_ref)
        # x_rec = G.decode(c_src, s_src)

        g_fake_logit, _ = D(x_fake, y_ref)
        # g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        # g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        # g_adv_com = 0.0
        # for idx in range(len(com_list)):
        #     # com_ref = C.moco(com_list[idx])
        #     com_src = G.cnt_encoder(com_list[idx])
        #     fake_component = G.decode(com_src, s_ref)
        #     # rec_component = G.decode(com_src, com_ref)
        #
        #     g_fake_com_logit, _ = D(fake_component, y_ref)
        #     # g_rec_com_logit, _ = D(rec_component, target)
        #     g_adv_com_fake = calc_adv_loss(g_fake_com_logit, 'g')
        #     # g_adv_com_rec = calc_adv_loss(g_rec_com_logit, 'g')
        #     g_adv_com = float(g_adv_com) + g_adv_com_fake / len(com_list)
        # g_adv_com_total = 0.0
        # for index_1 in range(args.batch_size):
        #
        #     g_adv_com = 0.0
        #     for idx_2 in range(len(comp_batch[index_1])):
        #         com_src = G.cnt_encoder(comp_batch[index_1][idx_2].unsqueeze(0))
        #         com_ref = C.moco(x_ref[index_1].unsqueeze(0))
        #         fake_component = G.decode(com_src, com_ref)
        #
        #         g_fake_com_logit, _ = D(fake_component, y_org[index_1].unsqueeze(0))
        #         g_adv_com_fake = calc_adv_loss(g_fake_com_logit, 'g')
        #         g_adv_com = float(g_adv_com) + g_adv_com_fake / len(comp_list[index_1])
        #
        #     g_adv_com_total = float(g_adv_com_total) + g_adv_com / args.batch_size


        g_adv = g_adv_fake

        # g_imgrec = calc_recon_loss(x_rec, x_org)
        #
        # s_fake = C.moco(x_fake)
        # s_ref_ema = C_EMA.moco(x_ref)
        #
        # queue_1 = torch.zeros((0, 512), dtype=torch.float).cuda(args.gpu)
        # for i in range(len(negative)):
        #     n_cont = C_EMA.moco(negative[i])
        #     queue_1 = torch.cat((queue_1, n_cont), dim=0)

        # g_sty_contrastive_total = 0.0
        # for i in range(len(negative)):
        #     s_neg_ema = C_EMA.moco(negative[i])
        #     g_sty_contrastive =calc_contrastive_loss(args, s_fake, s_ref_ema, s_neg_ema)
        #     g_sty_contrastive_total = g_sty_contrastive_total + g_sty_contrastive

        # g_sty_contrastive = calc_contrastive_loss(args, s_fake, s_ref_ema, queue_1)

        # g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_vec * g_sty_contrastive
        g_loss = args.w_adv * g_adv

        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        if args.distributed:
            average_gradients(G)
            average_gradients(C)
        c_opt.step()
        g_opt.step()

        ##################
        # END Train GANs #
        ##################

        # queue = queue_data(queue, k_cont)
        # queue = dequeue_data(queue)

        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                # g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                # g_styconts.update(g_sty_contrastive.item(), x_org.size(0))

            moco_losses.update(moco_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/STYCONT', g_styconts.avg, summary_step)

                add_logs(args, logger, 'C/MOCO', moco_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '
                      'C[{moco_losses.avg:.2f}]'.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses,
                                                        moco_losses=moco_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)

