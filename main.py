import os
import argparse
import logging
import time
import numpy as np
import load_data
from algorithms import san, sag, svrg, snm, vsn, sana, svrg2, gd, newton, sps, taps, sgd, adam, sps2, SP2L2p, spsdam, spsL1, SP2L1p, SP2maxp, SP2max, SP2
import utils
import pickle


# Press the green button in the gutter to run the script.

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store', dest='name',
                        help="name of setup")
    parser.add_argument('--save', action='store', type=bool, dest='save', default=False)
    parser.add_argument('--use_saved', action='store', type=bool, dest='use_saved', default=False)
    parser.add_argument('--type', action='store', dest='type', type=int, default=0,
                        help="type of problem, 0 means classification and 1 means regression.")
    parser.add_argument('--dataset', action='store', dest='data_set',
                        help="data set name")
    parser.add_argument('--data_path', action='store', dest='data_path',
                        help='path to load data')
    parser.add_argument('--result_folder', action='store', dest='folder',
                        help="folder path to store experiments results")
    parser.add_argument('--log_file', default='log.txt')
    parser.add_argument('--n_repetition', action='store', type=int, dest='n_rounds', default=10,
                        help="number of repetitions run for algorithm")
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=100)
    parser.add_argument('--reg_power_order', action='store', dest='reg_power_order', type=float, default=1.0,
                        help="can be chosen in 1,2,3. 1 means reg=1/sqrt{n}; 2 means 1/n; 3 means 1/n^2.")
    parser.add_argument('--loss', default="L2", help="loss function")
    parser.add_argument('--regularizer', default="L2", help="regularizer type")
    parser.add_argument('--scale_features', action='store', type=float, dest='scale_features', default=True)
    parser.add_argument('--reg', action='store', type=float, dest='reg', default=None)
    parser.add_argument('--lamb', action='store', type=float, dest='lamb', default=None)
    parser.add_argument('--lamb_schedule', action='store', default=False, dest='lamb_schedule',
                        help="name of the lamb scheduling")
    parser.add_argument('--line-search', action='store', default=False, dest='line_search',
                        help="option to use a line search")
    parser.add_argument('--b', action='store', type = int, default=1, dest='b',
                        help="batch size")
    parser.add_argument('--delta', action='store', type=float, dest='delta', default=None)
    parser.add_argument("--lr", action='store', type=float, dest='lr', default=1.0)
    parser.add_argument("--beta", action='store', type=float, dest='beta', default=0.0)
    parser.add_argument("--tol", action='store', type=float, dest='tol', default=None)
    parser.add_argument('--run_san', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sana', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sag', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sag_lin', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_msag', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_svrg2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_snm', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_vsn', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_gd', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_newton', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sps2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2L2p', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2L1p', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2maxp', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2max', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_SP2', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_spsdam', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_spsL1', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_sgd', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_dai', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_adam', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_taps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('--run_motaps', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    ## Parameters for TAPS and MOTAPS
    parser.add_argument("--tau", action='store', type=float, dest='tau', default=None)
    parser.add_argument("--tau_lr", action='store', type=float, dest='tau_lr', default=None)   
    parser.add_argument("--motaps_lr", action='store', type=float, dest='motaps_lr', default=None)
    opt = parser.parse_args()
    return opt


def build_problem(opt):
    folder_path = os.path.join(opt.folder, opt.data_set)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    logging.basicConfig(filename=os.path.join(folder_path, opt.log_file),
                        level=logging.INFO, format='%(message)s')
    logging.info(time.ctime(time.time()))
    logging.info(opt)

    # load data
    X, y = load_data.get_data(opt.data_path)
    X = X.toarray()  # convert from scipy sparse matrix to dense
    logging.info("Data Sparsity: {}".format(load_data.sparsity(X)))
    if opt.type == 1:
        problem_type = "classification"
    elif opt.type == 2:
        problem_type = "regression"
    else:
        problem_type = "unknown"

    criterion, penalty, X, y = load_data.load_problem(X, y, problem_type,
                loss_type = opt.loss, regularizer_type = opt.regularizer, 
                bias_term = True, scale_features = opt.scale_features)

    if opt.reg is None:
        n, d = X.shape
        # sig_min = np.min(np.diag(X@X.T))
        # reg = sig_min  / (n**(opt.reg_power_order))
        reg = 1.0  / (n**(opt.reg_power_order))
    else:
        reg = opt.reg
    logging.info("Regularization param: {}".format(reg))

    return folder_path, criterion, penalty, reg, X, y 

def set_minibatch_size(n,b):
    r'''Takes number of data points, and a given minibatch size b, 
    check if b is possible and sets alternative if not possible'''
    # if b> n/5:
    #     b = np.ceil(n/5)
    if b >= n:
        b=n
    return b

def run(opt, folder_path, criterion, penalty, reg, X, y):
    n, d = X.shape
    opt.b = int(set_minibatch_size(n,opt.b)) # set minibatch size
    logging.info("Number of data points: {:d}; Number of features: {:d}".format(n, d))
    epochs = opt.epochs
    n_rounds = opt.n_rounds
    x_0 = np.zeros(d)  # np.random.randn(d)
    s_0 = np.zeros(1)

    dict_grad_iter = {}
    dict_loss_iter = {}
    dict_time_iter = {}
    dict_stepsize_iter = {}
    dict_slack_iter = {}
    def collect_save_dictionaries(algo_name, output_dict):
        # "grad_iter" : grad_iter, "loss_iter" : loss_iter, "grad_time" : grad_time, "stepsizes" : stepsizes
        dict_grad_iter[algo_name] = output_dict['grad_iter']
        dict_loss_iter[algo_name] = output_dict['loss_iter']
        dict_time_iter[algo_name] = output_dict['grad_time']
        if "stepsizes" in output_dict:
            dict_stepsize_iter[algo_name] = output_dict['stepsizes']
        if "slack" in output_dict:
            dict_slack_iter[algo_name] = output_dict['slack']
        if opt.save:  
            utils.save(folder_path, algo_name, dict_grad_iter, dict_loss_iter, dict_time_iter)



    if opt.run_svrg2:
        np.random.seed(0)  # random seed to reproduce the experiments
        svrg2_lr = opt.lr  # 0.001*reg
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg2_lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        algo_name = "SVRG2"
        output_dict = utils.run_algorithm(algo_name=algo_name, algo=svrg2, algo_kwargs=kwargs, n_repeat=n_repetition)
        collect_save_dictionaries(algo_name, output_dict)
    elif opt.use_saved:
        algo_name = "SVRG2"
        grad_iter, loss_iter, grad_time = utils.load(folder_path, algo_name)
        if grad_iter:
            dict_grad_iter["SVRG2"] = grad_iter

    if opt.run_san:
        np.random.seed(0)  # random seed to reproduce the experiments
        is_uniform = True # #TODO: remove this option 
        if is_uniform: # This could be in the initialization of san instead of here?
            dist = None
        else:
            p_0 = 1. / (n + 1)
            logging.info("Probability p_0: {:}".format(p_0))
            dist = np.array([p_0] + [(1 - p_0) / n] * n)

        kwargs = {"loss": criterion, "data": X, "label": y, "lr": opt.lr, "reg": reg, "dist": dist, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        algo_name = "SAN"
        output_dict = utils.run_algorithm(algo_name=algo_name, algo=san, algo_kwargs=kwargs, n_repeat=n_repetition)
        collect_save_dictionaries(algo_name, output_dict)
    elif opt.use_saved:
        algo_name = "SAN"
        grad_iter, loss_iter, grad_time = utils.load(folder_path, algo_name)
        if grad_iter:
            dict_grad_iter["SAN"] = grad_iter

    if opt.run_sana:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        algo_name = "SANA"          
        output_dict = utils.run_algorithm(algo_name="SANA", algo=sana,
                                                   algo_kwargs=kwargs, n_repeat=n_repetition)
        collect_save_dictionaries(algo_name, output_dict)
    elif opt.use_saved:
        algo_name = "SANA" 
        grad_iter, loss_iter, grad_time = utils.load(folder_path, algo_name)
        if grad_iter:
            dict_grad_iter["SANA"] = grad_iter

    if opt.run_vsn:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "tol": opt.tol}
        algo_name ="VSN"
        output_dict = utils.run_algorithm(algo_name=algo_name, algo=vsn, algo_kwargs=kwargs, n_repeat=n_repetition)
        collect_save_dictionaries(algo_name, output_dict)
    elif opt.use_saved:
        algo_name = "VSN" 
        grad_iter, loss_iter, grad_time = utils.load(folder_path, algo_name)
        if grad_iter:
            dict_grad_iter["VSN"] = grad_iter

    
    if opt.run_snm:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "reg": reg, "epoch": epochs,
                  "x_0": x_0.copy(), "tol": opt.tol}
        output_dict = utils.run_algorithm(
            algo_name="SNM", algo=snm, algo_kwargs=kwargs, n_repeat=n_repetition)
        collect_save_dictionaries("SNM", output_dict)
    elif opt.use_saved:
        grad_iter, loss_iter, grad_time = utils.load(folder_path, "SNM")
        if grad_iter:
            dict_grad_iter["SNM"] = grad_iter

    if opt.run_gd:
        # 1/L, L is the smoothness constant
        if opt.loss == "L2" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_ridge(X, reg)
        elif opt.loss == "Logistic" and opt.regularizer == "L2":
            gd_lr = 1. / utils.lipschitz_logistic(X, reg)
        else:
            print("Warning!!! GD learning rate")
            gd_lr = 0.01
        logging.info("Learning rate used for Gradient descent: {:f}".format(gd_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": gd_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        grad_iter, loss_iter, grad_time = utils.run_algorithm(
            algo_name="GD", algo=gd, algo_kwargs=kwargs, n_repeat=n_rounds)
        output_dict = utils.run_algorithm(
            algo_name="GD", algo=gd, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("Newton", output_dict)

    if opt.run_newton:
        newton_lr = 1.0
        logging.info("Learning rate used for Newton method: {:f}".format(newton_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": newton_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol}
        output_dict = utils.run_algorithm(
            algo_name="Newton", algo=newton, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("Newton", output_dict)
    elif opt.use_saved:
        grad_iter, loss_iter, grad_time = utils.load(folder_path, "Newton")
        if grad_iter:
            dict_grad_iter["Newton"] = grad_iter

    if opt.run_sgd:
        np.random.seed(0)
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SGD learning rate")
            lr_max = 0.01
        else:
            lr_max = 1.0/(1.0*L_max)

          #lrs = 0.5*lr_max*np.ones(n_iters)

        lrs = lr_max*(1./np.sqrt(np.arange(1, n * epochs + 1)))
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SGD" 
        else:
            beta = opt.beta
            algo_name = "SGDM" + str(beta) 
        logging.info("Learning rate max used for SGD method: {:f}".format(lr_max))
        kwargs = {"loss": criterion, "data": X, "label": y, "lrs": lrs, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=sgd, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)

    if opt.run_sps:
        np.random.seed(0)
        sps_lr = 0.5
        eps = 0.000000001
        if opt.beta is None:
            beta = 0.5
            algo_name = "SP"
        else:
            beta = opt.beta
            algo_name = "SPM" + str(beta)

        if opt.lamb is None:
            sps_max = 10.0# Nico suggested 10 or 100. But 100 was too large for convex
        else:
            sps_max = opt.lamb
            algo_name = algo_name + "-l-" + str(sps_max)    
        logging.info("Learning rate used for SP method: {:f}".format(sps_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "eps": eps, "sps_max": sps_max, "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=sps, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)
    if opt.run_sps2:
        np.random.seed(0)
        sps2_lr =1.0
        eps=0.01
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "eps": eps,  "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=sps2, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)
    
    if opt.run_SP2L2p:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2L2$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2L2$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=SP2L2p, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)


    if opt.run_SP2L1p:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2L1$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2L1$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        output = utils.run_algorithm(
            algo_name=algo_name, algo=SP2L1p, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output)


    if opt.run_SP2maxp:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2max$^+$"
        else:
            beta = opt.beta
            algo_name = "SP2max$^+$M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        output = utils.run_algorithm(
            algo_name=algo_name, algo=SP2maxp, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output)


    if opt.run_SP2max:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2max"
        else:
            beta = opt.beta
            algo_name = "SP2maxM" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=SP2max, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)


    if opt.run_SP2:
        np.random.seed(0)
        sps2_lr =1.0

        if opt.lamb is None:
            lamb = 0.0
        else:
            lamb = opt.lamb
        if opt.beta == 0.0:
            beta = 0.0
            algo_name = "SP2"
        else:
            beta = opt.beta
            algo_name = "SP2M" + str(beta)
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sps2_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(),"s_0": s_0.copy(), "regularizer": penalty,
                  "tol": opt.tol, "lamb": lamb,  "beta": beta}
        output_dict = utils.run_algorithm(
            algo_name=algo_name, algo=SP2, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries(algo_name, output_dict)


    if opt.run_adam:
        np.random.seed(0)
        lr = 0.001  # default should be lr = 0.001, but works well with 0.005

        logging.info("Learning rate max used for ADAM method: {:f}".format(lr))
        # loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None,
        #  beta1 =0.9, beta2 =0.999, eps = 10**(-8.0), verbose = False
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, 
                  "tol": opt.tol}

        output_dict = utils.run_algorithm(
            algo_name="ADAM", algo=adam, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("ADAM", output_dict)

    if opt.run_sag:
        np.random.seed(0)  # random seed to reproduce the experiments
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        L = utils.compute_L(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SVRG learning rate")
            sag_lr = 0.01
        else:
            b= opt.b
            sag_lr = 1/L_max#1/(L* (n/b)*(b-1)/(n-1) + ((n-b)/(n-1))*Ln_max/b ) #opt.b*1.0/(4.0*L_max)
            sag_lr = sag_lr/16
        # in the SAG paper, the lr given by theory is 1/16L.
        # sag_lr = 0.25 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SAG: {:f}".format(sag_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sag_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol, "b" : opt.b}
        output_dict = utils.run_algorithm(
            algo_name="SAG", algo=sag, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("SAG", output_dict)
    # elif opt.use_saved:
    #     grad_iter, loss_iter, grad_time = utils.load(folder_path, "SAG")
    #     if grad_iter:
    #         dict_grad_iter["SAG"] = grad_iter

    if opt.run_sag_lin:
        np.random.seed(0)  # random seed to reproduce the experiments
        logging.info("Learning rate used for SAG: {:f}".format(sag_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": sag_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol, "b" : opt.b, "linesearch": True}
        output_dict = utils.run_algorithm(
            algo_name="SAG-l", algo=sag, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("SAG-l", output_dict)
    # else:
    #     grad_iter, loss_iter, grad_time = utils.load(folder_path, "SAG")
    #     if grad_iter:
    #         dict_grad_iter["SAG"] = grad_iter

    if opt.run_msag:
        np.random.seed(0)  # random seed to reproduce the experiments
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": 1, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol, "b" : opt.b, "MSAG" : True, "fstar" : True}
        output_dict = utils.run_algorithm(
            algo_name="MSAG", algo=sag, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("MSAG", output_dict)
    # else:
    #     grad_iter, loss_iter, grad_time = utils.load(folder_path, "SAG")
    #     if grad_iter:
    #         dict_grad_iter["SAG"] = grad_iter


    if opt.run_svrg:
        np.random.seed(0)  # random seed to reproduce the experiments
        L_max = utils.compute_L_max(X, reg, opt.loss,opt.regularizer)
        L = utils.compute_L(X, reg, opt.loss,opt.regularizer)
        if L_max ==None:
            print("Warning!!! SVRG learning rate")
            svrg_lr = 0.01
        else:
            b= opt.b
            svrg_lr =1/(L* (n/b)*(b-1)/(n-1) + 3*((n-b)/(n-1))*L_max/b ) #opt.b*1.0/(4.0*L_max)
            svrg_lr = svrg_lr/2
            # svrg_lr = opt.b/(2.0*L_max)

        # in the book "Convex Optimization: Algorithms and Complexity, Sébastien Bubeck",
        # the Theorem 6.5 indicates that the theory choice lr of SVRG should be 1/10L.
        # svrg_lr = 0.4 / (max_squared_sum + 4.0 * reg)  # theory lr
        logging.info("Learning rate used for SVRG: {:f}".format(svrg_lr))
        kwargs = {"loss": criterion, "data": X, "label": y, "lr": svrg_lr, "reg": reg,
                  "epoch": epochs, "x_0": x_0.copy(), "regularizer": penalty, "tol": opt.tol, "b" : opt.b}
        output_dict = utils.run_algorithm(
             algo_name="SVRG", algo=svrg, algo_kwargs=kwargs, n_repeat=n_rounds)
        collect_save_dictionaries("SVRG", output_dict)
    # else:
    #     grad_iter, loss_iter, grad_time = utils.load(folder_path, "SVRG")
    #     if grad_iter:
    #         dict_grad_iter["SVRG"] = grad_iter

## Final return of run()     
    return dict_grad_iter, dict_loss_iter, dict_time_iter, dict_stepsize_iter

if __name__ == '__main__': 
  
    opt = get_args()   #get options and parameters from parser
    folder_path, criterion, penalty, reg, X, y  = build_problem(opt)  #build the optimization problem
    dict_grad_iter, dict_loss_iter, dict_time_iter, dict_stepsize_iter  = run(opt, folder_path, criterion, penalty, reg, X, y)
    
    #Plot the training loss and gradient convergence
    utils.plot_iter(result_dict=dict_grad_iter, problem=opt.data_set, title = opt.name + "-grad" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, tol=opt.tol, yaxislabel=r"$\| \nabla f \|^2$")
    utils.plot_iter(result_dict=dict_loss_iter, problem=opt.data_set, title = opt.name + "-loss" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, tol=opt.tol, yaxislabel=r"$f(w^t)-f^*$")
    # utils.plot_iter(result_dict=dict_slack_iter, problem=opt.data_set, title = opt.name + "-slack" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, tol=opt.tol, yaxislabel=r"$min_i s_i^t$")
    # utils.plot_iter(result_dict=dict_loss_iter, problem=opt.data_set, title = opt.name + "-max-loss" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, yaxislabel=r"$\max_i f_i(w^t)$")
    utils.plot_iter(result_dict=dict_stepsize_iter, problem=opt.data_set, title = opt.name + "-stepsize" + "-reg-" + "{:.2e}".format(reg), save_path=folder_path, yaxislabel="step sizes")

      