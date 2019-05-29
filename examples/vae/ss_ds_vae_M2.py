import argparse

import torch
import torch.nn as nn
import torch.distributions.constraints as constraints

from visdom import Visdom

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from utils.custom_mlp import MLP, Exp
from utils.mnist_cached import MNISTCached, mkdir_p, setup_data_loaders
from utils.vae_plots import mnist_test_tsne_ssvae, plot_conditional_samples_ssvae, plot_conditional_samples_ssvae_irt 


class SSDSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised, distant supervised variational auto-encoder on the MNIST image dataset

    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """
    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,),
                 config_enum=None, use_cuda=False, aux_loss_multiplier=None):

        super(SSDSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter

        # encoder for irt difficulty will look a lot like encoder_z
        self.encoder_diff = MLP([self.input_size] +
                             hidden_sizes + [[1, 1]],
                             activation=nn.Softplus,
                             output_activation=[None, Exp],
                             allow_broadcast=self.allow_broadcast,
                             use_cuda=self.use_cuda)

        self.encoder_y = MLP([self.input_size] + hidden_sizes + [[self.output_size, 1, 1]],
                             activation=nn.Softplus,
                             output_activation=[nn.Softmax, None, Exp],
                             allow_broadcast=self.allow_broadcast,
                             use_cuda=self.use_cuda)

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = MLP([self.input_size + self.output_size + 1] +
                             hidden_sizes + [[z_dim, z_dim]],
                             activation=nn.Softplus,
                             output_activation=[None, Exp],
                             allow_broadcast=self.allow_broadcast,
                             use_cuda=self.use_cuda)

        self.decoder = MLP([z_dim + self.output_size + 1] +
                           hidden_sizes + [self.input_size],
                           activation=nn.Softplus,
                           output_activation=nn.Sigmoid,
                           allow_broadcast=self.allow_broadcast,
                           use_cuda=self.use_cuda)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None, ds=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(b) = normal(0,1)              # item difficulty
        p(x|y,z,diff) = bernoulli(loc(y,z,diff))   # an image
        loc is given by a neural network  `decoder`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("data", batch_size):

            # sample the handwriting style from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y (which digit to write) is unsupervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (1.0 * self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # sample the item difficulty from the prior distribution
            # if the difficulty is supervised, observe it 
            diff_prior_loc = torch.zeros(batch_size, 1, **options) 
            diff_prior_scale = torch.ones(batch_size, 1, **options).fill_(1.e3)
            diffs = pyro.sample('diff', dist.Normal(diff_prior_loc, diff_prior_scale).to_event(1), obs=ds)  

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            #loc = self.decoder.forward([zs, ys, diffs])
            loc = self.decoder.forward([zs, ys, diffs])
            pyro.sample("x", dist.Bernoulli(loc).to_event(1), obs=xs)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        batch_size = xs.size(0)
        # sample difficulties first
        #print(xs.size(), loc_diff.size())  
        #diffs = pyro.sample('diff', dist.Normal(loc_diff, scale_diff)) 

        with pyro.plate("data", batch_size):

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            #loc_diff, scale_diff = self.encoder_diff.forward(xs)
            #print(loc_diff.size())
            #diffs = pyro.sample('diff', dist.Normal(loc_diff.unsqueeze(0), scale_diff.unsqueeze(0)))
            if ys is None:
                #print(diffs.size())
                alpha, loc_diff, scale_diff = self.encoder_y.forward(xs)
                #print(alpha.size())
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
                #print(ys.size()) 
            else:
                _, loc_diff, scale_diff = self.encoder_y.forward(xs) 

            diffs = pyro.sample('diff', dist.Normal(loc_diff, scale_diff).to_event(1))
            #print(diffs.size())
                
            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            #loc, scale = self.encoder_z.forward([xs, ys, diffs])
            loc, scale = self.encoder_z.forward([xs, ys, diffs])
            #print(loc.size()) 
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


    def model_irt(self, models, items, obs, xs):
        '''
        p(theta) = normal(0,1)          # model ability (distant supervision)
        p(c | b, theta) = link(theta - b)  # probability of correct 
        '''
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)
        num_models = len(set(models))
        num_items = len(set(items))
        #print('model', num_models, num_items) 
        options = dict(dtype=xs.dtype, device=xs.device)
        batch_size = xs.size(0)

        # vectorize
        models = torch.tensor(models, dtype=torch.long, device=xs.device)
        items = torch.tensor(items, dtype=torch.long, device=xs.device)
        obs = torch.tensor(obs, dtype=torch.float, device=xs.device)

        with pyro.plate("thetas"):
            ability = pyro.sample('theta', dist.Normal(torch.zeros(num_models, **options),
                torch.ones(num_models, **options)))
        with pyro.plate("diffs"):
            diff = pyro.sample('b', dist.Normal(torch.zeros(num_items, **options),
                torch.tensor(num_items, **options).fill_(1.e3)))

        with pyro.plate("data"):
            # for now, I need to make the (too-naive) assumption that everything is independent
            pyro.sample("obs", dist.Bernoulli(logits=ability[models] - diff[items]), obs=obs)
            #print('model', diff.size())
            #pyro.sample("obs", dist.Bernoulli(logits=ability - diff), obs=obs)
        
    def guide_irt(self, models, items, obs, xs):
        '''
        IRT guide:

        ''' 
        num_models = len(set(models))
        num_items = len(set(items))
        #print('guide', num_models, num_items) 
        options = dict(dtype=xs.dtype, device=xs.device)
        batch_size = xs.size(0)

        # vectorize
        models = torch.tensor(models, dtype=torch.long, device=xs.device)
        items = torch.tensor(items, dtype=torch.long, device=xs.device)
        obs = torch.tensor(obs, dtype=torch.float, device=xs.device)


        # register learnable params in the param store
        with pyro.plate("thetas"):
            m_theta_param = pyro.param("loc_ability", torch.zeros(num_models, **options))
            s_theta_param = pyro.param("scale_ability", torch.ones(num_models, **options),
                            constraint=constraints.positive)
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample("theta", dist_theta)

        # items 
        with pyro.plate("diffs"):
            irt_batch_size = 256
            loc_diffs_all, scale_diffs_all = [], []
            for i in range(0, len(xs), irt_batch_size):
                if len(xs[i:]) < irt_batch_size:
                    batch_xs = xs[i:]
                    _, loc_diffs, scale_diffs = self.encoder_y.forward(batch_xs)            
                    loc_diffs_all.extend(loc_diffs)
                    scale_diffs_all.extend(scale_diffs) 
                else:
                    # pick out the appropriate images from xs based on items idx
                    batch_xs = xs[i:i+irt_batch_size]
                    _, loc_diffs, scale_diffs = self.encoder_y.forward(batch_xs)            
                    loc_diffs_all.extend(loc_diffs)
                    scale_diffs_all.extend(scale_diffs)
            #_, loc_diffs, scale_diffs = self.encoder_y.forward(xs) 
            loc_diffs_all = torch.tensor(loc_diffs_all, **options) 
            scale_diffs_all = torch.tensor(scale_diffs_all, **options)            
            dist_b = dist.Normal(loc_diffs_all, scale_diffs_all)
            #print(dist_b.size())
            pyro.sample('b', dist_b)


    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        #diff, _ = self.encoder_diff.forward(xs)
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        #alpha = self.encoder_y.forward([xs, diff])
        alpha, loc_d, scale_d = self.encoder_y.forward(xs)
        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                #diff, _ = self.encoder_diff.forward(xs)
                #alpha = self.encoder_y.forward([xs, diff])
                alpha, loc_d, scale_d = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)
    print(num_losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    irt_batches = len(data_loaders['irt'])  # should only count as 1 
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses
    epoch_losses_irt = 0. 

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # RUN IRT batch here (outside of other logic)
    models, items, obs, xs = data_loaders['irt'] 
    #models = models[0:100]
    #items = items[0:100]
    #obs = obs[0:100] 
    #xs = xs[0:1000]


    irt_batch_size = 256
    epoch_losses_irt = losses[-1].step(models, items, obs, xs) 
    '''
    for i in range(0, len(obs), irt_batch_size):
        if len(models[i:]) < irt_batch_size:
            pass
            #batch_xs = xs[items[i:]]
            #epoch_losses_irt += losses[-1].step(models[i:], items[i:], obs[i:], batch_xs) 
        else:
            # pick out the appropriate images from xs based on items idx
            batch_xs = xs[items[i:i+irt_batch_size]]
            epoch_losses_irt += losses[-1].step(models[i:i+irt_batch_size], 
                    items[i:i+irt_batch_size], 
                    obs[i:i+irt_batch_size], 
                    batch_xs)
    '''

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys, idx) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys, idx) = next(unsup_iter)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses - 1):  # irt is last 1
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup, epoch_losses_irt 


def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    # use the appropriate data loader
    for (xs, ys, idx) in data_loader:
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs))
        actuals.append(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = torch.sum(pred[i] == act[i])
            accurate_preds += (v.item() == 10)

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    return accuracy


def visualize(ss_vae, viz, test_loader):
    if viz:
        # for a few difficulties, plot everything
        for diff in [-3., -2., -1., 0., 1., 2., 3.]:
            plot_conditional_samples_ssvae_irt(ss_vae, viz, diff)
        mnist_test_tsne_ssvae(ssvae=ss_vae, test_loader=test_loader)


def main(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    viz = None
    if args.visualize:
        #viz = Visdom()
        viz=True 
        mkdir_p("./vae_results")

    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = SSDSVAE(z_dim=args.z_dim,
                   hidden_layers=args.hidden_layers,
                   use_cuda=args.cuda,
                   config_enum=args.enum_discrete,
                   aux_loss_multiplier=args.aux_loss_multiplier)

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    guide = config_enumerate(ss_vae.guide, args.enum_discrete, expand=True)
    elbo = (JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO)(max_plate_nesting=1)
    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if args.aux_loss:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)
    
    # add irt loss 
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    loss_irt = SVI(ss_vae.model_irt, ss_vae.guide_irt, optimizer, loss=elbo) 
    losses.append(loss_irt) 

    try:
        # setup the logger if a filename is provided
        logger = open(args.logfile, "w") if args.logfile else None

        data_loaders = setup_data_loaders(MNISTCached, args.cuda, args.batch_size, sup_num=args.sup_num, irt_file=args.irt_file)

        # how often would a supervised batch be encountered during inference
        # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
        # until we have traversed through the all supervised batches
        periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * args.sup_num))

        # number of unsupervised examples
        unsup_num = MNISTCached.train_data_size - args.sup_num

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc, corresponding_test_acc = 0.0, 0.0

        # run inference for a certain number of epochs
        for i in range(0, args.num_epochs):

            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup, epoch_losses_irt = \
                run_inference_for_epoch(data_loaders, losses, periodic_interval_batches)

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = map(lambda v: v / args.sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

            str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))
            str_print += " irt loss: {}".format(epoch_losses_irt)

            validation_accuracy = get_accuracy(data_loaders["valid"], ss_vae.classifier, args.batch_size)
            str_print += " validation accuracy {}".format(validation_accuracy)

            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, args.batch_size)
            str_print += " test accuracy {}".format(test_accuracy)

            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy

            print_and_log(logger, str_print)

        final_test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, args.batch_size)
        print_and_log(logger, "best validation accuracy {} corresponding testing accuracy {} "
                      "last testing accuracy {}".format(best_valid_acc, corresponding_test_acc, final_test_accuracy))

        # save the model
        outfile = args.outfile 
        print_and_log(logger, 'saving model parameters to {}'.format(outfile)) 
        pyro.get_param_store().save(outfile) 
        
        # visualize the conditional samples
        visualize(ss_vae, viz, data_loaders["test"])

    finally:
        # close the logger file object if we opened it earlier
        if args.logfile:
            logger.close()


EXAMPLE_RUN = "example run: python ss_vae_M2.py --seed 0 --cuda -n 2 --aux-loss -alm 46 -enum parallel " \
              "-sup 3000 -zd 50 -hl 500 -lr 0.00042 -b1 0.95 -bs 200 -log ./tmp.log"

if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.3')

    parser = argparse.ArgumentParser(description="SS-VAE\n{}".format(EXAMPLE_RUN))

    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('--jit', action='store_true',
                        help="use PyTorch jit to speed up training")
    parser.add_argument('-n', '--num-epochs', default=50, type=int,
                        help="number of epochs to run")
    parser.add_argument('--aux-loss', action="store_true",
                        help="whether to use the auxiliary loss from NIPS 14 paper "
                             "(Kingma et al). It is not used by default ")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=46, type=float,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', default="parallel",
                        help="parallel, sequential or none. uses parallel enumeration by default")
    parser.add_argument('-sup', '--sup-num', default=3000,
                        type=float, help="supervised amount of the data i.e. "
                                         "how many of the images have supervised labels")
    parser.add_argument('-zd', '--z-dim', default=50, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[500], type=int,
                        help="a tuple (or list) of MLP layers to be used in the neural networks "
                             "representing the parameters of the distributions in our model")
    parser.add_argument('-lr', '--learning-rate', default=0.00042, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('-log', '--logfile', default="./tmp.log", type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--seed', default=None, type=int,
                        help="seed for controlling randomness in this example")
    parser.add_argument('--visualize', action="store_true",
                        help="use a visdom server to visualize the embeddings")
    parser.add_argument('--irt-file') 
    parser.add_argument('--outfile', help='where should we save the model after training?')
    args = parser.parse_args()

    # some assertions to make sure that batching math assumptions are met
    assert args.sup_num % args.batch_size == 0, "assuming simplicity of batching math"
    assert MNISTCached.validation_size % args.batch_size == 0, \
        "batch size should divide the number of validation examples"
    assert MNISTCached.train_data_size % args.batch_size == 0, \
        "batch size doesn't divide total number of training data examples"
    assert MNISTCached.test_size % args.batch_size == 0, "batch size should divide the number of test examples"

    main(args)
