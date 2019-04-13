from fastai.torch_core import *
from fastai.vision import *
from fastai.tabular.models import *
from fastai.tabular import *
from fastai.layers import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
import torch

__all__ = ['ImageTabularTextLearner', 'collate_mixed', 'image_tabular_text_learner', 'normalize_custom_funcs']

class ImageTabularTextModel(nn.Module):
    def __init__(self, emb_szs:ListSizes, n_cont:int, vocab_sz:int, encoder, use_trainer):
        super().__init__()
        self.use_trainer = use_trainer
        self.cnn = create_body(models.resnet34)
        nf = num_features_model(self.cnn) * 2
        drop = .5

        self.lm_encoder = SequentialRNN(encoder[0], PoolingLinearClassifier([400 * 3] + [32], [.4]))

        self.tab = TabularModel(emb_szs, n_cont, 128, [512, 256])

        self.reduce = nn.Sequential(*([AdaptiveConcatPool2d(), Flatten()] + bn_drop_lin(nf, 512, bn=True, p=drop, actn=nn.ReLU(inplace=True))))
        self.merge = nn.Sequential(*bn_drop_lin(512 + 128 + 32, 128, bn=True, p=drop, actn=nn.ReLU(inplace=True)))
        self.final = nn.Sequential(*bn_drop_lin(128, 1, bn=False, p=0., actn=None))

    def forward(self, img:Tensor, x:Tensor, text:Tensor) -> Tensor:
        imgCnn = self.cnn(img)
        imgLatent = self.reduce(imgCnn)
        tabLatent = self.tab(x[0], x[1])
        textLatent = self.lm_encoder(text)

        cat = torch.cat([imgLatent, F.relu(tabLatent), F.relu(textLatent[0])], dim=1)

        pred = self.final(self.merge(cat))
        pred = torch.sigmoid(pred) * 4 # making sure this is in the range 0-4

        if(not self.use_trainer):
            return pred
        else:
            return pred, textLatent
    
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def collate_mixed(samples, pad_idx:int=0):
    # Find max length of the text from the MixedItemList
    max_len = max([len(s[0].data[2]) for s in samples])

    for s in samples:
        res = np.zeros(max_len + pad_idx, dtype=np.int64)
        res[:len(s[0].data[2])] = s[0].data[2]
        s[0].data[2] = res

    return data_collate(samples)

def split_layers(model:nn.Module) -> List[nn.Module]:
    groups = [[model.cnn, model.lm_encoder]]
    groups += [[model.tab, model.reduce, model.merge, model.final]]
    return groups

class RNNTrainerCustom(RNNTrainer):
    def on_loss_begin(self, last_output:Tuple[Tensor,Tensor,Tensor], **kwargs):
        "Save the extra outputs for later and only returns the true output."
        self.raw_out,self.out = last_output[1][1],last_output[1][2]
        return {'last_output': last_output[0]}


def _normalize_images_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x[0].device),std.to(x[0].device)
    x[0] = normalize(x[0],mean,std)
    return x,y

def normalize_custom_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_images_batch, mean=mean, std=std),
            partial(denormalize, mean=mean, std=std))

class ImageTabularTextLearner(Learner):
    def __init__(self, data:DataBunch, model:nn.Module, use_trainer:bool=False, alpha:float=2., beta:float=1., **learn_kwargs):
        super().__init__(data, model, **learn_kwargs)
        if(use_trainer):
            self.callbacks.append(RNNTrainerCustom(self, alpha=alpha, beta=beta))
        self.split(split_layers)

def image_tabular_text_learner(data, len_cont_names, vocab_sz, data_lm, use_trainer:bool=False):
    l = text_classifier_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    l.load_encoder('fine_tuned_enc')

    emb = data.train_ds.x.item_lists[1].get_emb_szs()
    model = ImageTabularTextModel(emb, len_cont_names, vocab_sz, l.model, use_trainer)

    learn = ImageTabularTextLearner(data, model, use_trainer, metrics=[mae, rmse])
    return learn