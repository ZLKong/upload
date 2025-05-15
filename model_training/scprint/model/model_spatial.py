# from scprint.base.base_model import BaseModel
import copy
import os
from functools import partial

# from galore_torch import GaLoreAdamW
from math import factorial
from typing import Dict, Optional, List, Union

import ipdb
import lightning as L
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.tuner.lr_finder import _LRCallback
from scipy.sparse import load_npz
from scprint.model.esm_layers import *
from torch import Tensor, nn, optim
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)

# from .linear_transformer import FastTransformerEncoderWrapper as FastTransformerEncoder
from . import decoders, encoders, loss, utils
from .flash_attn import FlashTransformerEncoder
from .loss import grad_reverse
from .utils import simple_masker

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


class scPrint(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        genes: list,
        organisms: list = ["NCBITaxon:9606"],
        precpt_gene_emb: Optional[str] = None,
        gene_pos_enc: Optional[list] = None,
        normalization: str = "sum",
        d_model: int = 512,
        nhead: int = 8,
        attn_bias: str = "none",
        d_hid: int = 512,
        # spatial omics modeling
        clip_model_type: str = "openai/clip-vit-base-patch32",
        combine_weight: float = 1,
        edge_dim: int = 12,
        nlayers: int = 6,
        # hierarchical modeling parameters
        region_transformer_layers: int = 2,
        tissue_transformer_layers: int = 2,
        region_clip_model_type: Optional[str] = None,
        tissue_clip_model_type: Optional[str] = None,
        # -----
        expr_encoder_layers: int = 2,
        layers_cls: list[int] = [],
        classes: Dict[str, int] = {},
        labels_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        dropout: float = 0.2,
        transformer: str = "fast",
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos"
        domain_spec_batchnorm: str = "None",
        n_input_bins: int = 0,
        num_batch_labels: int = 0,
        mvc_decoder: str = "None",
        pred_embedding: list[str] = [],
        cell_emb_style: str = "cls",
        freeze_embeddings: bool = True,
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        zinb: bool = True,
        lr: float = 0.0001,
        optim="adamW",  # TODEL
        weight_decay=0.01,  # TODEL
        **flash_attention_kwargs,
    ):
        """
        scPRINT transformer for single cell biology and the inference of Gene Regulatory networks

        Args:
            genes (list): List of gene names the model will work with.
            precpt_gene_emb (np.array, optional): Gene embeddings of size (len(genes), d_model). Should be in the same order as the genes. Defaults to None.
            gene_pos_enc (list, optional): Gene position encoding of the same size as genes. Provides a location value for each gene in genes. Defaults to None.
            d_model (int, optional): Dimension of the model. Defaults to 512.
            nhead (int, optional): Number of heads in the multihead attention models. Defaults to 8.
            d_hid (int, optional): Dimension of the feedforward network model. Defaults to 512.
            nlayers (int, optional): Number of layers in the transformer model. Defaults to 6.
            region_transformer_layers (int, optional): Number of transformer layers in the region transformer. Defaults to 2.
            tissue_transformer_layers (int, optional): Number of transformer layers in the tissue transformer. Defaults to 2.
            region_clip_model_type (str, optional): CLIP model for region-level images. Defaults to same as cell-level.
            tissue_clip_model_type (str, optional): CLIP model for tissue-level images. Defaults to same as cell-level.
            expr_encoder_layers (int, optional): Number of layers in the expression encoder. Defaults to 2.
            layers_cls (list[int], optional): List specifying the number of layers in the classifier. Defaults to [].
            classes (Dict[str, int], optional): Classes to predict with the number of classes for each. Defaults to {}.
            labels_hierarchy (Dict[str, Dict[int, list[int]]], optional): Class hierarchy for classes with hierarchical classes. Defaults to {}.
            dropout (float, optional): Dropout value. Defaults to 0.2.
            transformer (str, optional): Transformer type to use. One of "linear", "flash", "flashsparse", "scprint". Defaults to "fast".
            domain_spec_batchnorm (str, optional): Whether to apply domain-specific batch normalization. Defaults to "None".
            expr_emb_style (str, optional): Style of input embedding. One of "continuous", "binned_pos", "cont_pos". Defaults to "continuous".
            mvc_decoder (str, optional): Style of MVC decoder. One of "None", "inner product", "concat query", "sum query". Defaults to "None".
            pred_embedding (list[str], optional): List of classes to use for plotting embeddings. Defaults to [].
            cell_emb_style (str, optional): Style of cell embedding. One of "cls", "avg-pool", "w-pool". Defaults to "cls".
            freeze_embeddings (bool, optional): Whether to freeze the embeddings during training. Defaults to True.
            label_decoders (Optional[Dict[str, Dict[int, str]]], optional): Label decoders to use for plotting the UMAP during validations. Defaults to None.
            zinb (bool, optional): Whether to use Zero-Inflated Negative Binomial distribution. Defaults to True.
            lr (float, optional): Learning rate. Defaults to 0.0001.
            optim (str, optional): Optimizer type. Defaults to "adamW".
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.01.
            **flash_attention_kwargs (dict): Additional keyword arguments for the model. see @flashformer.py

        Notes:
            for other parameters of the model that are not part of its class definition, see @trainer.trainer.py

        Raises:
            ValueError: If the expr_emb_style is not one of "continuous", "binned_pos", "cont_pos".
        """
        super().__init__()

        # ipdb.set_trace()
        # training flags
        self.do_denoise = True
        self.noise = [0.6]
        self.do_cce = False
        self.cce_sim = 0.5
        self.cce_scale = 0.002
        self.do_ecs = False
        self.ecs_threshold = 0.3
        self.ecs_scale = 0.05
        self.do_mvc = False
        self.mvc_scale = 1.0
        self.class_embd_diss_scale = 0.2
        self.do_adv_cls = False
        self.adv_class_scale = 0.1
        self.do_cls = False
        self.mean_attn_tot = None
        self.mean_attn_tot_c = 0
        self.do_adv_batch = False
        self.run_full_forward = True
        self.class_scale = 0.4
        self.do_next_tp = False
        self.do_generate = False
        self.mask_ratio = []
        self.warmup_duration = 500
        self.weight_decay = 0.01
        self.optim = "adamW"
        self.fused_adam = False
        self.lr_reduce_patience = 1
        self.lr_reduce_factor = 0.6
        self.test_every = 1
        self.lr_reduce_monitor = "val_loss"
        self.name = ""
        self.lr = lr
        self.lrfinder_steps = 0
        self.doplot = True
        self.get_attention_layer = []
        self.embs = None
        self.pred_log_adata = True
        self.attn = utils.Attention(len(classes) + 2 + len(genes))
        self.predict_depth_mult = 3
        self.predict_mode = "none"
        self.keep_all_cls_pred = False
        # should be stored somehow
        self.d_model = d_model
        self.normalization = normalization
        self.organisms = organisms
        self.edge_dim = edge_dim
        self.attn_bias = attn_bias
        self.nlayers = nlayers
        self.gene_pos_enc = gene_pos_enc
        self.mvc_decoder = mvc_decoder
        self.domain_spec_batchnorm = domain_spec_batchnorm
        # need to store
        self.n_input_bins = n_input_bins
        self.transformer = transformer
        self.label_counts = classes
        self.classes = list(classes.keys())
        self.cell_emb_style = cell_emb_style
        self.label_decoders = label_decoders
        self.pred_embedding = pred_embedding
        # compute tensor for mat_labels_hierarchy
        self.mat_labels_hierarchy = {}
        self.labels_hierarchy = labels_hierarchy
        if "strict_loading" in flash_attention_kwargs:
            flash_attention_kwargs.pop("strict_loading")

        for k, v in labels_hierarchy.items():
            tens = torch.zeros((len(v), classes[k]))
            for k2, v2 in v.items():
                tens[k2 - classes[k], v2] = 1
            self.mat_labels_hierarchy[k] = tens.to(bool)
        self.expr_emb_style = expr_emb_style

        if self.expr_emb_style not in ["category", "continuous", "none"]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.genes = genes
        self.vocab = {i: n for i, n in enumerate(genes)}

        # encoder
        # gene encoder
        if precpt_gene_emb is not None:
            embeddings = pd.read_parquet(precpt_gene_emb).loc[self.genes]
            if len(embeddings) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(embeddings) < len(self.genes):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(embeddings))
            sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(
                torch.tensor(embeddings.values)
            )

            self.gene_encoder = encoders.GeneEncoder(
                len(self.vocab), d_model, weights=sembeddings, freeze=freeze_embeddings
            )
        else:
            self.gene_encoder = encoders.GeneEncoder(len(self.vocab), d_model)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if expr_emb_style in ["continuous", "full_pos"]:
            self.expr_encoder = encoders.ContinuousValueEncoder(
                d_model, dropout, layers=expr_encoder_layers
            )
        elif expr_emb_style == "binned_pos":
            assert n_input_bins > 0
            self.expr_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        else:
            self.expr_encoder = torch.nn.Identity()

        # Positional Encoding
        if self.gene_pos_enc is not None:
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )

        self.cell_embs_count = len(self.classes) + 2
        # Class Encoder
        # always have [base_cell_emb, time_embedding, depth_embedding] + any other class info
        # base cell embedding will store other cell specific information
        self.class_encoder = encoders.CategoryValueEncoder(
            self.cell_embs_count - 1, d_model
        )
        # self.time_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        self.depth_encoder = encoders.ContinuousValueEncoder(
            d_model, dropout, layers=expr_encoder_layers
        )

        # Transformer
        # Linear
        if transformer == "linear":
            # linear transformer using the fast transformer package
            # self.transformer = FastTransformerEncoder(
            #    d_model, nhead, d_hid, nlayers, dropout, "linear"
            # )
            raise NotImplementedError("Linear transformer is not implemented")
        # regular or flash
        else:
            self.transformer = FlashTransformerEncoder(
                d_model,
                nhead,
                nlayers,
                dropout=dropout,
                use_flash_attn=(transformer == "flash"),
                **flash_attention_kwargs,
            )

        ########## cell level CLIP model ########## 
        self.clip_model_type = clip_model_type
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_type)
        self.clip_model.requires_grad_(False)
        self.projection_layer = nn.Linear(
            self.clip_model.config.hidden_size, d_model
        )  # project from clip to ours
        self.fusion_layer = FusionLayer(d_model)
        self.combine_weight = combine_weight
        
        ########## region level CLIP model ########## 
        if region_clip_model_type is None:
            region_clip_model_type = clip_model_type
        print(f"loading region-level clip: {region_clip_model_type}")
        self.region_clip_model = CLIPVisionModel.from_pretrained(region_clip_model_type)
        self.region_clip_model.requires_grad_(False)
        self.region_projection_layer = nn.Linear(
            self.region_clip_model.config.hidden_size, d_model
        )

        ########## tissue level CLIP model ########## 
        if tissue_clip_model_type is None:
            tissue_clip_model_type = clip_model_type
        print(f"loading tissue-level clip: {tissue_clip_model_type}")
        self.tissue_clip_model = CLIPVisionModel.from_pretrained(tissue_clip_model_type)
        self.tissue_clip_model.requires_grad_(False)
        self.tissue_projection_layer = nn.Linear(
            self.tissue_clip_model.config.hidden_size, d_model
        )

        # Hierarchical transformers
        if region_transformer_layers > 0:
            self.region_transformer = FlashTransformerEncoder(
                d_model, nhead, region_transformer_layers, dropout, True, **flash_args
            )
        else:
            self.region_transformer = nn.Identity()
            
        if tissue_transformer_layers > 0:
            self.tissue_transformer = FlashTransformerEncoder(
                d_model, nhead, tissue_transformer_layers, dropout, True, **flash_args
            )
        else:
            self.tissue_transformer = nn.Identity()

        # Fusion layers for region and tissue
        self.region_fusion_layer = FusionLayer(d_model)
        self.tissue_fusion_layer = FusionLayer(d_model)
        
        # Hierarchical integration
        self.cell_to_region_projection = nn.Linear(d_model, d_model)
        self.region_to_tissue_projection = nn.Linear(d_model, d_model)
        self.hierarchical_fusion = FusionLayer(d_model)

        # Decoders
        # expression prediction decoder
        if expr_emb_style == "none":
            self.expr_decoder = torch.nn.Identity()
        elif zinb:
            self.expr_decoder = decoders.ExprDecoder(
                d_model, len(genes), d_hid, norm=self.normalization
            )
        else:
            self.expr_decoder = decoders.MSEExprDecoder(
                d_model, len(genes), d_hid, norm=self.normalization
            )

        # classifier decoder
        self.cls_decoders = nn.ModuleDict()
        for k, v in classes.items():
            cls_decoder_name = decoders.ClsDecoder.__name__
            self.cls_decoders[k] = getattr(decoders, cls_decoder_name)(
                d_model, v, layers_cls
            )

        # class adv decoder
        self.class_adv_decoders = nn.ModuleDict()
        for cls_name in self.classes:
            cls_decoder_name = decoders.ClsDecoder.__name__
            self.class_adv_decoders[cls_name] = getattr(decoders, cls_decoder_name)(
                d_model, classes[cls_name], layers_cls
            )

        # batch adv decoder
        # the batch adv decoder has the same size output as the number of datasets
        self.batch_adv_decoder = decoders.ClsDecoder(
            d_model, num_batch_labels, layers_cls
        )

        # MVC decoder
        if mvc_decoder != "None":
            # NOTE: "concat query" and "sum query" MVC decoder is only available with gene_embedding (token embedding from gene)
            # all mvcdecoder cannot be combined with scaling
            if mvc_decoder == "inner product":
                self.mvc_dec = decoders.MVCDecoder(d_model, d_model)
            elif mvc_decoder == "concat query":
                self.mvc_dec = decoders.MVCDecoder_ConcatQuery(d_model, d_model)
            elif mvc_decoder == "sum query":
                self.mvc_dec = decoders.MVCDecoder_SumQuery(d_model, d_model)
            else:
                raise ValueError(
                    f"Unknown mvc_decoder: {mvc_decoder}. Use inner product, concat query, sum query or None"
                )

        self.save_hyperparameters()

    def on_load_checkpoint(self, checkpoints):
        weights = {}

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
    ):
        """
        _encode given inputs to the model encode into embeddings.

        Args:
            @see self.forward()

        Returns:
            Tensor: the encoded data
        """
        batch_size = gene_pos.shape[0]
        max_len = gene_pos.shape[1]
        # embed all the genes
        gene_embed = self.gene_encoder(gene_pos)  # (batch, seq_len, embsize)

        # embed values
        value_embed = None
        if expression is not None:
            # Apply expression scaling with the normalization method
            if self.normalization == "sum":
                cell_scales = expression.sum(1).reshape(-1, 1)
                if (cell_scales == 0.0).sum() > 0:
                    expression = expression + 1e-6
                    cell_scales = expression.sum(1).reshape(-1, 1)
                expression = expression / cell_scales

            # Scale by depth
            if full_depth is not None:
                cell_scales = full_depth.reshape(-1, 1)
                if (cell_scales == 0.0).sum() > 0:
                    cell_scales = cell_scales + 1e-6
                expression = expression * cell_scales

            # Encode expression values
            value_embed = self.expr_encoder(expression, mask)

        # apply mask
        if mask is not None:
            if value_embed is not None:
                value_embed.masked_fill(mask.unsqueeze(-1), 0)
            gene_embed.masked_fill(mask.unsqueeze(-1), 0)

        # Positional encoding for genes
        # make new embeding with position encoding
        if self.gene_pos_enc is not None:
            gene_embed = gene_embed + self.pos_encoder(gene_pos)

        out = gene_embed
        # Add value embedding
        if value_embed is not None:
            out = out + value_embed

        # Prepend class embedding
        if cell_embs is None:
            base_clss_ids = (
                torch.tensor([0], device=out.device)
                .long()
                .repeat((batch_size, 1))
            )
            if timepoint is None:
                # tp class is 1 or 1 + n
                tp_cls_vec = (
                    torch.ones(batch_size, 1, device=out.device) * 1
                )
            else:
                # 1 + timepoint
                tp_cls_vec = timepoint.unsqueeze(1)
            # tp_cls_vec = tp_cls_vec.long()

            depth_cls_vec = expression.sum(1).reshape(-1, 1)
            cell_embs = torch.cat(
                [
                    self.class_encoder(base_clss_ids)
                    + self.depth_encoder(depth_cls_vec),
                    # self.time_encoder(tp_cls_vec.to(out.dtype)),
                    self.class_encoder(
                        (tp_cls_vec.to(int) % (self.cell_embs_count - 2) + 1).long()
                    ),
                ],
                dim=1,
            )  # (batch, 2, embsize)

        # Prepend cell embeddings for cls, time, etc
        # Seq = [base cell emb, time, ...(depth as factor), genes]
        out = torch.cat([cell_embs, out], dim=1)  # (batch, # classes + seq_len, embsize)

        return out

    def _decoder(
        self,
        transformer_output,
        depth_mult,
        get_gene_emb=False,
        do_sample=False,
        do_mvc=False,
        do_class=False,
    ):
        cell_emb = self.get_cell_embs(transformer_output)

        # decode for gene expression and return # here we have to define where the model outputs start
        start_seq = self.cell_embs_count  # start of raw outputs (can be genes or proteins)
        end_seq = transformer_output.shape[1]

        # get gene embedding matrix
        # option: get gene embedding
        gene_embedding = None
        if get_gene_emb or (self.mvc_decoder != "None" and do_mvc):
            gene_embedding = transformer_output[
                :, start_seq:end_seq, :
            ]  # (batch, seq_len, embsize)

        # Decode, use expdecoder for expression level
        output = {}
        if (not do_sample) and (not do_mvc) and (not do_class):
            if isinstance(self.expr_decoder, decoders.MSEExprDecoder):
                # just get MSE, not zinb
                output["expr_pred"], output["expr_pred_unnorm"] = self.expr_decoder(
                    transformer_output[:, start_seq:end_seq, :], depth_mult
                )
            else:
                # get zinb dist
                output["expr_pred"], output["expr_pred_unnorm"] = self.expr_decoder(
                    transformer_output[:, start_seq:end_seq, :], depth_mult
                )

        if do_class:
            cls_res = {}
            for k in self.cls_decoders:
                logits = self.cls_decoders[k](cell_emb)
                cls_res[k] = torch.log_softmax(logits, -1)
            output["cls_pred"] = cls_res
        if do_sample:
            output["expr_sample"], output["expr_sample_unnorm"] = self.expr_decoder(
                transformer_output[:, start_seq:end_seq, :], depth_mult, True
            )
        if do_mvc and self.mvc_decoder != "None":
            if not isinstance(gene_embedding, tuple):
                output["mvc_pred"] = self.mvc_dec(
                    transformer_output[:, :start_seq, :], gene_embedding
                )
            else:
                # Use one of the cached embeddings
                g_e = gene_embedding
                output["mvc_pred"] = self.mvc_dec(
                    transformer_output[:, :start_seq, :], g_e
                )
        if gene_embedding is not None:
            output["gene_emb"] = gene_embedding

        return output

    def _fuse(self, transformer_output, image, mask=None):
        # Image is shape (batch_size, 3, 224, 224)
        if image is None:
            return transformer_output
        clip_out = self.clip_model(image).last_hidden_state  # (batch, seq_len, clip_dim)
        proj_clip = self.projection_layer(clip_out)  # (batch, seq_len, d_model)
        # Fuse
        fused = self.fusion_layer(
            transformer_output, proj_clip, self.combine_weight
        )  # (batch, seq_len+cls, d_model)
        return fused
        
    def _process_region_image(self, region_image):
        if region_image is None:
            return None
        clip_out = self.region_clip_model(region_image).last_hidden_state
        return self.region_projection_layer(clip_out)
        
    def _process_tissue_image(self, tissue_image):
        if tissue_image is None:
            return None
        clip_out = self.tissue_clip_model(tissue_image).last_hidden_state
        return self.tissue_projection_layer(clip_out)

    def forward(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,  # (new_minibatch_of_nxt_cells,)
        get_gene_emb: bool = False,
        depth_mult: Optional[Tensor] = None,
        do_sample: bool = False,
        do_mvc: bool = False,
        do_class: bool = False,
        get_attention_layer: list = [],
        region_image: Optional[Tensor] = None,
        tissue_image: Optional[Tensor] = None,
    ):
        """
        forward also called on self(), a full forward pass on the model

        Args:
            gene_pos (Tensor): A tensor of shape (minibatch, seq_len)
                representing the genes used for each cell in the minibatch.
            expression (Tensor, optional): A tensor of shape (minibatch, seq_len)
                representing the expression levels of genes in the minibatch. Defaults to None.
            mask (Tensor, optional): A tensor of shape (minibatch, seq_len)
                used to mask certain elements in the sequence during the forward pass. Defaults to None.
            full_depth (Tensor, optional): A tensor of shape (minibatch,)
                representing the full depth of each sequence in the minibatch. Defaults to None.
            timepoint (Tensor, optional): A tensor of shape (minibatch,)
                representing the timepoint associated with each sequence in the minibatch. Defaults to None.
            get_gene_emb (bool, optional): A flag indicating whether to return the gene embeddings.
                If True, the gene embeddings are included in the output. Defaults to False.
            do_sample (bool, optional): A flag indicating whether to sample the expression levels.
                If True, the expression levels are sampled during the forward pass. Defaults to False.
            get_attention_layer (list, optional): A list indicating which attention layers to return.
                If not empty, the specified attention layers are included in the output. Defaults to [].

        Returns:
            dict of output Tensors: A dictionary containing the output tensors from the forward pass.
                The keys of the dictionary depend on the input flags (get_gene_emb, do_sample, get_attention_layer).
                at minima, the dictionary codntains the following:
                - "mean": the mean expression levels
                - "zero_logits": the logits for zero-inflated expression levels
                - "disp": the dispersion parameter
                - "cell_embs": the cell embeddings per class
                - "cell_emb": the main cell embedding
                - "cls_output": the output of the classifier
        """

        # import ipdb

        # ipdb.set_trace()

        # --- 1. Cell-level processing ---
        encoding = self._encoder(gene_pos, expression, mask, full_depth, timepoint)

        # Attention bias for flashattention
        bias = None
        if self.attn_bias == "edge":
            adj = torch.zeros(
                (
                    encoding.size(0),
                    encoding.size(1),
                    encoding.size(1),
                    self.edge_dim,
                ),
                device=encoding.device,
            )
            adj = adj.float()

            bias = adj

        # Main transformer
        transformer_output = self.transformer(
            encoding,
            return_qkv=get_attention_layer,
            bias=bias if self.attn_bias != "none" else None,
            bias_layer=list(range(self.nlayers - 1)),
        )
        qkvs = None
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output

        # Fuse to get unified per cell embedding (cell-gene fuse)
        cell_fused_output = self._fuse(transformer_output, image)
        
        ####### 2. Region-level processing #######
        
        # Get cell-level representation 
        cell_representation = self.get_cell_embs(cell_fused_output)
        
        region_features = self._process_region_image(region_image)
        
        # If we have region features, transform and fuse cell features
        if region_features is not None:
            # Project cell features to region space
            cell_region_features = self.cell_to_region_projection(cell_representation)
            
            # Process with region transformer
            region_output = self.region_transformer(region_features)
            
            # Fuse region features with cell-derived features
            region_fused = self.region_fusion_layer(
                region_output, cell_region_features.unsqueeze(1), 1.0
            )
            
            # Get region representation from CLS token
            region_representation = region_fused[:, 0, :]
        else:
            # If no region image, use cell features directly
            region_representation = cell_representation
        
        # --- 3. Tissue-level processing ---
        tissue_features = self._process_tissue_image(tissue_image)
        
        # If we have tissue features, transform and fuse region features
        if tissue_features is not None:
            # Project region features to tissue space
            region_tissue_features = self.region_to_tissue_projection(region_representation)
            
            # Process with tissue transformer
            tissue_output = self.tissue_transformer(tissue_features)
            
            # Fuse tissue features with region-derived features
            tissue_fused = self.tissue_fusion_layer(
                tissue_output, region_tissue_features.unsqueeze(1), 1.0
            )
            
            # Get tissue representation from CLS token
            tissue_representation = tissue_fused[:, 0, :]
        else:
            # If no tissue image, use region features directly
            tissue_representation = region_representation
        
        # --- 4. Hierarchical integration ---
        # Fuse cell-level output with hierarchical information
        hierarchical_rep = tissue_representation.unsqueeze(1).expand(-1, cell_fused_output.size(1), -1)
        final_output = self.hierarchical_fusion(cell_fused_output, hierarchical_rep, 0.5)

        # --- 5. Decoding ---
        depth_mult = expression.sum(1) if depth_mult is None else depth_mult
        decoder_output = self._decoder(
            final_output,
            depth_mult,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
        )
        
        # Store multi-level representations for potential future use
        decoder_output["cell_representation"] = cell_representation
        decoder_output["region_representation"] = region_representation
        decoder_output["tissue_representation"] = tissue_representation

        # Return results with attention if requested
        if qkvs is not None:
            return decoder_output, qkvs
        else:
            return decoder_output

    def configure_optimizers(self):
        """@see pl.LightningModule"""
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        # not working because of poor weight decay implem
        if self.optim == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "adamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "galore":
            raise NotImplementedError("Galore optimizer not implemented")
            # param_groups = [
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" not in k
            #        ]
            #    },
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" in k
            #        ],
            #        "rank": 128,
            #        "update_proj_gap": 200,
            #        "scale": 0.25,
            #        "proj_type": "std",
            #    },
            # ]
            # optimizer = GaLoreAdamW(param_groups, lr=self.hparams.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
            verbose=True,
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.lr_reduce_monitor,
        }
        self.lrfinder_steps = 0
        for val in self.trainer.callbacks:
            if type(val) is _LRCallback:
                self.lrfinder_steps = val.num_training
            if type(val) is LearningRateFinder:
                self.lrfinder_steps = val._num_training_steps
        return [optimizer], [lr_dict]

    def on_fit_start(self):
        """@see pl.LightningModule"""
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx,
    ):
        """
        training_step defines the train loop. It is independent of forward

        @see pl.LightningModule

        Returns:
            _type_: _description_
        """
        # TASK 1 & 2 & 3 (first pass, expression reconstruction, label prediction)
        total_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_sim=self.cce_sim,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log_dict(losses, prog_bar=True, sync_dist=True)
        return total_loss

    def _full_training(
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],
        do_next_tp: bool = False,
        do_cce: bool = False,
        cce_sim: float = 0.5,
        do_ecs: bool = False,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_adv_batch: bool = False,
        do_cls: bool = False,
        do_generate: bool = False,
        run_full_forward: bool = True,
        mask_ratio: list[float] = [0.15],
    ):
        """
        _full_training implement the trainng steps: forward (multiple sometimes), loss

        Args:
            batch (dict[Tensors]): A dictionary containing tensors for the training batch:
                - "x": the expression levels of genes in the minibatch
                - "genes": the genes used for each cell in the minibatch
                - "class": the class to predict for each cell
                - "depth": the full depth of each cell in the minibatch
            do_denoise (bool, optional): A flag to indicate whether to perform denoising. Defaults to False.
            noise (list[float], optional): A list of noise levels to be used in denoising. Defaults to [].
            do_next_tp (bool, optional): A flag to indicate whether to perform next time point prediction. Defaults to False.
            do_cce (bool, optional): A flag to indicate whether to perform cross-categorical entropy. Defaults to False.
            cce_sim (float, optional): The similarity threshold for cross-categorical entropy. Defaults to 0.5.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity. Defaults to False.
            do_mvc (bool, optional): A flag to indicate whether to perform multi-view coding. Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification. Defaults to False.
            do_generate (bool, optional): A flag to indicate whether to perform data generation. Defaults to False.
            mask_ratio (list, optional): A list of mask ratios to be used in the training. Defaults to [0.15].

        Returns:
            loss, losses: the total loss as float and the individual losses as dict
        """
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]

        expression = batch["x"]
        gene_pos = batch["genes"]
        total_count = batch["depth"]
        clss = batch.get("class", None)
        batch_idx = batch.get("dataset", None)
        image = batch.get("image", None)

        total_loss = 0
        losses = {}
        cell_embs = []
        if run_full_forward:
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                mask=None,
                full_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            output.pop("disp")
            output.pop("zero_logits")
            output.pop("mean")
            l, tot = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            cell_embs.append(output["cell_emb"].clone())
            full_cell_embs = output["cell_embs"].clone()
            total_loss += tot
            losses.update({"full_forward_" + k: v for k, v in l.items()})
            do_mvc = False if do_mvc else do_mvc
            do_cls = False if do_cls else do_cls

        for i in mask_ratio:
            mask = simple_masker(
                shape=gene_pos.shape,
                mask_ratio=i,
            ).to(gene_pos.device)
            output = self.forward(
                gene_pos,
                expression=expression,
                image=image,
                mask=mask,
                full_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            l, tot = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            # we only want to do them once
            do_mvc = False if do_mvc else do_mvc
            do_cls = False if do_cls else do_cls

            cell_embs.append(output["cell_emb"].clone())
            total_loss += tot
            losses.update(
                {"mask_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
            )
        # TASK 3. denoising
        if do_denoise:
            for i in noise:
                expr = utils.downsample_profile(expression, dropout=i)
                output = self.forward(
                    gene_pos,
                    expression=expr,
                    image=image,
                    mask=None,
                    depth_mult=expression.sum(1),
                    full_depth=total_count,
                    do_mvc=do_mvc,
                    do_class=do_cls,
                )
                l, tot = self._compute_loss(
                    output,
                    expression,
                    clss,
                    batch_idx,
                    do_ecs,
                    do_adv_cls & do_cls,
                    do_adv_batch & do_cls,
                )
                do_mvc = False if do_mvc else do_mvc
                do_cls = False if do_cls else do_cls

                cell_embs.append(output["cell_emb"].clone())
                total_loss += tot
                losses.update(
                    {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
                )
                # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 6. expression generation
        if do_generate:
            output = self._generate(
                output["cell_embs"] if not run_full_forward else full_cell_embs,
                gene_pos,
                depth_mult=expression.sum(1),
                image=image,
                full_depth=None,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            cell_embs.append(output["cell_emb"].clone())
            l, tloss = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss

        # TASK 7. next time point prediction
        if do_next_tp:
            pass

        # TASK 4. contrastive cell embedding
        if do_cce:
            loss_cce = 0
            for i, cell_emb1 in enumerate(cell_embs[:-1]):
                for cell_emb2 in cell_embs[(i + 1) :]:
                    loss_cce += loss.similarity(
                        cell_emb1, cell_emb2, cce_sim
                    )  # (nlabels, minibatch, minibatch)
            fact = factorial(len(cell_embs))
            total_loss += loss_cce * self.cce_scale / fact
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": loss_cce / fact})

        # TASK 8. KO profile prediction
        # if we have that information
        # TASK 9. PDgrapher-drug-like perturbation prediction (L1000?)
        return total_loss, losses

    def _compute_loss(
        self,
        output,
        expression,
        clss,
        batch_idx,
        do_ecs=False,
        do_adv_cls=False,
        do_adv_batch=False,
        do_mse=0,
    ):
        """
        _compute_loss compute the loss of the model given output from the forward pass

        Args:
            output (dict): A dictionary containing the output of the forward pass.
            expression (Tensor): A tensor containing the expression levels of genes.
            mask (Tensor): A tensor indicating the masked positions in the input data.
            clss (Tensor): A tensor containing the class classes for each cell.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity.
                Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification.
                Defaults to False.
            do_mse (float, optional): A scaling factor to indicate whether and how much to weight mean
            squared error loss in addition to zinb loss.
                Defaults to 0.

        Raises:
            ValueError: Raised when an invalid operation or input is encountered.

        Returns:
            tuple: A tuple containing the total loss as a float and the individual losses as a dictionary.
        """
        total_loss = 0
        losses = {}
        # TASK 1. reconstruct masked expression
        if "zero_logits" in output:
            loss_expr = loss.zinb(
                theta=output["disp"],
                pi=output["zero_logits"],
                mu=output["mean"],
                target=expression,
            )
            if do_mse:
                loss_expr += loss.mse(
                    input=torch.log(output["mean"] + 1)
                    * (1 - torch.sigmoid(output["zero_logits"])),
                    target=torch.log(expression + 1),
                )
        elif "disp" in output:
            loss_expr = loss.nb(
                theta=output["disp"],
                mu=output["mean"],
                target=expression,
            )
        elif "mean" in output:
            loss_expr = loss.mse(
                input=output["mean"],
                target=expression,
            )
        else:
            loss_expr = 0
        total_loss += loss_expr
        losses.update({"expr": loss_expr})

        # TASK 2. predict classes
        if len(self.classes) > 0:
            ## Calculate pairwise cosine similarity for the embeddings
            cos_sim_matrix = (
                torch.nn.functional.cosine_similarity(
                    output["cell_embs"].unsqueeze(2),
                    output["cell_embs"].unsqueeze(1),
                    dim=3,
                )
                .abs()
                .mean(0)
            )
            ## Since we want to maximize dissimilarity, we minimize the negative of the average cosine similarity
            ## We subtract from 1 to ensure positive values, and take the mean off-diagonal (i != j)
            loss_class_emb_diss = cos_sim_matrix.fill_diagonal_(0).mean()
            ## Apply the custom dissimilarity loss to the cell embeddings
            losses.update({"class_emb_sim": loss_class_emb_diss})
            total_loss += self.class_embd_diss_scale * loss_class_emb_diss
            ## compute class loss
            loss_cls = 0
            loss_adv_cls = 0
            for j, clsname in enumerate(self.classes):
                if "cls_output_" + clsname not in output:
                    continue
                # if clsname == "organism_ontology_term_id":
                #     continue
                # setting the classes from index to one hot
                loss_cls += loss.classification(
                    clsname,
                    pred=output["cls_output_" + clsname],
                    cl=clss[:, j],
                    maxsize=self.label_counts[clsname],
                    labels_hierarchy=self.mat_labels_hierarchy,
                )
            total_loss += self.class_scale * loss_cls
            if loss_cls != 0:
                losses.update({"cls": loss_cls})
            # TASK 2bis. adversarial label prediction
            if do_adv_cls:
                embs = output["cell_embs"][:, 2:, :].clone()
                for j, adv_cls in enumerate(self.classes):
                    ind = torch.arange(len(self.classes))
                    mean_embs = torch.mean(embs[:, ind != j, :], dim=1)
                    mean_embs = grad_reverse(mean_embs, lambd=1.0)
                    adv_pred = self.cls_decoders[adv_cls](mean_embs)
                    loss_adv_cls += loss.classification(
                        adv_cls,
                        pred=adv_pred,
                        cl=clss[:, j],
                        maxsize=self.label_counts[adv_cls],
                        labels_hierarchy=self.mat_labels_hierarchy,
                    )

                total_loss += self.adv_class_scale * loss_adv_cls
                losses.update({"adv_cls": loss_adv_cls})

        if (
            do_adv_batch
            and self.grad_reverse_discriminator_loss is not None
            and batch_idx is not None
        ):
            mean_emb = torch.mean(output["cell_embs"][:, 2:, :].clone(), dim=1)
            loss_adv = self.grad_reverse_discriminator_loss(mean_emb, batch_idx)
            total_loss += loss_adv * self.class_scale / 16
            losses.update({"adv_batch": loss_adv})
        # TASK 2ter. cell KO effect prediction
        # (just use a novel class, cell state and predict if cell death or not from it)
        # add large timepoint and set the KO gene to a KO embedding instead of expression embedding
        # TODO: try to require the gene id to still be predictable (with weight tying)
        if "mvc_disp" in output:
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        # TASK 5. elastic cell similarity
        if do_ecs:
            loss_ecs = loss.ecs(output["cell_emb"], ecs_threshold=self.ecs_threshold)
            total_loss += self.ecs_scale * loss_ecs
            losses.update({"ecs": loss_ecs})
        # for key, value in losses.items():
        #     # if contains nan, raise error
        #     if torch.isnan(value).any():
        #         ipdb.set_trace()
        return losses, total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """@see pl.LightningModule"""
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        # making sure that we don't do this during lrfinder
        # @mf: stop update the lr for each epoch
        # It seems this will not increase the lr
        for i, pg in enumerate(optimizer.param_groups):
            if (
                self.global_step < self.warmup_duration + self.lrfinder_steps
            ) and self.lrfinder_steps < self.global_step:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_duration)
                pg["lr"] = lr_scale * self.hparams.lr
        for i, pg in enumerate(optimizer.param_groups):
            # if pg["lr"] < 2e-5:
            #    pg["lr"] = 2e-5
            self.log("lr_" + str(i), pg["lr"])

    def on_validation_start(self):
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        self.embs = None
        self.counter = 0

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        """
        validation_step defines the validation loop. It is independent of forward
        @see pl.LightningModule

        Args:
            batch (list[Tensor]): @see training_step
        """
        val_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_sim=self.cce_sim,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        image = batch.get("image", None)
        region_image = batch.get("region_image", None)
        tissue_image = batch.get("tissue_image", None)
        
        # TODO: make this faster by only calling val loss
        if self.embs is not None:
            if self.embs.shape[0] < 100_000:
                self.info = torch.cat([self.info, batch["class"]])
                self._predict(
                    gene_pos,
                    expression,
                    depth,
                    image=image,
                    region_image=region_image,
                    tissue_image=tissue_image,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=300_000,
                )
        else:
            self.info = batch["class"]
            self._predict(
                gene_pos,
                expression,
                depth,
                image=image,
                region_image=region_image,
                tissue_image=tissue_image,
                pred_embedding=self.pred_embedding,
                max_size_in_mem=300_000,
            )
        self.log("val_loss", val_loss, sync_dist=True)
        self.log_dict(losses, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        """@see pl.LightningModule"""
        self.embs = self.all_gather(self.embs).view(-1, self.embs.shape[-1])
        self.info = self.all_gather(self.info).view(-1, self.info.shape[-1])
        self.pred = (
            self.all_gather(self.pred).view(-1, self.pred.shape[-1])
            if self.pred is not None
            else None
        )
        self.pos = self.all_gather(self.pos).view(-1, self.pos.shape[-1])
        if not self.trainer.is_global_zero:
            # print("you are not on the main node. cancelling logging step")
            return
        if self.trainer.state.stage != "sanity_check":
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics["val_loss"])
            # run the test function on specific dataset
            self.log_adata(
                gtclass=self.info, name="validation_part_" + str(self.counter)
            )
            if (self.current_epoch + 1) % self.test_every == 0:
                self.on_test_epoch_end()

    def test_step(self, *args, **kwargs):
        print("step")
        pass

    def on_test_epoch_end(self):
        print("start test")
        model_copy = copy.deepcopy(self)
        name = self.name + "_step" + str(self.global_step)
        metrics = utils.test(model_copy, name, filedir=FILEDIR)
        print(metrics)
        print("done test")
        self.log_dict(metrics, sync_dist=True, rank_zero_only=True)

    def on_predict_epoch_start(self):
        """@see pl.LightningModule"""
        self.embs = None
        self.attn.data = None
        self.attn.attn = None
        self.counter = 0
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(self, batch, batch_idx):
        """
        embed given gene expression, encode the gene embedding and cell embedding.

        Args:
            batch @see training_step

        Returns:
            Tensor: _description_
        """
        return self._predict(
            batch["genes"],
            batch["x"],
            batch["depth"],
            batch["image"],
            self.predict_mode,
            self.pred_embedding,
            self.get_attention_layer,
            self.predict_depth_mult,
        )

    def _predict(
        self,
        gene_pos,
        expression,
        depth,
        image=None,
        region_image=None,
        tissue_image=None,
        predict_mode="none",
        pred_embedding=[],
        get_attention_layer=[],
        depth_mult=6,
        keep_output=True,
        max_size_in_mem=300_000,
    ):
        """
        @see predict_step will save output of predict in multiple self variables

        - embs: the cell embeddings (means from label specific embeddings given by self.pred_embedding)
        - pred: the predicted cell classes
        - pos: the genes used
        - expr_pred: the expression prediction. [mean, disp, zero_logits]
        - mean_attn: the mean attention across cells for the given layer (in self.get_attention_layer)

        these will be finalized in self.on_predict_epoch_end()

        Args:
            @see training_step
            other important arguments:
            keep_output (bool, optional): whether to keep the output in memory. Defaults to True.
            self.get_attention_layer (list, optional): the layers to get the attention from. Defaults to [].
            self.pred_embedding (list, optional): the classes to predict. Defaults to [].

        """
        if predict_mode == "none":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                region_image=region_image,
                tissue_image=tissue_image,
                depth_mult=expression.sum(1),
                full_depth=depth,
                get_attention_layer=get_attention_layer,
                do_class=True,
            )
            if len(get_attention_layer) > 0:
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]
        elif predict_mode == "denoise":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                region_image=region_image,
                tissue_image=tissue_image,
                depth_mult=expression.sum(1) * depth_mult,
                full_depth=depth * depth_mult,
                get_attention_layer=get_attention_layer,
                do_class=True,
            )
            if len(get_attention_layer) > 0:
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]
        elif predict_mode == "generate":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                region_image=region_image,
                tissue_image=tissue_image,
                full_depth=depth,
                do_mvc=False,
                do_class=False,
            )
            cell_embs = output["cell_embs"]
            output = self._generate(
                output["cell_embs"],
                gene_pos,
                image=image,
                region_image=region_image,
                tissue_image=tissue_image,
                full_depth=None,  # otherwise we have 2 depths passed
                depth_mult=expression.sum(1),
                do_class=self.do_cls,
                do_mvc=False,
            )
        else:
            raise ValueError(
                "predict_mode needs to be one of ['none', 'denoise', 'generate']"
            )

        if len(pred_embedding) == 0:
            pred_embedding = self.classes
        ind = [self.classes.index(i) + 2 for i in pred_embedding]
        if not keep_output:
            return {
                "embs": torch.mean(cell_embs[:, ind, :], dim=1),
                "class": (
                    torch.stack(
                        [
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            for clsname in self.classes
                        ]
                    ).transpose(0, 1)
                    if len(self.classes) > 0
                    else None
                ),
                "pos": gene_pos,
                "expr": (
                    [output["mean"], output["disp"], output["zero_logits"]]
                    if "disp" in output
                    else [output["mean"]]
                ),
            }
        if self.embs is None:
            self.embs = torch.mean(cell_embs[:, ind, :], dim=1)
            # self.embs = output["cls_output_" + "cell_type_ontology_term_id"]
            self.pred = (
                torch.stack(
                    [
                        (
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            if not self.keep_all_cls_pred
                            else output["cls_output_" + clsname]
                        )
                        for clsname in self.classes
                    ]
                ).transpose(0, 1)
                if len(self.classes) > 0
                else None
            )
            self.pos = gene_pos
            self.expr_pred = (
                [output["mean"], output["disp"], output["zero_logits"]]
                if "disp" in output
                else [output["mean"]]
            )
        else:
            self.embs = torch.cat(
                # [self.embs, output["cls_output_" + "cell_type_ontology_term_id"]]
                [self.embs, torch.mean(cell_embs[:, ind, :], dim=1)]
            )
            self.pred = torch.cat(
                [
                    self.pred,
                    (
                        torch.stack(
                            [
                                (
                                    torch.argmax(output["cls_output_" + clsname], dim=1)
                                    if not self.keep_all_cls_pred
                                    else output["cls_output_" + clsname]
                                )
                                for clsname in self.classes
                            ]
                        ).transpose(0, 1)
                        if len(self.classes) > 0
                        else None
                    ),
                ],
            )
            self.pos = torch.cat([self.pos, gene_pos])
            self.expr_pred = (
                [
                    torch.cat([self.expr_pred[0], output["mean"]]),
                    torch.cat([self.expr_pred[1], output["disp"]]),
                    torch.cat([self.expr_pred[2], output["zero_logits"]]),
                ]
                if "disp" in output
                else [torch.cat([self.expr_pred[0], output["mean"]])]
            )
        if self.embs is not None:
            if self.embs.shape[0] > max_size_in_mem:
                print("logging")
                self.log_adata(name="predict_part_" + str(self.counter))
                self.counter += 1
                self.pos = None
                self.expr_pred = None
                self.pred = None
                self.embs = None

    def on_predict_epoch_end(self):
        """@see pl.LightningModule will"""
        if self.pos.shape[0] < 100:
            return
        if self.pred_log_adata:
            print("adding on disk")
            return self.log_adata(name="predict_part_" + str(self.counter))

    def get_cell_embs(self, layer_output):
        """
        get_cell_embs

        Args:
            layer_output (Tensor): The output tensor from a layer in the model.

        Raises:
            ValueError: Raised when an unknown cell embedding style is encountered.

        Returns:
            Tensor: The cell embeddings tensor.
        """
        if self.cell_emb_style == "cls" and self.classes is not None:
            # (minibatch, embsize)
            cell_emb = layer_output[:, : 2 + len(self.classes)]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth_mult: Tensor,
        image: Optional[Tensor] = None,
        region_image: Optional[Tensor] = None,
        tissue_image: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        tp: Optional[Tensor] = None,
        gen_iters: int = 1,
        **decoder_kwargs,
    ):
        """
        _generate given cell_embeddings, generate an expression profile

        the goal was to iterate multiple times,
        to create a trajectory and reach a certain state
        should call forward multiple times

        Args:
            cell_emb(:obj:`Tensor`): A tensor representing cell embeddings. It has a shape of (minibatch, embsize).
            src(:obj:`Tensor`): A tensor representing the source data. It has a shape of (minibatch, seq_len).
            values(:obj:`Tensor`): An optional tensor representing the values. It has a shape of (minibatch, seq_len).
            gen_iters(:obj:`int`): An integer representing the number of generation iterations.
            classes(:obj:`Tensor`): An optional tensor representing the classes. It has a shape of (batch,).
        """
        if tp is not None:
            tp = tp / gen_iters
        for i in range(gen_iters):
            encoding = self._encoder(
                cell_embs=cell_embs,
                gene_pos=gene_pos,
                full_depth=full_depth,
                timepoint=tp * (i + 1) if tp is not None else None,
            )  # (minibatch, seq_len, embsize)
            transformer_output = self.transformer(encoding)
            transformer_output = self._fuse(transformer_output, image, region_image, tissue_image, mask=None)
            cell_embs = self.get_cell_embs(transformer_output)
        output = self._decoder(
            transformer_output, depth_mult=depth_mult, **decoder_kwargs
        )
        return output  # (minibatch, seq_len)

    def get_adata(self, doplot=False):
        """
        Generates an AnnData object containing embeddings, class information, predictions, and attention weights.
        This method creates an AnnData object using the model's embeddings, classes, predictions, and attention weights.
        It can optionally generate and log visualization plots to TensorBoard and/or Weights & Biases.
        Parameters
        ----------
        doplot : bool, optional
            If True, generates and logs UMAP visualizations.
            If None, uses the model's default doplot setting.
            Default is False.
        Returns
        -------
        anndata.AnnData
            An AnnData object containing embeddings, class information, predictions, and attention weights.
            The object can be used for downstream analysis and visualization.
        """
        if doplot is None:
            doplot = self.doplot
        adata, fig = utils.make_adata(
            self.embs,
            self.classes,
            self.pred if not self.keep_all_cls_pred else None,
            self.attn.get(),
            self.global_step,
            self.label_decoders,
            self.labels_hierarchy,
            gtclass=None,
            name=None,
            mdir=None,
            doplot=doplot,
        )
        if doplot:
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata

    def log_adata(self, gtclass=None, name=""):
        """
        log_adata will log an adata from predictions.
        It will log to tensorboard and wandb if available

        see @utils.log_adata
        """
        try:
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"
        except:
            mdir = "data/"
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        adata, fig = utils.make_adata(
            self.embs,
            self.classes,
            self.pred if not self.keep_all_cls_pred else None,
            self.attn.get(),
            self.global_step,
            self.label_decoders,
            self.labels_hierarchy,
            gtclass,
            self.name + "_" + name + "_" + str(self.global_rank),
            mdir,
            self.doplot,
        )
        if self.doplot:
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata

    def _predict_denoised_expression(self, gene_pos, expression, depth):
        """
        Args:
            gene_pos (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            expression (:obj:`Tensor`): token values, shape [batch_size, seq_len]

        Returns:
            dict of output Tensors.
        """
        output = self.forward(gene_pos, expression, full_depth=depth)
        return output
