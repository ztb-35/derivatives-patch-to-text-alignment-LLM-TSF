from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, PatchEmbedding_Slope_Acc
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.beta_S = configs.beta_S
        self.beta_A = configs.beta_A
        self.patch2embed = nn.Linear(configs.d_model, self.d_llm)
        self.return_attn = configs.return_attn
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.patch_embedding_Slope_Acc = PatchEmbedding_Slope_Acc(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.mapping_layer_1 = nn.Linear(self.vocab_size, 5)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.reprogramming_layerV2 = ReprogrammingLayerV2(configs.llm_dim, configs.n_heads, 0.1, False, "mean")
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'abnormally analysis':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, seq_trend, seq_seasonal, seq_resid, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, slope_loss, acc_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], slope_loss, acc_loss
        elif self.task_name == 'abnormally analysis':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_1_D = (x_enc.squeeze(-1)[:, 1:] - x_enc.squeeze(-1)[:, :-1])
        x_1_D = torch.nn.functional.pad(x_1_D, (1, 0), mode="replicate").unsqueeze(-1)#first order derivative
        x_2_D = (x_1_D.squeeze(-1)[:, 1:] - x_1_D.squeeze(-1)[:, :-1])#second order derivative
        x_2_D = torch.nn.functional.pad(x_2_D, (1, 0), mode="replicate").unsqueeze(-1)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        #soft labels for trend and acceleration
        prompt_slope = self.tokenizer(' flat increase decrease', return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        slope_embeddings = self.llm_model.get_input_embeddings()(prompt_slope.to(x_enc.device))  # (batch, prompt_token, dim)
        prompt_acc = self.tokenizer(' stable speeding slowing', return_tensors="pt", padding=True, truncation=True,
                                      max_length=2048).input_ids#acceleration
        acc_embeddings = self.llm_model.get_input_embeddings()( prompt_acc.to(x_enc.device))  # (batch, prompt_token, dim)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_1_D = x_1_D.permute(0, 2, 1).contiguous()
        x_2_D = x_2_D.permute(0, 2, 1).contiguous()
        #enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float))#add the prompt embedding with the patching, add the trend, seasonality to patch
        enc_1_D_out, slope = self.patch_embedding_Slope_Acc(x_1_D.to(torch.float))#patching ofr derivatives
        enc_2_D_out, acc = self.patch_embedding_Slope_Acc(x_2_D.to(torch.float))
        # slope soft labels
        p_inc = torch.sigmoid(self.beta_S * slope)#smooth mapping slope to probability
        p_dec = torch.sigmoid(-self.beta_S * slope)
        p_flat = torch.exp(-0.5 * (self.beta_S * slope) ** 2)
        P = torch.stack([p_flat, p_inc, p_dec], dim=-1)  # (BN, L, 3)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        # degree soft labels
        p_speed = torch.sigmoid(self.beta_A * (acc*slope))
        p_slow = torch.sigmoid(-self.beta_A * (acc*slope))
        p_linear = torch.exp(-0.5 * (self.beta_S * (acc*slope)) ** 2)
        #acc*slope>0 means fast, acc*slope<0 means slow
        Q = torch.stack([p_linear, p_speed, p_slow], dim=-1)  # (BN, L, 2)
        #get the slope and acceleration prompt for each patch of input
        patches_prompt = self.build_patch_prompts(P, Q)
        # patches_prompt: list[list[str]] with outer len = B and inner len = L
        B = len(patches_prompt)
        L = len(patches_prompt[0])

        # 1) flatten to list[str]
        flat_patches_prompts = [patches_prompt[i][j] for i in range(B) for j in range(L)]
        flat_patches_prompts = self.tokenizer(flat_patches_prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048).input_ids
        flat_patches_prompts_embeddings = self.llm_model.get_input_embeddings()(flat_patches_prompts.to(x_enc.device))
        patches_prompts_embeddings = flat_patches_prompts_embeddings.view(B,L,self.d_llm,-1).permute(0,1,3,2)
        #concatenate patches with their corresponding prompt
        enc_out = enc_out.unsqueeze(2)
        #enc_out = torch.cat([enc_out, patches_prompts_embeddings], dim=2).mean(dim=2).squeeze(2)
        enc_out = self.patch2embed(enc_out)
        enc_out = torch.cat([enc_out, patches_prompts_embeddings], dim=2)
        #make the context source embedding consists of slope, acceleration and learned prototypes
        context_embedding = torch.cat([slope_embeddings.squeeze(), acc_embeddings.squeeze(), source_embeddings], dim=0)
        # acceleration
        #enc_out = self.reprogramming_layer(enc_out, context_embedding, context_embedding)
        enc_out_dict = self.reprogramming_layerV2(
            target_tokens=enc_out,
            key_bank=context_embedding,  # (S, d_llm)
            value_bank=context_embedding,  # (S, d_llm)
            bank_mask=None  # or a mask if you have padded slots
        )
        if self.return_attn:
            enc_out = enc_out[0]
            attn_map = enc_out[1]
        else:
            enc_out = enc_out_dict
        #compute the slope and acceleration
        slope_logit = torch.einsum("bld,kd->blk", enc_out, slope_embeddings.squeeze())
        acc_logit = torch.einsum("bld,kd->blk", enc_out, acc_embeddings.squeeze())
        slope_logprobs = torch.log_softmax(slope_logit, dim=-1)
        acc_logprobs = torch.log_softmax(acc_logit, dim=-1)
        slope_loss = (P * (torch.log(P + 1e-8) - slope_logprobs)).sum(dim=-1).mean()
        acc_loss = (Q * (torch.log(Q + 1e-8) - acc_logprobs)).sum(dim=-1).mean()
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out_hidden = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out_hidden[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        if self.task_name == 'abnormally analysis':
            return dec_out_hidden
        else:
            return dec_out, slope_loss, acc_loss

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def build_patch_prompts(self, P_slope, Q_accel):
        B, L, _ = P_slope.shape
        s_idx = P_slope.argmax(dim=-1)  # (B, L): 0=flat, 1=inc, 2=dec
        a_idx = Q_accel.argmax(dim=-1)  # (B, L): 0=linear, 1=speeding, 2=slowing

        # 3x3 phrase table: rows = slope [flat, inc, dec], cols = accel [linear, speeding, slowing]
        phrases = [
            [" the trend of this patch series is flat and stable",  # flat + linear
             " the trend of this patch series is flat and speeding up at local minimum",  # flat + speeding
             " the trend of this patch series is flat and slowing down at local maximum"  # flat + slowing
             ],
            [" the trend of this patch series is increasing at a stable rate",  # inc + linear
             " the trend of this patch series is increasing and speeding up",  # inc + speeding
             " the trend of this patch series is increasing but slowing down"  # inc + slowing
             ],
            [" the trend of this patch series is decreasing at a stable rate",  # dec + linear
             " the trend of this patch series is decreasing and speeding up downward",  # dec + speeding
             " the trend of this patch series is decreasing but slowing its fall"  # dec + slowing
             ],
        ]

        prompts = [[phrases[s_idx[i, j].item()][a_idx[i, j].item()]
                    for j in range(L)] for i in range(B)]
        return prompts  # list[list[str]] with shape (B, L)
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class ReprogrammingLayerV2(nn.Module):
    """
    Cross-attention from per-patch token sequence (patch token + prompt tokens) to a semantic bank.

    Inputs:
      target_tokens : (B, L, Tq, d)     # queries per patch (patch token + its prompt tokens), d = d_llm
      key_bank      : (S, d)  or (B, S, d)
      value_bank    : (S, d)  or (B, S, d)  (if None -> same as key_bank)
      bank_mask     : (S,) or (B, S)  with 1=keep, 0=mask   (optional)

    Returns:
      y : (B, L, d)                     # fused per-patch representation (pooled over Tq)
      A_patch_bank (optional): (B, L, S)  # attention map averaged over heads and Tq
    """

    def __init__(self, d_llm: int, n_heads: int, attention_dropout: float = 0.1, return_attn: bool = False,
                 pool: str = "mean"):
        super().__init__()
        assert d_llm % n_heads == 0, "d_llm must be divisible by n_heads"
        assert pool in {"mean", "first"}, "pool must be 'mean' or 'first'"

        self.n_heads = n_heads
        self.d_head  = d_llm // n_heads
        self.scale   = 1.0 / sqrt(self.d_head)
        self.return_attn = return_attn
        self.pool = pool

        self.query_projection = nn.Linear(d_llm, n_heads * self.d_head)
        self.key_projection   = nn.Linear(d_llm, n_heads * self.d_head)
        self.value_projection = nn.Linear(d_llm, n_heads * self.d_head)
        self.out_projection   = nn.Linear(n_heads * self.d_head, d_llm)

        self.ln_q = nn.LayerNorm(d_llm)
        self.ln_kv = nn.LayerNorm(d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self,
                target_tokens,           # (B, L, Tq, d)
                key_bank,                # (S, d) or (B, S, d)
                value_bank,# (S, d) or (B, S, d)
                bank_mask  # (S,) or (B, S)
                ):
        B, L, Tq, d = target_tokens.shape
        if value_bank is None:
            value_bank = key_bank

        # ---- Q from target tokens ----
        Q_in = self.ln_q(target_tokens)                                  # (B, L, Tq, d)
        H, E = self.n_heads, self.d_head
        Q = self.query_projection(Q_in).view(B, L, Tq, H, E)              # (B, L, Tq, H, E)

        # ---- K/V from bank (shared or batched) ----
        if key_bank.dim() == 2:
            # shared
            S = key_bank.size(0)
            K_in = self.ln_kv(key_bank).unsqueeze(0).expand(B, S, d)      # (B, S, d)
            V_in = self.ln_kv(value_bank).unsqueeze(0).expand(B, S, d)    # (B, S, d)
            bank_mask_b = None if bank_mask is None else bank_mask.unsqueeze(0).expand(B, -1)  # (B, S)
        else:
            raise ValueError("key_bank must be (S,d)")

        K = self.key_projection(K_in).view(B, S, H, E)                    # (B, S, H, E)
        V = self.value_projection(V_in).view(B, S, H, E)                  # (B, S, H, E)

        # ---- scores: (B, H, L, Tq, S) ----
        scores = torch.einsum("blthe,bshe->bhlts", Q, K) * self.scale     # (B, H, L, Tq, S)

        # ---- optional mask over bank ----
        if bank_mask_b is not None:
            m = bank_mask_b.unsqueeze(1).unsqueeze(2)                     # (B, 1, 1, S)
            scores = scores.masked_fill(m == 0, float("-inf"))

        # ---- attention & weighted sum ----
        A = torch.softmax(scores, dim=-1)                                 # (B, H, L, Tq, S)
        A = self.dropout(A)
        out = torch.einsum("bhlts,bshe->blthe", A, V)                     # (B, L, Tq, H, E)

        # ---- pool over query tokens (Tq) to get one output per patch ----

        # ---- pool over query tokens (Tq) to get one output per patch ----
        if self.pool == "first":
            out_pooled = out[:, :, 0, :, :]  # (B, L, H, E)  # use the "patch token"
        else:
            out_pooled = out.mean(dim=2)  # (B, L, H, E)  # mean over patch+prompt tokens

        y = self.out_projection(out_pooled.reshape(B, L, H * E))          # (B, L, d)

        if self.return_attn:
            # average over heads and Tq to get a clean (B, L, S) heatmap
            A_patch_bank = A.mean(dim=1).mean(dim=3)                      # (B, L, S)
            return y, A_patch_bank
        return y


