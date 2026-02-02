import torch
import os

# VERSION = os.getenv("USE_VERSION", "1_0")
# V2_0 = VERSION == "2_0"

# RETAIN_TOKN = float(os.getenv("RETAIN_TOKN", "0.58"))

sparse_token_list_46 = {5: 0.7, 10: 0.5, 20: 0.3}  # (4*1+3*0.7+9*0.5+20*0.3)/36=0.46
sparse_token_list_58 = {5: 0.8, 10: 0.5, 20: 0.4}  # (6*1+5*0.8+10*0.5+15*0.4)/36=0.58
sparse_token_list_75 = {17: 0.5}  # (18*1+18*0.5)/36=0.75
sparse_token_list_66 = {17: 0.33}  # (18*1+18*0.33)/36=0.66

# sparse_token_list_192 = [300, 200, 110] if not V2_0 else [300, 200, 118]  # 2*576  4*300 10*200  16*110
# sparse_token_list_128 = [303, 110, 36] if not V2_0 else [238, 108, 60]
# sparse_token_list_96 = [238, 48, 26] if not V2_0 else [246, 54, 28]
# sparse_token_list_64 = [66, 30, 17] if not V2_0 else [66, 34, 20]

sparse_token_dict = {
    # 192: sparse_token_list_192,
    # 128: sparse_token_list_128,
    # 96: sparse_token_list_96,
    # 64: sparse_token_list_64,
    0.46: sparse_token_list_46,
    0.58: sparse_token_list_58,
    0.75: sparse_token_list_75,
    0.66: sparse_token_list_66,
}


def attn_postprocess_topk(self_attn_weights, text_range, vision_range, t_token_idx, layer_idx, budgets):
    """
    self_attn_weights: [B, H, L, L]
    t_token_idx: 选择的text token索引[N]

    """
    self_attn_weights = self_attn_weights.mean(1)  # B, L[Q], L[K]

    relation_vis_text = self_attn_weights[:, t_token_idx, vision_range[0] : vision_range[1]]  # B, L[text], L[vision]

    relation_vis_text = relation_vis_text.mean(1)  # B, L[vision]

    s_flag = True  # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[budgets]
    v_token_num = vision_range[1] - vision_range[0]
    token_num = int(sparse_token_list[layer_idx] * v_token_num)
    reduce_token_num = v_token_num - token_num
    # reduce_token_num = v_token_num - sparse_token_list[layer_dict[layer_idx]]
    # new_vision_range = (vision_range[0], vision_range[0] + sparse_token_list[layer_dict[layer_idx]])
    new_vision_range = (vision_range[0], vision_range[0] + token_num)
    new_text_range = (text_range[0] - reduce_token_num, text_range[1] - reduce_token_num)
    if v_token_num != 0:
        # mask = torch.zeros_like(relation_vis, dtype=bool)
        # _, indices = torch.topk(
        #     relation_vis_text, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1
        # )
        _, indices = torch.topk(relation_vis_text, min(token_num, v_token_num - 1), dim=1)
        indices = indices + vision_range[0]
        indices = indices[0].tolist()
        indices.sort()
        # mask[0][indices] = 1
    else:
        # mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False

    return indices, s_flag, relation_vis_text, new_vision_range, new_text_range


def select_attn_head_by_sum(self_attn_weights, t_token_idx, v_token_start, text_token_start):
    # [1,28,token_num,token_num] -> [28,text_token_num,visual_token_num]
    each_head_text_to_visual_attn = self_attn_weights[0][:, t_token_idx, v_token_start:text_token_start]
    # [28,text_token_num,visual_token_num] -> [28]
    sum_attn_per_head = each_head_text_to_visual_attn.sum((1, 2))
    select_attn_head_idx = sum_attn_per_head.topk(14)[1]

    return self_attn_weights[:, select_attn_head_idx, :, :][:, :, :]


if __name__ == "__main__":
    self_attn_weights, vision_range, text_range = torch.rand(1, 16, 1084, 1084), (100, 700), (700, 800)
    t_token_idx = torch.tensor([700, 701, 702])
    layer_idx = 2
    indices, s_flag, relation_vis_text = attn_postprocess_topk(
        self_attn_weights, text_range, vision_range, t_token_idx, layer_idx
    )
    print(indices, s_flag, relation_vis_text)
