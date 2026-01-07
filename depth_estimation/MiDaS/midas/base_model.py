import torch


class BaseModel(torch.nn.Module):
    # def load(self, path):
    #     """Load model from file.
    #
    #     Args:
    #         path (str): file path
    #     """
    #     parameters = torch.load(path, map_location=torch.device('cpu'))
    #
    #     if "optimizer" in parameters:
    #         parameters = parameters["model"]
    #
    #     self.load_state_dict(parameters)

    def load(self, path):
        state = torch.load(path, map_location="cpu")

        # 兼容多种打包格式
        if isinstance(state, dict) and "optimizer" in state and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 过滤不会影响推理的 buffer/辅助表（不同 timm 版本常见）
        DROP = (
            "attn_mask",
            "relative_position_index",
            "relative_position_bias_table",
            "relative_coords_table",
            "relative_coords",          # 防止不同命名
        )
        state = {k: v for k, v in state.items() if all(s not in k for s in DROP)}

        # 宽松加载：允许缺失/多余的非关键键
        missing, unexpected = self.load_state_dict(state, strict=False)

        # 打印信息便于自检（可保留或注释掉）
        print("[MiDaS load] Missing (trimmed):", [m for m in missing if "head" not in m])
        print("[MiDaS load] Unexpected:", unexpected)
